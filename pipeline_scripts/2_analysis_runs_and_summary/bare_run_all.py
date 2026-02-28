#!/usr/bin/env python3
"""Mutation + evaluation harness runner for unified benchmark instances."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List


# Allow imports from the refactored pipeline directory tree.
SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PIPELINE_DIR / "1_patch_mutate_and_eval"))
sys.path.insert(0, str(PIPELINE_DIR / "0_data_construction"))

from mutate_patch import mutate_patch_text
from policy_checks import count_from_bm_diff
from swebench_eval import create_predictions_from_mutated_instances, evaluate_prediction_jobs


BENCHMARK_NAMES = {
    "swe-bench_plus-plus": "TuringEnterprises/SWE-Bench-plus-plus",
    "swe-bench_multilingual": "SWE-bench/SWE-bench_Multilingual",
    "multi-swe-bench": "ByteDance-Seed/Multi-SWE-bench",
}

DEFAULT_MUTATIONS = ("gs", "unwrap", "unsafe", "panic")


def _load_instances_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _normalize_mutations(mutations_arg: str) -> List[str]:
    out = []
    for raw in mutations_arg.split(","):
        value = raw.strip()
        if not value:
            continue
        if value == "panic!":
            value = "panic"
        if value not in DEFAULT_MUTATIONS:
            raise ValueError(f"Unsupported mutation: {value}")
        out.append(value)
    if "gs" not in out:
        out.insert(0, "gs")
    return out


def _get_patch_text(instance: dict) -> str:
    patch = instance.get("fix_patch")
    if isinstance(patch, str) and patch.strip():
        return patch
    patch = instance.get("patch")
    if isinstance(patch, str) and patch.strip():
        return patch
    return ""


def build_mutated_instances(instances: List[dict], mutations: List[str]) -> List[dict]:
    rows: List[dict] = []
    for instance in instances:
        source_benchmark = instance.get("source_benchmark")
        hf_bm = BENCHMARK_NAMES.get(source_benchmark)
        patch = _get_patch_text(instance)
        instance_id = instance.get("instance_id")
        if not instance_id or not patch or not hf_bm:
            continue

        base = {
            "instance_id": instance_id,
            "source_benchmark": source_benchmark,
            "hf_bm": hf_bm,
            "repo": instance.get("repo"),
            "org": instance.get("org"),
            "number": instance.get("number"),
        }

        if "gs" in mutations:
            gs_row = dict(base)
            gs_row["mutation"] = "gs"
            gs_row["diff"] = patch
            gs_row["mutation_count"] = 0
            gs_row["policy_count_results"] = count_from_bm_diff(patch)
            rows.append(gs_row)

        for mutation in mutations:
            if mutation == "gs":
                continue
            mutated_patch, mutation_count = mutate_patch_text(patch, mutation)
            mut_row = dict(base)
            mut_row["mutation"] = mutation
            mut_row["diff"] = mutated_patch
            mut_row["mutation_count"] = mutation_count
            mut_row["policy_count_results"] = count_from_bm_diff(mutated_patch)
            rows.append(mut_row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances-jsonl", default="data/instances_unified.jsonl", type=Path)
    parser.add_argument("--mutations", default="gs,unwrap,unsafe,panic")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mutated-out-jsonl", default="results/mutated_instances.jsonl", type=Path)
    parser.add_argument("--policy-out-jsonl", default="results/policy_check_results.jsonl", type=Path)
    parser.add_argument("--predictions-dir", default="data/mutated_patches", type=Path)
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--eval-output-dir", default="results/harness_eval", type=Path)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    instances = _load_instances_jsonl(args.instances_jsonl)
    if args.limit:
        instances = instances[: args.limit]

    mutations = _normalize_mutations(args.mutations)
    mutated_instances = build_mutated_instances(instances, mutations)
    _write_jsonl(args.mutated_out_jsonl, mutated_instances)
    _write_jsonl(
        args.policy_out_jsonl,
        [
            {
                "source_benchmark": row["source_benchmark"],
                "instance_id": row["instance_id"],
                "mutation": row["mutation"],
                "policy_count_results": row["policy_count_results"],
            }
            for row in mutated_instances
        ],
    )

    prediction_jobs = create_predictions_from_mutated_instances(
        mutated_instances,
        out_dir=args.predictions_dir,
    )

    print(f"Loaded instances: {len(instances)}")
    print(f"Mutated rows: {len(mutated_instances)}")
    print(f"Prediction jobs: {len(prediction_jobs)}")
    print(f"Mutated output: {args.mutated_out_jsonl}")
    print(f"Policy output: {args.policy_out_jsonl}")

    if args.run_eval:
        summaries = evaluate_prediction_jobs(
            prediction_jobs,
            max_workers=args.max_workers,
            output_dir=args.eval_output_dir,
            fail_fast=args.fail_fast,
        )
        failed = [row for row in summaries if row.get("returncode") != 0]
        print(f"Evaluation runs: {len(summaries)}")
        print(f"Evaluation failures: {len(failed)}")
        if failed:
            for row in failed:
                print(
                    f"  - {row['benchmark']} [{row['mutation']}] "
                    f"rc={row['returncode']} stderr={row['stderr_path']}"
                )
    else:
        print("Skipped harness evaluation (use --run-eval to execute swebench harness).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
