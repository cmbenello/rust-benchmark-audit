#!/usr/bin/env python3
"""Mutation + evaluation harness runner for unified benchmark instances."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


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


def _slug(text: str) -> str:
    cleaned = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_")


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


def _parse_predictions_jsonl_lines(lines: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        row = json.loads(line)
        instance_id = row.get("instance_id")
        model_patch = row.get("model_patch")
        if isinstance(instance_id, str) and isinstance(model_patch, str):
            out[instance_id] = model_patch
    return out


def _load_predictions_from_path(path: Path) -> Dict[str, str]:
    with path.open("r", encoding="utf-8") as f:
        return _parse_predictions_jsonl_lines(f.readlines())


def _load_predictions_from_commit(commit: str, rel_path: str) -> Dict[str, str] | None:
    proc = subprocess.run(
        ["git", "show", f"{commit}:{rel_path}"],
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        return None
    return _parse_predictions_jsonl_lines(proc.stdout.splitlines())


def apply_external_prediction_overrides(
    mutated_instances: List[dict],
    *,
    external_predictions_dir: Path | None = None,
    external_predictions_commit: str | None = None,
    require_external_predictions: bool = False,
) -> dict:
    if not external_predictions_dir and not external_predictions_commit:
        for row in mutated_instances:
            row.setdefault("mutation_source", "generated")
        return {
            "pairs_total": 0,
            "pairs_external": 0,
            "rows_replaced": 0,
            "missing_pairs": [],
        }

    pair_to_patch_map: Dict[tuple[str, str], Dict[str, str]] = {}
    pair_to_source: Dict[tuple[str, str], str] = {}
    pairs = sorted(
        {
            (str(row.get("hf_bm", "")), str(row.get("mutation", "")))
            for row in mutated_instances
            if row.get("hf_bm") and row.get("mutation")
        }
    )

    missing_pairs = []
    for benchmark, mutation in pairs:
        filename = f"{_slug(benchmark)}_{_slug(mutation)}_mutated.jsonl"
        patch_map = None

        if external_predictions_commit:
            rel_path = f"data/mutated_patches/{filename}"
            patch_map = _load_predictions_from_commit(external_predictions_commit, rel_path)
            if patch_map is not None:
                pair_to_source[(benchmark, mutation)] = f"commit:{external_predictions_commit}"

        if patch_map is None and external_predictions_dir:
            path = external_predictions_dir / filename
            if path.exists():
                patch_map = _load_predictions_from_path(path)
                pair_to_source[(benchmark, mutation)] = f"path:{path}"

        if patch_map is None:
            missing_pairs.append((benchmark, mutation))
            continue

        pair_to_patch_map[(benchmark, mutation)] = patch_map

    if require_external_predictions and missing_pairs:
        missing_text = ", ".join(f"{bm}[{mut}]" for bm, mut in missing_pairs)
        raise RuntimeError(f"Missing required external prediction files for: {missing_text}")

    rows_replaced = 0
    for row in mutated_instances:
        benchmark = str(row.get("hf_bm", ""))
        mutation = str(row.get("mutation", ""))
        key = (benchmark, mutation)
        patch_map = pair_to_patch_map.get(key)
        if not patch_map:
            row.setdefault("mutation_source", "generated")
            continue

        instance_id = str(row.get("instance_id", ""))
        external_patch = patch_map.get(instance_id)
        if not external_patch:
            row.setdefault("mutation_source", "generated")
            continue

        row["diff"] = external_patch
        row["mutation_count"] = 1
        row["policy_count_results"] = count_from_bm_diff(external_patch)
        row["mutation_source"] = pair_to_source.get(key, "external")
        rows_replaced += 1

    return {
        "pairs_total": len(pairs),
        "pairs_external": len(pair_to_patch_map),
        "rows_replaced": rows_replaced,
        "missing_pairs": [f"{bm}[{mut}]" for bm, mut in missing_pairs],
    }


def build_mutated_instances(
    instances: List[dict],
    mutations: List[str],
    *,
    mutation_style: str = "heuristic",
) -> List[dict]:
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
            mutated_patch, mutation_count = mutate_patch_text(
                patch,
                mutation,
                style=mutation_style,
            )
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
    parser.add_argument(
        "--mutation-style",
        default="heuristic",
        choices=["heuristic", "adversarial"],
        help="Mutation strategy for non-gs variants.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mutated-out-jsonl", default="results/mutated_instances.jsonl", type=Path)
    parser.add_argument("--policy-out-jsonl", default="results/policy_check_results.jsonl", type=Path)
    parser.add_argument("--predictions-dir", default="data/mutated_patches", type=Path)
    parser.add_argument(
        "--external-predictions-dir",
        default=None,
        type=Path,
        help="Directory containing prebuilt prediction JSONL files to override generated mutations.",
    )
    parser.add_argument(
        "--external-predictions-commit",
        default=None,
        help=(
            "Git commit containing prebuilt prediction JSONL files under "
            "data/mutated_patches/*.jsonl to override generated mutations."
        ),
    )
    parser.add_argument(
        "--require-external-predictions",
        action="store_true",
        help="Fail if any benchmark+mutation file is missing from external prediction sources.",
    )
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--eval-output-dir", default="results/harness_eval", type=Path)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--docker-host", default=None)
    parser.add_argument(
        "--include-known-incompatible",
        action="store_true",
        help="Include benchmarks known to error with current swebench/datasets versions.",
    )
    args = parser.parse_args()

    instances = _load_instances_jsonl(args.instances_jsonl)
    if args.limit:
        instances = instances[: args.limit]

    mutations = _normalize_mutations(args.mutations)
    mutated_instances = build_mutated_instances(
        instances,
        mutations,
        mutation_style=args.mutation_style,
    )
    override_stats = apply_external_prediction_overrides(
        mutated_instances,
        external_predictions_dir=args.external_predictions_dir,
        external_predictions_commit=args.external_predictions_commit,
        require_external_predictions=args.require_external_predictions,
    )
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
    print(f"Mutation style: {args.mutation_style}")
    if args.external_predictions_commit or args.external_predictions_dir:
        print(
            "External override: "
            f"pairs={override_stats['pairs_external']}/{override_stats['pairs_total']} "
            f"rows_replaced={override_stats['rows_replaced']}"
        )
        missing_pairs = override_stats.get("missing_pairs") or []
        if missing_pairs:
            print("Missing external pairs: " + ", ".join(missing_pairs))
    print(f"Prediction jobs: {len(prediction_jobs)}")
    print(f"Mutated output: {args.mutated_out_jsonl}")
    print(f"Policy output: {args.policy_out_jsonl}")

    if args.run_eval:
        summaries = evaluate_prediction_jobs(
            prediction_jobs,
            max_workers=args.max_workers,
            output_dir=args.eval_output_dir,
            fail_fast=args.fail_fast,
            docker_host=args.docker_host,
            skip_known_incompatible=not args.include_known_incompatible,
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
