#!/usr/bin/env python3
"""Evaluate prebuilt prediction JSONL files directly with SWE-bench harness."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
PIPELINE_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PIPELINE_DIR / "1_patch_mutate_and_eval"))

from swebench_eval import evaluate_prediction_jobs


SUPPORTED_MUTATIONS = ("gs", "unwrap", "unsafe", "panic")
KNOWN_DATASETS = (
    "SWE-bench/SWE-bench_Multilingual",
    "TuringEnterprises/SWE-Bench-plus-plus",
    "ByteDance-Seed/Multi-SWE-bench",
)


def _slug(text: str) -> str:
    cleaned = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_")


SLUG_TO_DATASET = {_slug(name): name for name in KNOWN_DATASETS}


def _count_predictions(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            # Ensure line is valid JSON; fail fast on broken artifacts.
            json.loads(line)
            count += 1
    return count


def _split_filename(stem: str) -> Tuple[str, str] | None:
    for mutation in SUPPORTED_MUTATIONS:
        suffix = f"_{mutation}"
        if stem.endswith(suffix):
            dataset_slug = stem[: -len(suffix)]
            if dataset_slug:
                return dataset_slug, mutation
    return None


def _parse_mutations(raw: str) -> set[str]:
    out = set()
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        if token == "panic!":
            token = "panic"
        if token not in SUPPORTED_MUTATIONS:
            raise ValueError(f"Unsupported mutation: {token}")
        out.add(token)
    return out or set(SUPPORTED_MUTATIONS)


def _parse_benchmarks(raw: str | None) -> set[str] | None:
    if not raw:
        return None
    out: set[str] = set()
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        if token in KNOWN_DATASETS:
            out.add(token)
            continue
        # Accept short aliases.
        lowered = token.lower()
        if lowered in {"multilingual", "swe-bench_multilingual", "swebench_multilingual"}:
            out.add("SWE-bench/SWE-bench_Multilingual")
            continue
        if lowered in {"plus-plus", "plusplus", "swe-bench_plus-plus", "swebench++"}:
            out.add("TuringEnterprises/SWE-Bench-plus-plus")
            continue
        if lowered in {"multi-swe", "multi-swe-bench"}:
            out.add("ByteDance-Seed/Multi-SWE-bench")
            continue
        raise ValueError(f"Unsupported benchmark selector: {token}")
    return out


def build_jobs_from_predictions_dir(
    predictions_dir: Path,
    *,
    allowed_mutations: set[str],
    allowed_benchmarks: set[str] | None,
) -> Tuple[List[dict], List[str]]:
    jobs: List[dict] = []
    skipped: List[str] = []

    for path in sorted(predictions_dir.glob("*_mutated.jsonl")):
        parsed = _split_filename(path.stem.replace("_mutated", ""))
        if parsed is None:
            skipped.append(f"{path.name}: cannot parse dataset/mutation from filename")
            continue
        dataset_slug, mutation = parsed
        benchmark = SLUG_TO_DATASET.get(dataset_slug)
        if not benchmark:
            skipped.append(f"{path.name}: unknown dataset slug `{dataset_slug}`")
            continue
        if mutation not in allowed_mutations:
            continue
        if allowed_benchmarks and benchmark not in allowed_benchmarks:
            continue

        num_predictions = _count_predictions(path)
        jobs.append(
            {
                "benchmark": benchmark,
                "mutation": mutation,
                "predictions_path": str(path),
                "num_predictions": num_predictions,
            }
        )

    jobs.sort(key=lambda r: (r["benchmark"], r["mutation"]))
    return jobs, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-dir",
        default="data/mutated_patches",
        type=Path,
        help="Directory containing *_mutated.jsonl prediction artifacts.",
    )
    parser.add_argument(
        "--mutations",
        default="gs,unwrap,unsafe,panic",
        help="Comma-separated mutation filters.",
    )
    parser.add_argument(
        "--benchmarks",
        default=None,
        help=(
            "Optional comma-separated benchmark filters "
            "(dataset names or aliases: multilingual, plusplus, multi-swe)."
        ),
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

    allowed_mutations = _parse_mutations(args.mutations)
    allowed_benchmarks = _parse_benchmarks(args.benchmarks)

    jobs, skipped = build_jobs_from_predictions_dir(
        args.predictions_dir,
        allowed_mutations=allowed_mutations,
        allowed_benchmarks=allowed_benchmarks,
    )

    print(f"Predictions dir: {args.predictions_dir}")
    print(f"Discovered jobs: {len(jobs)}")
    if skipped:
        print(f"Skipped files: {len(skipped)}")
        for item in skipped:
            print(f"  - {item}")

    for job in jobs:
        print(
            f"  - {job['benchmark']} [{job['mutation']}] "
            f"predictions={job['num_predictions']} file={job['predictions_path']}"
        )

    if not args.run_eval:
        print("Dry run only. Use --run-eval to execute swebench harness.")
        return 0

    summaries = evaluate_prediction_jobs(
        jobs,
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
