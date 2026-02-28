#!/usr/bin/env python3
"""Helpers for preparing and running SWE-bench harness evaluations."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List


def _slug(text: str) -> str:
    cleaned = []
    for ch in text:
        if ch.isalnum() or ch in ("-", "_"):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    return "".join(cleaned).strip("_")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
            count += 1
    return count


def create_predictions_from_mutated_instances(
    mutated_instances: List[dict],
    out_dir: str | Path = "data/mutated_patches",
) -> List[dict]:
    """Write grouped predictions JSONL files for SWE-bench harness runs.

    Expects each row to include:
    - instance_id
    - diff
    - mutation
    - hf_bm (dataset name, e.g. TuringEnterprises/SWE-Bench-plus-plus)
    """
    out_dir = Path(out_dir)

    grouped: Dict[tuple[str, str], List[dict]] = {}
    for row in mutated_instances:
        benchmark = row.get("hf_bm")
        mutation = row.get("mutation")
        instance_id = row.get("instance_id")
        diff = row.get("diff")

        if not benchmark or not mutation or not instance_id or diff is None:
            continue
        key = (str(benchmark), str(mutation))
        grouped.setdefault(key, []).append(row)

    jobs: List[dict] = []
    for (benchmark, mutation), rows in sorted(grouped.items()):
        dataset_slug = _slug(benchmark)
        mutation_slug = _slug(mutation)
        predictions_path = out_dir / f"{dataset_slug}_{mutation_slug}_mutated.jsonl"

        prediction_rows = [
            {
                "instance_id": row["instance_id"],
                "model_name_or_path": mutation,
                "model_patch": row["diff"],
            }
            for row in rows
        ]
        _write_jsonl(predictions_path, prediction_rows)

        jobs.append(
            {
                "benchmark": benchmark,
                "mutation": mutation,
                "predictions_path": str(predictions_path),
                "num_predictions": len(prediction_rows),
            }
        )

    return jobs


def _collect_reports_for_run(run_id: str, logs_root: Path = Path("logs/run_evaluation")) -> List[dict]:
    run_dir = logs_root / run_id
    if not run_dir.exists():
        return []

    rows: List[dict] = []
    for report_path in run_dir.rglob("report.json"):
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue

        instance_dir = report_path.parent
        model_dir = instance_dir.parent

        data["instance_id"] = instance_dir.name
        data["model_name_or_path"] = model_dir.name
        data["run_id"] = run_id
        rows.append(data)
    return rows


def evaluate_prediction_jobs(
    prediction_jobs: List[dict],
    *,
    max_workers: int = 4,
    output_dir: str | Path = "results/harness_eval",
    fail_fast: bool = False,
) -> List[dict]:
    """Run swebench harness for each prediction job and collect run metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: List[dict] = []
    for job in prediction_jobs:
        benchmark = job["benchmark"]
        mutation = job.get("mutation", "unknown")
        predictions_path = job["predictions_path"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{_slug(benchmark)}_{_slug(mutation)}_{timestamp}"

        cmd = [
            sys.executable,
            "-m",
            "swebench.harness.run_evaluation",
            "--predictions_path",
            predictions_path,
            "--dataset_name",
            benchmark,
            "--max_workers",
            str(max_workers),
            "--run_id",
            run_id,
        ]

        res = subprocess.run(cmd, capture_output=True, text=True)
        stdout_path = output_dir / f"{run_id}.stdout.txt"
        stderr_path = output_dir / f"{run_id}.stderr.txt"
        stdout_path.write_text(res.stdout or "", encoding="utf-8")
        stderr_path.write_text(res.stderr or "", encoding="utf-8")

        report_rows = _collect_reports_for_run(run_id)
        reports_path = output_dir / f"{run_id}.reports.jsonl"
        _write_jsonl(reports_path, report_rows)

        summary = {
            "benchmark": benchmark,
            "mutation": mutation,
            "predictions_path": predictions_path,
            "num_predictions": job.get("num_predictions", 0),
            "run_id": run_id,
            "returncode": res.returncode,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "reports_path": str(reports_path),
            "num_reports": len(report_rows),
        }
        summaries.append(summary)

        if fail_fast and res.returncode != 0:
            break

    summary_jsonl = output_dir / "evaluation_runs.jsonl"
    _write_jsonl(summary_jsonl, summaries)

    summary_csv = output_dir / "evaluation_runs.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "benchmark",
                "mutation",
                "predictions_path",
                "num_predictions",
                "run_id",
                "returncode",
                "stdout_path",
                "stderr_path",
                "reports_path",
                "num_reports",
            ],
        )
        writer.writeheader()
        writer.writerows(summaries)

    return summaries


def evaluate_predictions(
    predictions: Dict[str, str] | List[dict],
    *,
    max_workers: int = 4,
    output_dir: str | Path = "results/harness_eval",
    fail_fast: bool = False,
) -> List[dict]:
    """Backwards-compatible wrapper around `evaluate_prediction_jobs`."""
    if isinstance(predictions, dict):
        jobs = []
        for benchmark, predictions_path in predictions.items():
            jobs.append(
                {
                    "benchmark": benchmark,
                    "mutation": "unknown",
                    "predictions_path": predictions_path,
                    "num_predictions": 0,
                }
            )
    else:
        jobs = predictions

    return evaluate_prediction_jobs(
        jobs,
        max_workers=max_workers,
        output_dir=output_dir,
        fail_fast=fail_fast,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-jsonl", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--mutation", default="manual")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--output-dir", default="results/harness_eval", type=Path)
    args = parser.parse_args()

    jobs = [
        {
            "benchmark": args.dataset_name,
            "mutation": args.mutation,
            "predictions_path": str(args.predictions_jsonl),
            "num_predictions": sum(1 for _ in args.predictions_jsonl.open("r", encoding="utf-8")),
        }
    ]
    summaries = evaluate_prediction_jobs(
        jobs,
        max_workers=args.max_workers,
        output_dir=args.output_dir,
    )
    print(json.dumps(summaries, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
