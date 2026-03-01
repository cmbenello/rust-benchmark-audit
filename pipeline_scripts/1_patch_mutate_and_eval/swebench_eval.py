#!/usr/bin/env python3
"""Helpers for preparing and running SWE-bench harness evaluations."""
from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

KNOWN_DATASET_INCOMPATIBLE = {
    "ByteDance-Seed/Multi-SWE-bench": (
        "Known dataset loader schema incompatibility with current "
        "swebench/datasets versions"
    )
}

LOCAL_NORMALIZE_BENCHMARKS = {"TuringEnterprises/SWE-Bench-plus-plus"}
RUNNER_SCRIPT = Path(__file__).resolve().parent / "run_swebench_eval.py"


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


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _resolve_docker_host(explicit_docker_host: str | None = None) -> str | None:
    if explicit_docker_host:
        return explicit_docker_host
    existing = os.environ.get("DOCKER_HOST")
    if existing:
        return existing

    desktop_sock = Path.home() / ".docker" / "run" / "docker.sock"
    if desktop_sock.exists():
        return f"unix://{desktop_sock}"

    default_sock = Path("/var/run/docker.sock")
    if default_sock.exists():
        return "unix:///var/run/docker.sock"

    return None


def _normalize_test_list(value):
    """Normalize FAIL_TO_PASS/PASS_TO_PASS values to list objects."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        # Preferred parser: strict JSON.
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Fallback for python-literal style strings (single quotes, etc.).
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                return []
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, tuple):
            return list(parsed)
        if isinstance(parsed, str):
            return [parsed]
        return []
    return []


def _infer_version_from_instance_id(instance_id: str | None) -> str | None:
    if not isinstance(instance_id, str):
        return None
    if "-" not in instance_id:
        return None
    return instance_id.rsplit("-", 1)[-1]


def _prepare_local_dataset_for_job(
    benchmark: str,
    predictions_path: str,
    output_dir: Path,
) -> tuple[str, bool, str, str | None]:
    """
    For benchmarks with known schema quirks, materialize a local JSONL dataset
    with normalized PASS_TO_PASS/FAIL_TO_PASS fields and return that path.
    """
    if benchmark not in LOCAL_NORMALIZE_BENCHMARKS:
        return benchmark, False, "", None

    try:
        from datasets import load_dataset
    except Exception:
        return benchmark, False, "", None

    try:
        pred_rows = _read_jsonl(Path(predictions_path))
        instance_ids = {r["instance_id"] for r in pred_rows if "instance_id" in r}
        if not instance_ids:
            return benchmark, False, "", None

        ds = load_dataset(benchmark, split="test")
        rows = []
        inferred_versions = 0
        for row in ds:
            iid = row.get("instance_id")
            if iid not in instance_ids:
                continue
            item = dict(row)
            # Some SWE-Bench++ rows have missing/None version fields.
            if not item.get("version"):
                inferred_version = _infer_version_from_instance_id(iid)
                if inferred_version:
                    item["version"] = inferred_version
                    inferred_versions += 1
            item["PASS_TO_PASS"] = _normalize_test_list(item.get("PASS_TO_PASS"))
            item["FAIL_TO_PASS"] = _normalize_test_list(item.get("FAIL_TO_PASS"))
            rows.append(item)

        if not rows:
            return benchmark, False, "", None

        dataset_path = output_dir / f"{_slug(benchmark)}_normalized_dataset.jsonl"
        _write_jsonl(dataset_path, rows)
        note = f"Normalized local dataset rows={len(rows)}, inferred_versions={inferred_versions}"
        return str(dataset_path), False, note, str(dataset_path)
    except Exception:
        # Fail open to the original benchmark so runs still proceed if normalization fails.
        return benchmark, False, "", None


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
    docker_host: str | None = None,
    skip_known_incompatible: bool = True,
) -> List[dict]:
    """Run swebench harness for each prediction job and collect run metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_docker_host = _resolve_docker_host(docker_host)

    summaries: List[dict] = []
    for job in prediction_jobs:
        benchmark = job["benchmark"]
        mutation = job.get("mutation", "unknown")
        predictions_path = job["predictions_path"]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{_slug(benchmark)}_{_slug(mutation)}_{timestamp}"

        if skip_known_incompatible and benchmark in KNOWN_DATASET_INCOMPATIBLE:
            summary = {
                "benchmark": benchmark,
                "mutation": mutation,
                "predictions_path": predictions_path,
                "num_predictions": job.get("num_predictions", 0),
                "run_id": run_id,
                "returncode": 0,
                "status": "skipped",
                "note": KNOWN_DATASET_INCOMPATIBLE[benchmark],
                "docker_host_used": resolved_docker_host or "",
                "dataset_name_used": benchmark,
                "stdout_path": "",
                "stderr_path": "",
                "reports_path": "",
                "num_reports": 0,
            }
            summaries.append(summary)
            continue

        dataset_arg, skip_for_specs, prep_note, dynamic_specs_dataset = _prepare_local_dataset_for_job(
            benchmark=benchmark,
            predictions_path=predictions_path,
            output_dir=output_dir,
        )
        if skip_for_specs:
            summary = {
                "benchmark": benchmark,
                "mutation": mutation,
                "predictions_path": predictions_path,
                "num_predictions": job.get("num_predictions", 0),
                "run_id": run_id,
                "returncode": 0,
                "status": "skipped",
                "note": prep_note,
                "docker_host_used": resolved_docker_host or "",
                "dataset_name_used": dataset_arg,
                "stdout_path": "",
                "stderr_path": "",
                "reports_path": "",
                "num_reports": 0,
            }
            summaries.append(summary)
            continue

        cmd = [
            sys.executable,
            str(RUNNER_SCRIPT),
            "--predictions_path",
            predictions_path,
            "--dataset_name",
            dataset_arg,
            "--max_workers",
            str(max_workers),
            "--run_id",
            run_id,
        ]
        if dynamic_specs_dataset:
            cmd.extend(["--dynamic_specs_dataset", dynamic_specs_dataset])

        env = os.environ.copy()
        if resolved_docker_host and not env.get("DOCKER_HOST"):
            env["DOCKER_HOST"] = resolved_docker_host

        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
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
            "status": "ok" if res.returncode == 0 else "failed",
            "note": prep_note,
            "docker_host_used": env.get("DOCKER_HOST", ""),
            "dataset_name_used": dataset_arg,
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
                "status",
                "note",
                "docker_host_used",
                "dataset_name_used",
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
    docker_host: str | None = None,
    skip_known_incompatible: bool = True,
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
        docker_host=docker_host,
        skip_known_incompatible=skip_known_incompatible,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-jsonl", required=True, type=Path)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--mutation", default="manual")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--output-dir", default="results/harness_eval", type=Path)
    parser.add_argument("--docker-host", default=None)
    parser.add_argument(
        "--include-known-incompatible",
        action="store_true",
        help="Include benchmarks known to error with current swebench/datasets versions.",
    )
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
        docker_host=args.docker_host,
        skip_known_incompatible=not args.include_known_incompatible,
    )
    print(json.dumps(summaries, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
