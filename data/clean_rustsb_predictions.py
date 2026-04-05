#!/usr/bin/env python3
"""Generate clean SWE-bench predictions files from rustbench gs_mutated JSONL.

The rustbench gs_mutated JSONL contains extra metadata fields (base_commit, org,
repo, problem, etc.) beyond the standard 3-field SWE-bench predictions format.
The Rust-SWE-Bench fork (GhabiX/Rust-SWE-Bench) may not handle these extra
fields correctly, causing patches to apply but correctness tests to fail.

This script:
1. Strips extra fields, keeping only {instance_id, model_name_or_path, model_patch}
2. Optionally sets model_name_or_path to "gold" to match the gold pathway
3. Verifies patches against the parquet source
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


DEFAULT_INPUT = Path("data/2_frozen_mutated_patches/user2f86_rustbench_gs_mutated.jsonl")
DEFAULT_PARQUET = Path("data/0_benchmark-sets/202603016_rust-swe-bench.parquet")
DEFAULT_OUTPUT = Path("data/2_frozen_mutated_patches/user2f86_rustbench_gs_predictions.jsonl")


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=DEFAULT_INPUT,
        help="Path to the rustbench gs_mutated JSONL with extra fields.",
    )
    parser.add_argument(
        "--parquet", type=Path, default=DEFAULT_PARQUET,
        help="Path to the rustbench parquet for verification.",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Path to write the clean predictions JSONL.",
    )
    parser.add_argument(
        "--model-name", default="gold",
        help="Value for model_name_or_path (default: 'gold').",
    )
    parser.add_argument(
        "--skip-verify", action="store_true",
        help="Skip parquet verification.",
    )
    args = parser.parse_args()

    rows = _read_jsonl(args.input)
    if not rows:
        print(f"ERROR: No rows in {args.input}", file=sys.stderr)
        return 1

    # Check what fields are present
    extra_fields = set(rows[0].keys()) - {"instance_id", "model_name_or_path", "model_patch"}
    if extra_fields:
        print(f"Stripping extra fields from predictions: {sorted(extra_fields)}")

    # Build clean predictions
    clean = []
    for row in rows:
        instance_id = row.get("instance_id")
        model_patch = row.get("model_patch")
        if not instance_id or model_patch is None:
            print(f"WARNING: skipping row missing instance_id or model_patch: {row.get('instance_id')}")
            continue
        clean.append({
            "instance_id": instance_id,
            "model_name_or_path": args.model_name,
            "model_patch": model_patch,
        })

    # Verify against parquet
    if not args.skip_verify and args.parquet.exists():
        df = pd.read_parquet(args.parquet)
        parquet_patches = {
            str(r["instance_id"]): str(r["patch"])
            for _, r in df[["instance_id", "patch"]].iterrows()
        }

        mismatches = 0
        missing = 0
        for pred in clean:
            iid = pred["instance_id"]
            pp = parquet_patches.get(iid)
            if pp is None:
                print(f"  WARNING: {iid} not found in parquet")
                missing += 1
            elif pred["model_patch"] != pp:
                print(f"  MISMATCH: {iid}")
                mismatches += 1

        if mismatches or missing:
            print(f"Verification: {mismatches} mismatches, {missing} missing")
        else:
            print(f"Verification: all {len(clean)} patches match parquet")

    _write_jsonl(args.output, clean)
    print(f"Wrote {len(clean)} clean predictions to {args.output}")
    print(f"  model_name_or_path: {args.model_name!r}")
    print(f"  Fields per row: {sorted(clean[0].keys()) if clean else '(empty)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
