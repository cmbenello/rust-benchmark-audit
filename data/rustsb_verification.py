#!/usr/bin/env python3
"""Verify that rustbench GS patches are byte-equivalent across JSONL and parquet sources."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_JSONL = Path(
    "data/2_frozen_mutated_patches/user2f86_rustbench_gs_mutated.jsonl"
)
DEFAULT_PARQUET = Path("data/0_benchmark-sets/202603016_rust-swe-bench.parquet")


@dataclass(frozen=True)
class Mismatch:
    instance_id: str
    jsonl_patch: str | None
    parquet_patch: str | None


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise ValueError(f"{path}: line {line_number} is not a JSON object")
            rows.append(row)
    return rows


def _to_bytes(value: str | None) -> bytes | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    return value.encode("utf-8")


def _load_jsonl_patches(path: Path) -> dict[str, str | None]:
    rows = _read_jsonl(path)
    patches: dict[str, str | None] = {}
    for row in rows:
        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ValueError(f"{path}: missing or empty instance_id in row {row!r}")
        if instance_id in patches:
            raise ValueError(f"{path}: duplicate instance_id {instance_id!r}")
        patches[instance_id] = row.get("model_patch")
    return patches


def _load_parquet_patches(path: Path) -> dict[str, str | None]:
    df = pd.read_parquet(path)
    required = {"instance_id", "patch"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path}: missing required columns: {sorted(missing)}")

    patches: dict[str, str | None] = {}
    for row in df[["instance_id", "patch"]].itertuples(index=False):
        instance_id = str(row.instance_id)
        if instance_id in patches:
            raise ValueError(f"{path}: duplicate instance_id {instance_id!r}")
        patches[instance_id] = row.patch
    return patches


def _compare_patches(
    jsonl_patches: dict[str, str | None], parquet_patches: dict[str, str | None]
) -> tuple[list[str], list[str], list[str], list[Mismatch]]:
    jsonl_ids = set(jsonl_patches)
    parquet_ids = set(parquet_patches)

    missing_in_parquet = sorted(jsonl_ids - parquet_ids)
    extra_in_parquet = sorted(parquet_ids - jsonl_ids)
    common_ids = sorted(jsonl_ids & parquet_ids)

    exact_matches: list[str] = []
    mismatches: list[Mismatch] = []

    for instance_id in common_ids:
        jsonl_patch = jsonl_patches.get(instance_id)
        parquet_patch = parquet_patches.get(instance_id)
        if _to_bytes(jsonl_patch) == _to_bytes(parquet_patch):
            exact_matches.append(instance_id)
        else:
            mismatches.append(
                Mismatch(
                    instance_id=instance_id,
                    jsonl_patch=jsonl_patch,
                    parquet_patch=parquet_patch,
                )
            )

    return missing_in_parquet, extra_in_parquet, exact_matches, mismatches


def _print_sample_mismatches(mismatches: Iterable[Mismatch], limit: int) -> None:
    for mismatch in list(mismatches)[:limit]:
        jsonl_len = len(_to_bytes(mismatch.jsonl_patch) or b"")
        parquet_len = len(_to_bytes(mismatch.parquet_patch) or b"")
        print(f"instance_id: {mismatch.instance_id}")
        print(f"  jsonl_bytes: {jsonl_len}")
        print(f"  parquet_bytes: {parquet_len}")
        print(
            "  byte_equal: ",
            _to_bytes(mismatch.jsonl_patch) == _to_bytes(mismatch.parquet_patch),
            sep="",
        )
        print("  jsonl_patch_preview:")
        print((mismatch.jsonl_patch or "")[:400])
        print("  parquet_patch_preview:")
        print((mismatch.parquet_patch or "")[:400])
        print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=DEFAULT_JSONL,
        help="Path to the rustbench GS JSONL file.",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=DEFAULT_PARQUET,
        help="Path to the rustbench parquet benchmark file.",
    )
    parser.add_argument(
        "--show-mismatches",
        type=int,
        default=10,
        help="How many mismatches to print in detail.",
    )
    args = parser.parse_args()

    jsonl_patches = _load_jsonl_patches(args.jsonl)
    parquet_patches = _load_parquet_patches(args.parquet)

    missing_in_parquet, extra_in_parquet, exact_matches, mismatches = _compare_patches(
        jsonl_patches, parquet_patches
    )

    total_jsonl = len(jsonl_patches)
    total_parquet = len(parquet_patches)
    common = len(exact_matches) + len(mismatches)

    print(f"jsonl_rows={total_jsonl}")
    print(f"parquet_rows={total_parquet}")
    print(f"common_instance_ids={common}")
    print(f"exact_byte_matches={len(exact_matches)}")
    print(f"mismatches={len(mismatches)}")
    print(f"missing_in_parquet={len(missing_in_parquet)}")
    print(f"extra_in_parquet={len(extra_in_parquet)}")

    if missing_in_parquet:
        print("\nmissing_in_parquet_sample:")
        for instance_id in missing_in_parquet[: args.show_mismatches]:
            print(f"  - {instance_id}")

    if extra_in_parquet:
        print("\nextra_in_parquet_sample:")
        for instance_id in extra_in_parquet[: args.show_mismatches]:
            print(f"  - {instance_id}")

    if mismatches:
        print("\nmismatch_details:")
        _print_sample_mismatches(mismatches, args.show_mismatches)

    if missing_in_parquet or extra_in_parquet or mismatches:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
