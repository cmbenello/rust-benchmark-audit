from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_PARQUET_PATH = Path(__file__).resolve().parent / "202603016_rust-swe-bench.parquet"
DEFAULT_REPOS = [
    "async-graphql/async-graphql",
    "tokio-rs/axum",
    "nushell/nushell",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter rows from the Rust SWE-bench parquet where repo matches a target set, "
            "and optionally sample from the filtered rows."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_PARQUET_PATH,
        help=f"Path to the parquet file. Default: {DEFAULT_PARQUET_PATH}",
    )
    parser.add_argument(
        "--repo",
        dest="repos",
        action="append",
        default=None,
        help=(
            "Repo to include. Repeat this flag to override the defaults with a custom repo list."
        ),
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Number of rows to sample from the filtered result. Defaults to all matching rows.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed used when --sample-size is set. Default: 0",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path. Supports .jsonl and .csv. If omitted, rows are printed to stdout.",
    )
    return parser.parse_args()


def write_output(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()

    if suffix == ".jsonl":
        with output_path.open("w", encoding="utf-8") as handle:
            for record in df.to_dict(orient="records"):
                handle.write(json.dumps(record) + "\n")
        return

    if suffix == ".csv":
        df.to_csv(output_path, index=False)
        return

    raise ValueError("Unsupported output format. Use a .jsonl or .csv output path.")


def default_output_path(input_path: Path, sample_size: int | None) -> Path:
    suffix = "repo_sample" if sample_size is not None else "repo_matches"
    return input_path.with_name(f"{input_path.stem}_{suffix}.csv")


def main() -> None:
    args = parse_args()
    repos = args.repos if args.repos is not None else DEFAULT_REPOS

    if args.sample_size is not None and args.sample_size <= 0:
        raise ValueError("--sample-size must be a positive integer.")

    df = pd.read_parquet(args.input)

    if "repo" not in df.columns:
        raise KeyError("The parquet file does not contain a 'repo' column.")

    filtered_df = df[df["repo"].isin(repos)].copy()

    if args.sample_size is not None and not filtered_df.empty:
        sample_size = min(args.sample_size, len(filtered_df))
        filtered_df = filtered_df.sample(n=sample_size, random_state=args.random_state)

    print(f"Loaded {len(df)} rows from {args.input}")
    print(f"Matched {len(filtered_df)} rows across {len(repos)} repos")
    print(f"Repos: {repos}")

    output_path = args.output or default_output_path(args.input, args.sample_size)
    write_output(filtered_df, output_path)
    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()