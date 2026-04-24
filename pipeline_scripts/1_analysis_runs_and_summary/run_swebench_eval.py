#!/usr/bin/env python3
"""Wrapper around swebench.harness.run_evaluation with optional dynamic specs patching."""
from __future__ import annotations

import argparse
import ast
import json
import os
import platform
from pathlib import Path


LANG_TO_EXT = {
    "python": "py",
    "py": "py",
    "javascript": "js",
    "js": "js",
    "typescript": "js",
    "ts": "js",
    "rust": "rs",
    "rs": "rs",
    "java": "java",
    "go": "go",
    "php": "php",
    "ruby": "rb",
    "rb": "rb",
    "c": "c",
    "cpp": "cpp",
    "c++": "cpp",
}


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _infer_version_from_instance_id(instance_id: str | None) -> str | None:
    if not isinstance(instance_id, str):
        return None
    if "-" not in instance_id:
        return None
    return instance_id.rsplit("-", 1)[-1]


def _parse_environment_config(raw):
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                return {}
    return {}


def _default_parser_for_extension(ext: str):
    if ext == "rs":
        from swebench.harness.log_parsers.rust import parse_log_cargo

        return parse_log_cargo
    return None


def _normalize_test_cmd(value):
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else None
    return None


def _infer_repo_from_instance_id(instance_id: str | None) -> str | None:
    if not isinstance(instance_id, str):
        return None
    head = instance_id.rsplit("-", 1)[0]
    if "__" not in head:
        return None
    owner, repo = head.split("__", 1)
    if not owner or not repo:
        return None
    return f"{owner}/{repo}"


def _prediction_instance_ids(predictions_path: Path) -> list[str]:
    rows = _read_jsonl(predictions_path)
    ids: list[str] = []
    for idx, row in enumerate(rows, start=1):
        iid = row.get("instance_id")
        if not isinstance(iid, str) or not iid.strip():
            raise ValueError(
                f"predictions row {idx} in {predictions_path} is missing a non-empty instance_id"
            )
        ids.append(iid)
    return ids


def _dataset_instance_ids(dataset_name: str, split: str) -> tuple[set[str], str]:
    dataset_path = Path(dataset_name)
    if dataset_path.exists():
        ids = {
            row["instance_id"]
            for row in _read_jsonl(dataset_path)
            if isinstance(row.get("instance_id"), str) and row.get("instance_id")
        }
        return ids, f"local:{dataset_path}"

    from datasets import load_dataset

    try:
        ds = load_dataset(dataset_name, split=split)
        used_split = split
    except ValueError as exc:
        # Community datasets can publish only train; support the common test->train fallback.
        if split != "test" or 'Unknown split "test"' not in str(exc):
            raise
        ds = load_dataset(dataset_name, split="train")
        used_split = "train"

    ids = {
        row["instance_id"]
        for row in ds
        if isinstance(row.get("instance_id"), str) and row.get("instance_id")
    }
    return ids, f"hf:{dataset_name}:{used_split}"


def _preflight_validate_prediction_membership(
    predictions_path: Path,
    dataset_name: str,
    split: str,
) -> tuple[int, int, str]:
    pred_ids = _prediction_instance_ids(predictions_path)
    pred_set = set(pred_ids)
    if len(pred_set) != len(pred_ids):
        raise ValueError(
            f"predictions file has duplicate instance_id values: {len(pred_ids) - len(pred_set)} duplicates"
        )

    dataset_ids, dataset_source = _dataset_instance_ids(dataset_name, split)
    missing = [iid for iid in pred_ids if iid not in dataset_ids]
    if missing:
        sample = ", ".join(missing[:10])
        raise ValueError(
            "preflight failed: some prediction instance_id values are not present in dataset "
            f"({dataset_source}). missing={len(missing)} sample=[{sample}]"
        )
    return len(pred_ids), len(dataset_ids), dataset_source


def _patch_swebench_specs_from_dataset(
    dataset_jsonl: Path,
    benchmark_hint: str | None = None,
    rust_version_override: str | None = None,
) -> tuple[int, int, int, int]:
    """
    Dynamically patch swebench constants so unsupported repos/versions in a local
    dataset can still run via common script generation.

    When ``rust_version_override`` is set and the row is a rust/rustbench row,
    we ensure ``docker_specs.rust_version`` is at least that value. This is
    needed because upstream swebench pins rust_version=1.81 for its 7 known
    rust repos, but many popular crates on crates.io now require Cargo 1.85+
    for ``edition2024``.
    """
    from swebench.harness.constants import MAP_REPO_TO_EXT, MAP_REPO_VERSION_TO_SPECS
    from swebench.harness.log_parsers import MAP_REPO_TO_PARSER

    hint_text = (benchmark_hint or "").lower()
    dataset_text = str(dataset_jsonl).lower()
    is_rustbench = "rustbench" in hint_text or "rustbench" in dataset_text

    def _bump_rust_version(existing_specs: dict) -> bool:
        """Force `docker_specs.rust_version` to the override when set.

        Returns True if the spec was modified.
        """
        if not rust_version_override:
            return False
        ds = existing_specs.get("docker_specs")
        if not isinstance(ds, dict):
            ds = {}
            existing_specs["docker_specs"] = ds
        if ds.get("rust_version") != rust_version_override:
            ds["rust_version"] = rust_version_override
            return True
        return False

    patched = 0
    skipped = 0
    parser_patched = 0
    rust_version_bumped = 0
    for row in _read_jsonl(dataset_jsonl):
        repo = row.get("repo") or _infer_repo_from_instance_id(row.get("instance_id"))
        language = str(row.get("language") or "").lower()
        ext = LANG_TO_EXT.get(language)
        if not ext and repo:
            ext = MAP_REPO_TO_EXT.get(repo)
        if not ext and is_rustbench:
            ext = "rs"
        version = row.get("version")
        if not version:
            version = _infer_version_from_instance_id(row.get("instance_id"))
        if version is None:
            skipped += 1
            continue
        version = str(version)

        if not repo or not ext:
            skipped += 1
            continue

        env_cfg = _parse_environment_config(row.get("environment_config"))
        repo_specs = MAP_REPO_VERSION_TO_SPECS.get(repo, {})
        has_existing_repo_version_spec = isinstance(repo_specs, dict) and version in repo_specs

        test_cmd = _normalize_test_cmd(env_cfg.get("test_cmd"))
        if not test_cmd:
            row_test_cmd = row.get("test_cmd")
            if row_test_cmd:
                test_cmd = _normalize_test_cmd(row_test_cmd)
        # Rustbench rows often omit per-instance test metadata. In that case,
        # keep upstream swebench repo/version specs when they already exist,
        # but optionally bump rust_version so transitive deps that require a
        # newer edition don't blow up the install step.
        if not test_cmd and has_existing_repo_version_spec:
            MAP_REPO_TO_EXT.setdefault(repo, ext)
            if ext == "rs" and is_rustbench:
                existing_specs = repo_specs[version]
                if isinstance(existing_specs, dict) and _bump_rust_version(existing_specs):
                    rust_version_bumped += 1
            if repo not in MAP_REPO_TO_PARSER:
                parser_fn = _default_parser_for_extension(ext)
                if parser_fn is not None:
                    MAP_REPO_TO_PARSER[repo] = parser_fn
                    parser_patched += 1
            continue
        if not test_cmd and ext == "rs" and is_rustbench:
            # If the repo/version is not known to swebench, we still need a
            # synthetic spec. Avoid --locked because rustbench instances often
            # have Cargo.lock files that pre-date the current rust toolchain
            # and fail `cargo test --locked` with "lockfile needs update".
            # --offline is ignored since the env image already has deps
            # vendored, so let cargo refresh the lock if necessary.
            test_cmd = ["RUSTFLAGS=-Awarnings cargo test --workspace --all-targets"]
        if not test_cmd:
            skipped += 1
            continue

        docker_specs = env_cfg.get("docker_specs", {})
        if not isinstance(docker_specs, dict):
            docker_specs = {}
        if ext == "rs" and is_rustbench:
            default_rust = rust_version_override or str(row.get("rust_version") or "1.81")
            docker_specs.setdefault("rust_version", default_rust)
            if rust_version_override and docker_specs.get("rust_version") != rust_version_override:
                docker_specs["rust_version"] = rust_version_override
                rust_version_bumped += 1

        specs = {
            "test_cmd": test_cmd,
            "pre_install": env_cfg.get("pre_install", []),
            "install": env_cfg.get("install", []),
            "build": env_cfg.get("build", []),
            "docker_specs": docker_specs,
        }
        # Keep keys compact: remove empty lists/dicts that are not needed.
        specs = {k: v for k, v in specs.items() if v not in (None, [], {}, "")}

        MAP_REPO_TO_EXT.setdefault(repo, ext)
        MAP_REPO_VERSION_TO_SPECS.setdefault(repo, {})
        MAP_REPO_VERSION_TO_SPECS[repo][version] = specs
        if repo not in MAP_REPO_TO_PARSER:
            parser_fn = _default_parser_for_extension(ext)
            if parser_fn is not None:
                MAP_REPO_TO_PARSER[repo] = parser_fn
                parser_patched += 1
        patched += 1

    return patched, skipped, parser_patched, rust_version_bumped


def _resolve_arch(requested_arch: str) -> str:
    if requested_arch != "auto":
        return requested_arch

    env_arch = os.environ.get("SWEBENCH_ARCH", "").strip().lower()
    if env_arch in {"x86_64", "arm64"}:
        return env_arch

    machine = platform.machine().lower()
    if machine in {"arm64", "aarch64"}:
        return "arm64"
    return "x86_64"


def _patch_default_test_spec_arch(selected_arch: str) -> None:
    """
    Force swebench TestSpec creation to use a host-compatible default arch.
    This avoids local Docker build failures on arm64 hosts when swebench defaults
    to x86_64.
    """
    from swebench.harness.test_spec import test_spec as ts

    original_make_test_spec = ts.make_test_spec

    def make_test_spec_with_arch(
        instance,
        namespace=None,
        base_image_tag="latest",
        env_image_tag="latest",
        instance_image_tag="latest",
        arch="x86_64",
    ):
        effective_arch = selected_arch if arch in (None, "x86_64") else arch
        return original_make_test_spec(
            instance,
            namespace=namespace,
            base_image_tag=base_image_tag,
            env_image_tag=env_image_tag,
            instance_image_tag=instance_image_tag,
            arch=effective_arch,
        )

    ts.make_test_spec = make_test_spec_with_arch


def _optional_namespace(namespace: str | None) -> str | None:
    if namespace is None:
        return None
    lowered = namespace.strip().lower()
    if lowered in {"", "none", "null"}:
        return None
    return namespace


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions_path", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--open_file_limit", type=int, default=4096)
    parser.add_argument("--cache_level", default="env")
    parser.add_argument(
        "--namespace",
        default="none",
        help='Use "none" for local-only images; set a namespace to pull/push remote images.',
    )
    parser.add_argument("--instance_image_tag", default="latest")
    parser.add_argument("--env_image_tag", default="latest")
    parser.add_argument("--report_dir", default=".")
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--rewrite_reports", action="store_true")
    parser.add_argument("--modal", action="store_true")
    parser.add_argument("--dynamic_specs_dataset", default=None, type=Path)
    parser.add_argument(
        "--arch",
        choices=["auto", "x86_64", "arm64"],
        default="auto",
        help=(
            "Architecture for local swebench Docker images. "
            "Use auto to infer from host (or SWEBENCH_ARCH env)."
        ),
    )
    parser.add_argument("--instance_ids", nargs="*", default=[])
    parser.add_argument(
        "--benchmark_hint",
        default=None,
        help="Original benchmark identifier (for dataset-specific fallback logic).",
    )
    parser.add_argument(
        "--preflight_only",
        action="store_true",
        help="Run prediction-vs-dataset instance_id validation and exit.",
    )
    parser.add_argument(
        "--rust_version_override",
        default=None,
        help=(
            "Force docker_specs.rust_version to this value for rust/rustbench rows. "
            "Useful because upstream swebench pins rust_version=1.81 on its 7 known "
            "rust repos, while many current crates.io deps require Cargo 1.85+ "
            "(edition2024). Recommended: 1.86 or newer."
        ),
    )
    args = parser.parse_args()

    pred_count, dataset_count, dataset_source = _preflight_validate_prediction_membership(
        predictions_path=Path(args.predictions_path),
        dataset_name=args.dataset_name,
        split=args.split,
    )
    print(
        "Preflight membership check passed: "
        f"predictions={pred_count}, dataset_instance_ids={dataset_count}, source={dataset_source}"
    )
    if args.preflight_only:
        return 0

    if args.dynamic_specs_dataset:
        patch_hint = args.benchmark_hint or args.dataset_name
        patched, skipped, parser_patched, rust_bumped = _patch_swebench_specs_from_dataset(
            args.dynamic_specs_dataset,
            benchmark_hint=patch_hint,
            rust_version_override=args.rust_version_override,
        )
        print(
            f"Dynamic swebench specs patch: patched={patched}, skipped_rows={skipped}, "
            f"parser_patched={parser_patched}, rust_version_bumped={rust_bumped}, "
            f"dataset={args.dynamic_specs_dataset}, benchmark_hint={patch_hint}"
        )

    selected_arch = _resolve_arch(args.arch)
    _patch_default_test_spec_arch(selected_arch)
    print(f"Using swebench local arch: {selected_arch}")

    from swebench.harness import run_evaluation
    from swebench.harness.test_spec import test_spec as ts

    # run_evaluation imports make_test_spec by value, so keep it aligned with the
    # patched default arch used by test_spec.test_spec.
    run_evaluation.make_test_spec = ts.make_test_spec

    run_evaluation.main(
        dataset_name=args.dataset_name,
        split=args.split,
        instance_ids=args.instance_ids or None,
        predictions_path=args.predictions_path,
        max_workers=args.max_workers,
        force_rebuild=args.force_rebuild,
        cache_level=args.cache_level,
        clean=args.clean,
        open_file_limit=args.open_file_limit,
        run_id=args.run_id,
        timeout=args.timeout,
        namespace=_optional_namespace(args.namespace),
        rewrite_reports=args.rewrite_reports,
        modal=args.modal,
        instance_image_tag=args.instance_image_tag,
        env_image_tag=args.env_image_tag,
        report_dir=args.report_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
