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


def _patch_swebench_specs_from_dataset(dataset_jsonl: Path) -> tuple[int, int, int]:
    """
    Dynamically patch swebench constants so unsupported repos/versions in a local
    dataset can still run via common script generation.
    """
    from swebench.harness.constants import MAP_REPO_TO_EXT, MAP_REPO_VERSION_TO_SPECS
    from swebench.harness.log_parsers import MAP_REPO_TO_PARSER

    patched = 0
    skipped = 0
    parser_patched = 0
    for row in _read_jsonl(dataset_jsonl):
        repo = row.get("repo")
        language = str(row.get("language") or "").lower()
        ext = LANG_TO_EXT.get(language)
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
        test_cmd = env_cfg.get("test_cmd")
        if not test_cmd:
            skipped += 1
            continue

        specs = {
            "test_cmd": test_cmd,
            "pre_install": env_cfg.get("pre_install", []),
            "install": env_cfg.get("install", []),
            "build": env_cfg.get("build", []),
            "docker_specs": env_cfg.get("docker_specs", {}),
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

    return patched, skipped, parser_patched


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
    args = parser.parse_args()

    if args.dynamic_specs_dataset:
        patched, skipped, parser_patched = _patch_swebench_specs_from_dataset(
            args.dynamic_specs_dataset
        )
        print(
            f"Dynamic swebench specs patch: patched={patched}, skipped_rows={skipped}, "
            f"parser_patched={parser_patched}, dataset={args.dynamic_specs_dataset}"
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
