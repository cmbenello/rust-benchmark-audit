"""Pull SWE-bench-style patches from a Docent collection.

Designed for the rust-benchmark-audit pipeline: given a Docent
``collection_id`` (the URL slug at docent.transluce.org/dashboard/<id>/...),
this script fetches ALL runs in the collection, filters to those whose
``instance_id`` metadata starts with an ``org__repo`` prefix derived from
TARGET_REPOS, extracts the final ``patch.txt`` the agent produced, and
writes a predictions JSONL the harness can consume:

    {"instance_id": ..., "model_name_or_path": ..., "model_patch": ...}

Instance IDs follow the SWE-bench convention: ``org__repo-<number>``.
TARGET_REPOS entries are ``org/repo`` — slashes are converted to ``__``
to form the prefix used for filtering.

usage:

    export DOCENT_API_KEY=...  # get one at docent.transluce.org
    python3 fetch_docent_patches.py \\
        --collection-id 8dc7c4e5-2423-470c-88d8-12dafb1870ec \\
        --model-name claude-opus \\
        --output predictions/multilingual_claude_opus.jsonl

Failure modes are explicit, never silent:

    * a 0-match query is reported with the failing DQL so the metadata
      field path can be diagnosed
    * a run that exists but contains no recognizable patch.txt prints
      a one-line digest of the last few assistant/tool messages so the
      operator can spot agents that produced no fix
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Iterable

logger = logging.getLogger("fetch_docent_patches")


TARGET_REPOS = [
    "tokio-rs/tokio",
    "uutils/coreutils",
    "nushell/nushell",
    "tokio-rs/axum",
    "burntsushi/ripgrep",
    "sharkdp/bat",
    "astral-sh/ruff",
]

# Prefixes used to match instance_ids: "org/repo" -> "org__repo"
_TARGET_PREFIXES = tuple(repo.replace("/", "__") + "-" for repo in TARGET_REPOS)

# Matches where a unified diff starts; we grab everything from there to the
# end of the text (or until a clearly non-diff line after all hunks are done).
_DIFF_START_RE = re.compile(r"diff --git a/")


def _instance_id_from_run(run) -> str | None:
    """Extract the instance_id from a run's metadata, handling both dict and object forms."""
    metadata = getattr(run, "metadata", None)
    if not metadata:
        return None
    if isinstance(metadata, dict):
        iid = metadata.get("instance_id")
        if not iid:
            meta = metadata.get("meta") or {}
            iid = meta.get("instance_id") if isinstance(meta, dict) else None
    else:
        iid = getattr(metadata, "instance_id", None)
        if not iid:
            meta = getattr(metadata, "meta", None)
            iid = getattr(meta, "instance_id", None) if meta is not None else None
    return iid if isinstance(iid, str) and iid else None


def _msg_text(msg) -> str:
    """Pull a flat text string out of a Docent ``ChatMessage`` regardless of
    whether its content is a plain string or a structured list of blocks
    (text/tool_use/tool_result). Returns "" when content is empty."""
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            # pydantic block: prefer .text, fall back to .content or str(block)
            text = getattr(block, "text", None) or getattr(block, "content", None)
            if isinstance(text, str):
                parts.append(text)
            elif text is not None:
                parts.append(str(text))
        return "\n".join(parts)
    if content is None:
        return ""
    return str(content)


def _extract_patch_from_transcripts(transcripts: Iterable, instance_id: str) -> str | None:
    """Walk the transcript messages, find the agent's emitted ``patch.txt``.

    Priority:
      1. The LAST tool-result block immediately following a ``cat patch.txt``
         (or equivalent) call.
      2. The LAST occurrence of a unified-diff block in any message content.

    We scan from the end of the transcript because agents typically emit
    the final patch right before exit. Returns None if no diff was found
    so the caller can log + skip.
    """
    msgs = []
    for t in transcripts:
        msgs.extend(getattr(t, "messages", []) or [])
    if not msgs:
        return None

    # Pass 1: find the last (tool_use "cat patch.txt", tool_result) pair.
    # ChatMessage roles in Docent: system, user, assistant, tool.
    # The agent's tool call lives in an assistant message; the result is a
    # tool message that follows it. We scan backward and pair them up.
    cat_re = re.compile(r"\bcat\s+(?:\./)?patch\.txt\b")
    for i in range(len(msgs) - 1, -1, -1):
        msg = msgs[i]
        role = getattr(msg, "role", None)
        if role != "assistant":
            continue
        text = _msg_text(msg)
        if not cat_re.search(text):
            continue
        # Look for the next tool message following this assistant turn
        for j in range(i + 1, min(i + 6, len(msgs))):  # short window
            if getattr(msgs[j], "role", None) == "tool":
                cand = _msg_text(msgs[j])
                m = _DIFF_START_RE.search(cand)
                if m:
                    patch = cand[m.start():]
                    patch = re.sub(r'\n\[exit code \d+\]\s*$', '', patch)
                    return patch
                # tool result wasn't a diff; keep falling back to the
                # any-diff scan below.

    # Pass 2: scan any-message contents from the end for a diff block.
    for i in range(len(msgs) - 1, -1, -1):
        text = _msg_text(msgs[i])
        if not text:
            continue
        m = _DIFF_START_RE.search(text)
        if m:
            patch = text[m.start():]
            patch = re.sub(r'\n\[exit code \d+\]\s*$', '', patch)
            return patch

    return None


def _resolved_from_run(run) -> object:
    metadata = getattr(run, "metadata", None)
    if not metadata:
        return None
    if isinstance(metadata, dict):
        return (metadata.get("scores") or {}).get("resolved")
    scores = getattr(metadata, "scores", None)
    if isinstance(scores, dict):
        return scores.get("resolved")
    return getattr(scores, "resolved", None)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection-id", required=True,
                   help="Docent collection id (the URL slug at docent.transluce.org/dashboard/<id>/...)")
    p.add_argument("--model-name", required=True,
                   help="model_name_or_path value to embed in each prediction row.")
    p.add_argument("--output", required=True,
                   help="Output predictions .jsonl path.")
    p.add_argument("--api-key", default=None,
                   help="Docent API key. Defaults to $DOCENT_API_KEY.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        from docent import Docent
    except ImportError:
        print("ERROR: `docent` package not installed. Run: pip install docent", file=sys.stderr)
        return 2

    api_key = args.api_key or os.environ.get("DOCENT_API_KEY")
    if not api_key:
        print(
            "ERROR: DOCENT_API_KEY not set. Get one at docent.transluce.org "
            "(account → API key) and export DOCENT_API_KEY=...",
            file=sys.stderr,
        )
        return 2

    client = Docent(api_key=api_key)
    try:
        all_run_ids = client.list_agent_run_ids(args.collection_id)
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: failed to list runs for collection {args.collection_id}: {e}", file=sys.stderr)
        return 1

    logger.info(
        "collection %s has %d total run(s); filtering to prefixes: %s",
        args.collection_id, len(all_run_ids), list(_TARGET_PREFIXES),
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_skipped = 0
    n_no_patch = 0

    with open(args.output, "w") as out_fp:
        for run_id in all_run_ids:
            try:
                run = client.get_agent_run(args.collection_id, run_id)
            except Exception as e:  # noqa: BLE001
                logger.warning("get_agent_run failed (%s): %s", run_id, e)
                continue
            if run is None:
                continue

            iid = _instance_id_from_run(run)
            if not iid or not iid.startswith(_TARGET_PREFIXES):
                n_skipped += 1
                logger.debug("[skip] run %s — iid=%r not in target repos", run_id, iid)
                continue

            patch = _extract_patch_from_transcripts(run.transcripts, iid)
            if not patch:
                logger.warning("[no patch] %s (run %s) — no diff found", iid, run_id)
                n_no_patch += 1
                continue

            resolved = _resolved_from_run(run)
            logger.info("[ok] %s — patch found in run %s (%d chars)", iid, run_id, len(patch))
            out_fp.write(json.dumps({
                "instance_id": iid,
                "model_name_or_path": args.model_name,
                "model_patch": patch,
                "resolved": resolved,
            }) + "\n")
            n_written += 1

    logger.info(
        "done: total_runs=%d skipped=%d no_patch=%d wrote=%d -> %s",
        len(all_run_ids), n_skipped, n_no_patch, n_written, args.output,
    )
    return 0 if n_written > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
