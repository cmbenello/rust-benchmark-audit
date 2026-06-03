"""Pull SWE-bench-style patches from a Docent collection.

Designed for the rust-benchmark-audit pipeline: given a Docent
``collection_id`` (the URL slug at docent.transluce.org/dashboard/<id>/...)
and a list of swe-bench instance_ids, this script finds the matching
agent run per instance, extracts the final ``patch.txt`` the agent
produced, and writes a predictions JSONL the harness can consume:

    {"instance_id": ..., "model_name_or_path": ..., "model_patch": ...}

usage:

    export DOCENT_API_KEY=...  # get one at docent.transluce.org
    python3 fetch_docent_patches.py \\
        --collection-id 8dc7c4e5-2423-470c-88d8-12dafb1870ec \\
        --instance-ids-file data/2_frozen_mutated_patches/adversarial_SWE-bench_SWE-bench_Multilingual_unsafe.jsonl \\
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


# Try several DQL where-clause shapes; Docent runs may store the instance_id
# as a top-level metadata key or nested under "meta". We probe in priority
# order and pick the first that returns at least one match for the FIRST
# instance_id; that shape is then reused for all subsequent queries.
_DQL_TEMPLATES = [
    "metadata_json->>'instance_id' = '{iid}'",
    "metadata_json->'meta'->>'instance_id' = '{iid}'",
    "metadata_json->>'meta.instance_id' = '{iid}'",
    "name = '{iid}'",
]


# Regex grabs a unified-diff blob, anchored on `diff --git` and ending at the
# first non-diff line (or end of string). This is intentionally permissive —
# patches may have multiple files, trailing whitespace, etc.
_DIFF_BLOCK_RE = re.compile(
    r"(diff --git a/[^\n]+\n(?:(?!\n(?:[^\-\+@diff\s\\ ]|diff --git ))[\s\S])+)",
    re.MULTILINE,
)


def _load_instance_ids(path: str) -> list[str]:
    """Read instance_ids from a JSONL (one row per line, ``instance_id`` key)."""
    out: list[str] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("bad jsonl line in %s: %s", path, e)
                continue
            iid = d.get("instance_id")
            if iid:
                out.append(iid)
    # Dedupe but preserve order
    seen: set[str] = set()
    deduped = []
    for iid in out:
        if iid not in seen:
            seen.add(iid)
            deduped.append(iid)
    return deduped


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
                m = _DIFF_BLOCK_RE.search(cand)
                if m:
                    return m.group(1)
                # tool result wasn't a diff; keep falling back to the
                # any-diff scan below.

    # Pass 2: scan any-message contents from the end for a diff block.
    for i in range(len(msgs) - 1, -1, -1):
        text = _msg_text(msgs[i])
        if not text:
            continue
        m = _DIFF_BLOCK_RE.search(text)
        if m:
            return m.group(1)

    return None


def _probe_where_clause(client, collection_id: str, instance_id: str) -> str | None:
    """Try the candidate DQL templates against one known instance_id and
    return the first that matches at least one run. Returns None if none
    of the templates match anything (likely metadata key path is different)."""
    for tmpl in _DQL_TEMPLATES:
        clause = tmpl.format(iid=instance_id)
        try:
            ids = client.select_agent_run_ids(collection_id, where_clause=clause, limit=1)
        except Exception as e:  # noqa: BLE001
            logger.warning("DQL probe failed for %r: %s", clause, e)
            continue
        if ids:
            logger.info("metadata field path confirmed via DQL: %s", tmpl)
            return tmpl
    return None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--collection-id", required=True,
                   help="Docent collection id (the URL slug at docent.transluce.org/dashboard/<id>/...)")
    p.add_argument("--instance-ids-file", required=True,
                   help="Path to a JSONL containing rows with `instance_id`. We extract the IDs.")
    p.add_argument("--model-name", required=True,
                   help="model_name_or_path value to embed in each prediction row.")
    p.add_argument("--output", required=True,
                   help="Output predictions .jsonl path.")
    p.add_argument("--api-key", default=None,
                   help="Docent API key. Defaults to $DOCENT_API_KEY.")
    p.add_argument("--limit-per-iid", type=int, default=3,
                   help="Max candidate runs to inspect per instance_id (default 3).")
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
    instance_ids = _load_instance_ids(args.instance_ids_file)
    if not instance_ids:
        print(f"ERROR: no instance_ids found in {args.instance_ids_file}", file=sys.stderr)
        return 2
    logger.info("loaded %d instance_id(s) from %s", len(instance_ids), args.instance_ids_file)

    # Discover which metadata field path holds the instance_id by probing
    # against the first ID. If none match, surface a helpful error rather
    # than silently writing an empty predictions file.
    template = _probe_where_clause(client, args.collection_id, instance_ids[0])
    if template is None:
        print(
            f"ERROR: none of the candidate DQL clauses matched any run for "
            f"instance_id={instance_ids[0]!r} in collection={args.collection_id}. "
            f"Tried: {', '.join(_DQL_TEMPLATES)}. "
            f"Open one run in the dashboard, inspect its metadata, and add "
            f"the right path to _DQL_TEMPLATES.",
            file=sys.stderr,
        )
        return 1

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    n_no_runs = 0
    n_no_patch = 0
    with open(args.output, "w") as out_fp:
        for iid in instance_ids:
            clause = template.format(iid=iid)
            try:
                ids = client.select_agent_run_ids(
                    args.collection_id, where_clause=clause, limit=args.limit_per_iid,
                )
            except Exception as e:  # noqa: BLE001
                logger.error("DQL failed for %s: %s", iid, e)
                continue
            if not ids:
                logger.warning("[no runs] %s — DQL: %s", iid, clause)
                n_no_runs += 1
                continue
            # Inspect candidate runs in order, take the first that has a patch.
            patch = None
            resolved = None
            for run_id in ids:
                try:
                    run = client.get_agent_run(args.collection_id, run_id)
                except Exception as e:  # noqa: BLE001
                    logger.error("get_agent_run failed (%s, %s): %s", iid, run_id, e)
                    continue
                if run is None:
                    continue
                            # Extract metadata.scores.resolved if present.
                metadata = getattr(run, "metadata", None)
                if metadata:
                    if isinstance(metadata, dict):
                        resolved = (
                            metadata.get("scores", {})
                            .get("resolved")
                        )
                    else:
                        scores = getattr(metadata, "scores", None)
                        if isinstance(scores, dict):
                            resolved = scores.get("resolved")
                        else:
                            resolved = getattr(scores, "resolved", None)
                patch = _extract_patch_from_transcripts(run.transcripts, iid)
                if patch:
                    logger.info("[ok] %s — found patch in run %s (%d chars)", iid, run_id, len(patch))
                    break
                logger.debug("[no patch in run] %s/%s", iid, run_id)
            if not patch:
                logger.warning("[no patch] %s — scanned %d run(s), no diff found", iid, len(ids))
                n_no_patch += 1
                continue
            out_fp.write(json.dumps({
                "instance_id": iid,
                "model_name_or_path": args.model_name,
                "model_patch": patch,
                "resolved": resolved
            }) + "\n")
            n_written += 1

    logger.info(
        "done: wrote=%d no_runs=%d no_patch=%d total=%d -> %s",
        n_written, n_no_runs, n_no_patch, len(instance_ids), args.output,
    )
    return 0 if n_written > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
