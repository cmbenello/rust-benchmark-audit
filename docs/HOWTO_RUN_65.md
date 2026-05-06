# Running the 65-instance evaluation

This is the runbook for evaluating the 65 manually-sampled gold patches
against the new `mjgaughan/rscore-swe-bench` harness. After dedup the
harness submits 59 unique instances. As of the latest run on the
`test-parser-updates` branch we get **32 resolved / 17 unresolved /
10 errored**.

The same machinery runs mutated (policy-violating) patch sets — just
swap the predictions file. See [Mutation runs](#mutation-runs) at the
bottom.

---

## TL;DR

```bash
# on a linux x86_64 box with docker + sudo or docker-group access
ssh cascade

# one-time setup
cd ~/work
git clone -b test-parser-updates git@github.com:mjgaughan/rscore-swe-bench.git
cd rscore-swe-bench
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .

# build the 65-row dataset + gold predictions
python3 ~/work/benchmark-policy-gap/pipeline_scripts/0_data_construction/build_instances_from_csv.py \
    --rust-sb       ~/work/benchmark-policy-gap/data/1_manually_sampled_data/sampled_rustsb_rows.csv  \
    --plus-plus     ~/work/benchmark-policy-gap/data/1_manually_sampled_data/sampled_pp_rows.csv      \
    --multilingual  ~/work/benchmark-policy-gap/data/1_manually_sampled_data/sampled_sbmulti_rows.csv \
    --multi-sb      ~/work/benchmark-policy-gap/data/1_manually_sampled_data/sampled_multisb_rows.csv \
    --unified       ~/work/benchmark-policy-gap/data/1_manually_sampled_data/20260218_unified_sample.csv \
    --output        selected_65.jsonl
python3 -c '
import json
with open("selected_65.jsonl") as f, open("selected_65_gold_pred.jsonl","w") as out:
    for line in f:
        d = json.loads(line)
        out.write(json.dumps({"instance_id": d["instance_id"],
                              "model_name_or_path": "gold",
                              "model_patch": d["patch"]}) + "\n")
'

# run
RUN_ID="full65_$(date +%Y%m%d_%H%M%S)"
sg docker -c "python -m swebench.harness.run_evaluation \
    --predictions_path selected_65_gold_pred.jsonl \
    --dataset_name selected_65.jsonl \
    --run_id $RUN_ID \
    --max_workers 8 \
    --cache_level env \
    --namespace '' \
    --timeout 1800 2>&1 | tee logs/${RUN_ID}.log"

# read results
cat gold.${RUN_ID}.json
```

The repo on cascade already has `~/work/rscore-swe-bench/run_all_65.sh`
that wraps that final command. There's also `selected_65.jsonl` and
`selected_65_gold_pred.jsonl` already on disk so you can skip the
dataset build if nothing has changed.

---

## Prerequisites

* **x86_64 Linux** with **Docker access** (membership in the `docker`
  group, or `sg docker -c '…'` like the script above does on cascade).
  Apple Silicon won't work end-to-end because some image manifests
  fail under emulation.
* **Python 3.11** (3.10 also works, the venv on cascade is 3.11).
* **At least 60 GB free** on the docker storage path. aptos alone
  builds ~15 GB of artifacts; a couple of those running concurrently
  can fill smaller disks. Run `docker image prune -f` between full
  runs.
* SSH access to `cascade` if you want to use the existing setup, or
  any equivalent linux box. On cascade everything lives in
  `~/work/rscore-swe-bench/`.

---

## Why the dataset has to be built (don't just `cat` the CSVs)

The four sampling CSVs are heterogeneous:

* `sampled_rustsb_rows.csv` already has SWE-bench-format columns
  (`base_commit`, `patch`, `test_patch`, `FAIL_TO_PASS`, …).
* `sampled_pp_rows.csv` and `sampled_sbmulti_rows.csv` are
  index-only — instance_id + repo + benchmark — and need to be joined
  against `20260218_unified_sample.csv` to pull the eval-relevant
  columns.
* `sampled_multisb_rows.csv` is the worst: `base_commit` is empty,
  the SHA lives inside a JSON-stringified `base` blob, the patch is
  in `fix_patch`, and `FAIL_TO_PASS` / `PASS_TO_PASS` have to be
  reconstructed by filtering the `f2p_tests` / `p2p_tests` blobs by
  `test == 'FAIL'` and `fix == 'PASS'` (etc.).

`pipeline_scripts/0_data_construction/build_instances_from_csv.py`
does all of that and writes a single SWE-bench-format JSONL. Each row
also carries a `benchmark` field which the new harness uses to dispatch
to the right per-benchmark spec dict in
`swebench/harness/constants/rust.py`.

Dedup is intentional: 5 nushell PRs appear in two or three benchmarks
each, and the harness uses `instance_id` as a primary key, so 65 rows
collapse to 59 unique submissions. Nothing to fix there.

---

## Reading the results

Each run produces a `gold.<RUN_ID>.json` summary at the repo root and
per-instance directories under
`logs/run_evaluation/<RUN_ID>/gold/<instance_id>/`. The summary's
key fields:

```python
{
  "total_instances":     65,    # rows in selected_65.jsonl
  "submitted_instances": 59,    # post-dedup
  "completed_instances": 49,    # finished with a report
  "resolved_instances":  32,    # gold patch passed all F2P + P2P tests
  "unresolved_instances":17,    # cargo finished but report says some test failed
  "error_instances":     10,    # build error / docker timeout / disk fill / etc.
  "resolved_ids":   [...],      # the lists you usually want
  "unresolved_ids": [...],
  "error_ids":      [...]
}
```

Per-instance dirs contain `report.json` (resolved bool + tests_status),
`run_instance.log` (harness-side trace), `test_output.txt` (cargo
output between the START_TEST_OUTPUT and END_TEST_OUTPUT markers).
The build log lives separately under
`logs/build_images/instances/sweb.eval.x86_64.<id>__latest/build_image.log`.

When triaging, **always read the raw `test_output.txt` line that says
`test result: ok. N passed; M failed; … filtered out`** before
trusting `unresolved_instances`. It's common for a "failure" in the
report to be a test cargo never even ran (see [Spot-fix patterns](#spot-fix-patterns)).

---

## Spot-fix patterns

What the remaining 17 unresolved + 10 errored typically need.

### 1. Disk-fill flakes (≈9 false errors per full run)

Symptoms in `run_instance.log`: `OSError: [Errno 28] No space left on
device` (only on `aptos-labs__aptos-core-16152`) or
`requests.exceptions.ReadTimeout: ... read timeout=60` (any docker API
call) on **other instances** that build at the same time.

Cause: aptos pulls in the entire Move/Diem toolchain (~15 GB unique)
and several smaller crates also build concurrently. With
`max_workers=8` the daemon ends up serving ~9 long builds and one
container's network calls time out.

Fix: rerun just the affected instance_ids at lower concurrency.

```bash
# build a retry-only jsonl from the previous run's error_ids
python3 - <<'PY'
import json
prev = json.load(open("gold.<previous_run_id>.json"))
errs = set(prev["error_ids"]) - {"aptos-labs__aptos-core-16152"}  # skip the real-disk-fill one
with open("selected_65.jsonl") as src, open("retry.jsonl","w") as out, open("retry_pred.jsonl","w") as pred:
    for line in src:
        d = json.loads(line)
        if d["instance_id"] in errs:
            out.write(line)
            pred.write(json.dumps({"instance_id": d["instance_id"],
                                   "model_name_or_path": "gold",
                                   "model_patch": d["patch"]}) + "\n")
PY

RUN_ID="retry_$(date +%s)"
sg docker -c "python -m swebench.harness.run_evaluation \
    --predictions_path retry_pred.jsonl --dataset_name retry.jsonl \
    --run_id $RUN_ID --max_workers 4 --cache_level env --namespace '' --timeout 1800"
```

`max_workers=4` has been enough every time. Union the new run's
`resolved_ids` into the headline number.

### 2. axum: scope mismatch (12 of the 17 unresolved)

Symptom: `gold.json` reports e.g. `tokio-rs__axum-1934` with 423 F2P
"failures", but the raw `test_output.txt` ends with `test result: ok.
16 passed; 0 failed; … 264 filtered out`.

Cause: `AXUM_SPECS` in
`swebench/harness/constants/rust.py` ships per-PR `test_cmd`s that
narrow cargo to a specific test path (`-- routing::tests::fallback`
in the 1934 example) so they finish quickly. The F2P/P2P lists for
that PR were scraped from the **whole** axum repo, so any test outside
the spec's filter is graded as missing → failed.

Fix: broaden the per-PR `test_cmd` to actually exercise the full
F2P/P2P scope. The simplest swap is replacing the narrow filter with
the v1.0 generic fallback's command:

```python
# in swebench/harness/constants/rust.py, AXUM_SPECS["1934"]:
"test_cmd": ["cargo test -p axum --all-features --no-fail-fast"],
```

…or, alternatively, trim each PR's F2P/P2P list down to just the
tests the scoped command actually runs. (A) is less invasive and what
we've been doing.

Affected PRs (all axum): `412, 423, 529, 530, 533, 682, 755, 868,
892, 1189, 1469, 1810, 1934`. axum-3059 is the same idea but with
markdown doctests.

### 3. async-graphql ecosystem drift (9 of the 10 errors)

Affected PRs: `10, 170, 1043, 1048, 1099, 1228, 1346, 1491, 1524`
(versions v1.x / v4.x / v5.x / v6.x).

Symptom: instance-image build fails at `cargo test` step with one of:

* `feature 'edition2024' is required` — happens when `rust_version`
  is < 1.85.
* `error[E0382]: borrow of moved value` / similar — happens when
  `rust_version` is >= 1.85 because the new edition2024 lints fire on
  pre-2024 source code.
* `Tm - TimeDelta no impl` — actix-http 1.0.1 doesn't compile against
  modern `time` crate.

It's a real catch-22: the transitive deps that cargo resolves today
need rust ≥ 1.85, but the original async-graphql source needs rust
≤ 1.84. We documented this as a paper finding rather than fixing it
here.

If we want to push further, the fix path is the
`_write_cargo_lock_script(...)` / Cargo.lock fixture pattern that
tokio specs already use. We'd:

1. Pull each PR's original-CI Cargo.lock (or rebuild it once with
   `cargo update --precise time@0.3.x` etc. to roll back the
   offending crates).
2. Drop it in `swebench/harness/constants/fixtures/` as
   `async-graphql__async-graphql-<n>.Cargo.lock`.
3. Add `pre_install: _write_cargo_lock_script(...)` to the per-PR
   spec and pass `--locked` in `install` and `test_cmd`.

This is mechanical but takes a few hours per PR.

### 4. async-graphql 1559 unresolved (1 case)

`F2P pass=1 fail=0`, `P2P pass=648 fail=1`. Cargo reports `666 ok,
0 failed`. The lone "fail" is
`extensions::tracing::Tracing (line 24) - compile`, a doctest that
only exists when feature `tracing` is on; enabling that feature at
PR-1559's commit breaks compile because the dep is aliased
`tracinglib`. There's a comment in `constants/rust.py` flagging this.
Treat as a known false-fail.

### 5. Multi-sb data gaps (2 unresolved nushell)

`nushell-11169` and `nushell-13357` have non-empty F2P/P2P but the
multi-sb pipeline upstream reports them as both not-yet-passing and
not-yet-failing in some test slots. They show small numbers of P2P
failures that look like real flakes. Lower priority.

### 6. coreutils-8478 (1 unresolved)

The plus-plus instance. Real F2P/P2P mismatch; spec needs tightening
or the F2P/P2P list trimming. Lower priority.

---

## Where the fixes live

* **Per-repo, per-PR rust toolchain + test_cmd**:
  `swebench/harness/constants/rust.py`. Add a key under the
  appropriate `SPECS_FOO` dict. The lookup order is
  `version → instance_suffix (PR number) → None`.
* **Cargo.lock fixtures**:
  `swebench/harness/constants/fixtures/<owner>__<repo>-<pr>.Cargo.lock`
  + a `pre_install: _write_cargo_lock_script("…")` entry.
* **Dataset construction**:
  `pipeline_scripts/0_data_construction/build_instances_from_csv.py`
  in this repo. Re-run any time the sampled CSVs change.
* **Run config**: `run_all_65.sh` on cascade (just a wrapper around
  `run_evaluation.py` with the right flags). `--cache_level env` is
  what we want — `instance` would persist instance images across
  runs and fill disk.

---

## Mutation runs

Same harness, just a different predictions file. Build a
`<mutation>_pred.jsonl` where each row is

```json
{"instance_id": "<id>", "model_name_or_path": "<mutation_label>",
 "model_patch": "<the mutated patch>"}
```

Use the same `selected_65.jsonl` for `--dataset_name`. Each prediction
is graded against the gold F2P/P2P set, so a mutation that introduces
a policy violation but keeps the same test outcomes will still resolve
— that's the construct we're measuring in the paper. Expect the
runtime to roughly match a gold run; instance-image builds are cached
across runs as long as `cache_level >= env`.

---

## Latest numbers

Run on `test-parser-updates` branch (commits up through `5e88531`),
2026-05-02:

```
total_instances        65
submitted_instances    59
resolved_instances     32   (54%)
unresolved_instances   17
error_instances        10
```

Breakdown of remaining work in the [Spot-fix patterns](#spot-fix-patterns)
section above.
