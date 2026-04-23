# AWS Rust-SWE-Bench Harness Attempt — 2026-04-23

## Summary

After the rust-swe-bench authors replied via email attributing local harness
failures to "local evaluation environment overhead" (Xihua, Apr 15), we
provisioned a fresh AWS EC2 VM sized well above their published minimums and
attempted to re-run their harness end-to-end.

**Outcome:** 162 of 500 gold instances completed (93.8% resolved) before the
VM became unresponsive — symptoms consistent with fork / process-table
exhaustion. The hang recurred on retry.

See `../../paper_updated.tex` (or `paper_updated.tex` in this dir) for the
Overleaf paragraph summarising this in the paper's Limitations section.

## Hardware

- **Instance:** AWS `c6i.8xlarge` (us-east-2)
- **CPU:** 32 vCPU (Intel Ice Lake)
- **RAM:** 61 GB
- **Disk:** 500 GB `gp3` SSD (6000 IOPS, 250 MB/s throughput)
- **OS:** Ubuntu 22.04.5 LTS, x86_64
- **Docker:** 29.4.0 (Community)

This is roughly **3× the harness' published minimum** (8 CPU / 16 GB / 120 GB)
on every dimension.

Note (Apr 23 email from Xihua): the rust-swe-bench authors themselves used
an **80-CPU / 300+ GB RAM / 1 TB** machine — i.e. ~10× the published minimum.
Their latest mitigation suggestion was `max_workers=1`, which we retried on
Apr 23. **The VM hung a third time within ~60 minutes at `max_workers=1`**
(faster than the 4-6hr hangs at `max_workers=8`), so concurrency is not the
root cause. Auto-stopped after the third hang.

## What we did

Applied all four of Xihua's initial troubleshooting steps:

1. Started from a fresh VM (no prior Docker cache).
2. Used fresh `run_id`s.
3. Reduced `max_workers` to 8 (below their stated ceiling of 24).
4. Inspected per-instance logs on failure.

## Patches required to run the harness at all

Two issues had to be fixed in the upstream harness before it would run:

### 1. `dockerfiles.py` — apt install flakiness
The default Dockerfile calls `apt update && apt install -y ...` without any
retry logic, against upstream Ubuntu mirrors that are unreliable from EC2
(we observed ~130 kB/s vs. 23 MB/s on the AWS regional mirror). See
`harness_patches/dockerfiles.py`:

- Wrapped apt calls in a 5-attempt retry loop with `--fix-missing`.
- Added `sed` to rewrite the apt sources to the AWS regional mirror
  (`us-east-2.ec2.archive.ubuntu.com`) before the first apt invocation.

### 2. `run_evaluation.py` — instance images never removed
`run_instance()` accepts a `rm_image: bool` argument used to decide whether to
remove the per-instance Docker image after evaluation. In
`run_instances()` (line 304 in the unpatched file), the `rm_image` positional
is **hardcoded `False`**, and the accompanying `clean_images()` call at the
end of `main()` is **commented out**.

Each per-instance image is ≈16 GB. With `cache_level=instance` (as documented
in the README's "build images" command), images are retained indefinitely.
On a 500 GB disk this saturates in ~10 hours of runtime. We patched
`rm_image=True` so images are removed after each test. See
`harness_patches/run_evaluation.py` line 305.

## Run sequence

1. `bootstrap.sh` — VM setup (Docker, Python, nofile limits, daemon tuning).
2. `run_gold.sh` — initial gold-only run, `max_workers=8` (first attempt).
3. `run_all.sh` — chained gold + 20 mutation files (superseded the above
   after the patches went in).
4. `run_gold_mw1.sh` (on VM only) — Xihua's Apr 23 retry with
   `max_workers=1`.

## Observations

- **Pace (workers=8, post-patch):** ~30 instances/hour. ETA for full gold
  ~14 hours; ETA for all 500 × 20 mutation files ~6-7 days.
- **VM hang ×2:** At ~4-6 hours of runtime, the VM stopped responding. Port
  22 accepted TCP connections but sshd could not complete banner exchange.
  Host CPU dropped to ~4%, EBS queue length normal, memory showed 30+ GB
  free. Consistent with fork exhaustion / process table full from the many
  parallel `cargo` jobs inside evaluation containers. Required reboot via
  EC2 API both times.
- **162/500 gold resolved at 93.8%:** Not 100% — the 10 unresolved cases
  look like test-status mismatches (e.g. `aya-rs__aya-656` had 5
  `PASS_TO_PASS` failures; `async-graphql` had several). Could be flaky
  tests or dataset drift; didn't dig deeper since this was a feasibility
  check.

## Contents of this directory

```
reports/                  # All 162 gold reports (report.json per instance)
logs/                     # Eval summary + per-run console logs
harness_patches/          # Patched dockerfiles.py + run_evaluation.py
bootstrap.sh              # VM setup script
run_all.sh, run_gold.sh   # Orchestration scripts used on the VM
paper_updated.tex         # Updated Overleaf source with new limitations text
```

## Cost

~$1.36/hr on-demand × ~6 days ≈ $200 in EC2 compute (plus EBS).

---

Report files retained in `reports/gold_run1/gold/<instance_id>/report.json`
for the 162 completed gold instances. Each has `patch_successfully_applied`,
`resolved`, and per-test status details.
