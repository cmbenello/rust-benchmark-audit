# Harness Run Summary (2026-02-28, Partial)

## Run Metadata
- Start command:
  - `python3 -u pipeline_scripts/2_analysis_runs_and_summary/bare_run_all.py --instances-jsonl data/instances_unified.jsonl --mutations gs,unwrap,unsafe,panic --run-eval --docker-host unix://$HOME/.docker/run/docker.sock --mutated-out-jsonl results/mutated_instances_20260228_005456.jsonl --policy-out-jsonl results/policy_check_results_20260228_005456.jsonl --predictions-dir /tmp/mutated_patches_20260228_005456 --eval-output-dir results/harness_eval_20260228_005456`
- Input instances: `27`
- Mutated rows generated: `108`
- Prediction jobs generated: `12`
- Run status: interrupted during `SWE-bench/SWE-bench_Multilingual` `panic` job due long wall-clock runtime.

## Completed Harness Portion
- Completed run id: `SWE-bench_SWE-bench_Multilingual_gs_20260228_005456`
- Completed benchmark/mutation: `SWE-bench/SWE-bench_Multilingual` + `gs`
- Completed instance reports: `10`
- Resolved: `9`
- Unresolved: `1`
- Unresolved instance: `uutils__coreutils-6377`

## Generated Artifacts
- `results/mutated_instances_20260228_005456.jsonl`
- `results/policy_check_results_20260228_005456.jsonl`
- `logs/run_evaluation/SWE-bench_SWE-bench_Multilingual_gs_20260228_005456`
- `logs/run_evaluation/SWE-bench_SWE-bench_Multilingual_panic_20260228_011017` (partial)

## Policy Count Totals (All 108 rows)
- `gs`: `unwrap=6`, `unsafe=1`, `panic=1`, `unsafe_without_safety_comment=0`
- `unwrap`: `unwrap=33`, `unsafe=1`, `panic=1`, `unsafe_without_safety_comment=0`
- `unsafe`: `unwrap=6`, `unsafe=28`, `panic=1`, `unsafe_without_safety_comment=27`
- `panic`: `unwrap=6`, `unsafe=1`, `panic=4`, `unsafe_without_safety_comment=0`
