# Replication Package

> **Note:** This is a *mostly*-replication package for **"Is it correct to be 'unsafe'? A pilot study of Rust safety in CodeGen SE benchmarks"**. A true replication package would subset this directory and remove unused data, scripts, and printouts. These superfluous files are retained for posterity and ongoing development.

---

## `/data` — Benchmark Data & Mutations

All benchmark datasets and mutations used in the pilot analysis.

| Directory | Contents |
|---|---|
| `0_benchmark-sets` | Full versions of the three unmutated benchmarks |
| `1_manually_sampled_data` | Rows manually sampled from each dataset for project coding policies (per-benchmark and unified) |
| `2_frozen_mutated_patches` | Claude-mutated sampled patches (`gold standard`, `panic`, `unsafe`, `unwrap`) |

---

## `/pipeline_scripts` — Analysis Pipeline Code

All code used in the analysis pipeline.

- **`0_data_construction`** — Scripts for checking the presence of policy-risky language within gold standard patches. Includes `per-project-safety-constructs.txt`, which contains manual annotations to project safety policies.
- **`1_analysis_runs_and_summary`** — Scripts for the analysis pipeline. See the README for more information.

---

## `/results` — Evaluation Harness Outputs

| Directory | Contents |
|---|---|
| `20260225_results` | Final policy check results for gold-standard patches |
| `20260309_harness_eval_results` | All results from the SWE-bench evaluation harness for `multilingual` and `plus-plus` |
