#!/bin/bash
# Chained runner: resume gold (low concurrency), then run all 20 mutation files at higher concurrency.
# Resume is free: harness short-circuits on existing report.json.
set -euo pipefail
source ~/rb-venv/bin/activate
export GITHUB_TOKENS=$(cat ~/.github_token)
cd ~/Rust-bench

mkdir -p ~/eval_logs
SUMMARY=~/eval_logs/SUMMARY.log

run_eval() {
  local run_id="$1"
  local pred_path="$2"
  local workers="$3"
  local log=~/eval_logs/${run_id}.log

  echo "========================================" | tee -a $SUMMARY
  echo "[$(date -u)] START $run_id (workers=$workers pred=$pred_path)" | tee -a $SUMMARY
  echo "========================================" | tee -a $SUMMARY

  python -m swebench.harness.run_evaluation \
    --dataset_name user2f86/rustbench \
    --predictions_path "$pred_path" \
    --run_id "$run_id" \
    --max_workers "$workers" \
    --cache_level instance \
    --split train \
    --config_path swebench/harness/logs/config.json \
    --build_image_only 0 \
    2>&1 | tee -a $log

  # Summarize this run
  local total=$(find ~/Rust-bench/logs/run_evaluation/$run_id -name "report.json" 2>/dev/null | wc -l)
  local resolved=$(find ~/Rust-bench/logs/run_evaluation/$run_id -name "report.json" -exec grep -l '"resolved": true' {} \; 2>/dev/null | wc -l)
  echo "[$(date -u)] DONE $run_id: total=$total resolved=$resolved" | tee -a $SUMMARY
  echo "" | tee -a $SUMMARY
}

# --- Phase 1: GOLD ---
run_eval "gold_run1" "gold" 8

# --- Phase 2: all 20 mutation files (higher worker count) ---
for patch in ~/patch/*.jsonl; do
  name=$(basename "$patch" .jsonl)
  run_eval "${name}_run1" "$patch" 16
done

echo "[$(date -u)] ALL DONE" | tee -a $SUMMARY
df -h / | tee -a $SUMMARY
docker system df | tee -a $SUMMARY
