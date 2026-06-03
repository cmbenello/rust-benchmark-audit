#!/bin/bash
# Gold sanity run: build images + run eval with gold predictions.
# Applies Xihua's recommendations: fresh run_id, cache_level=instance, moderate workers.
set -euo pipefail
source ~/rb-venv/bin/activate
export GITHUB_TOKENS=$(cat ~/.github_token)
cd ~/Rust-bench

mkdir -p ~/eval_logs
RUN_ID="gold_$(date +%Y%m%d_%H%M%S)"
LOG=~/eval_logs/${RUN_ID}.log

echo "=== RUN_ID=$RUN_ID ===" | tee -a $LOG
echo "=== start: $(date -u) ===" | tee -a $LOG
echo "=== disk before ===" | tee -a $LOG
df -h / | tee -a $LOG

python -m swebench.harness.run_evaluation \
  --dataset_name user2f86/rustbench \
  --predictions_path gold \
  --run_id "$RUN_ID" \
  --max_workers 8 \
  --cache_level instance \
  --split train \
  --config_path swebench/harness/logs/config.json \
  --build_image_only 0 \
  2>&1 | tee -a $LOG

echo "=== end: $(date -u) ===" | tee -a $LOG
echo "=== disk after ===" | tee -a $LOG
df -h / | tee -a $LOG
echo "=== docker df ===" | tee -a $LOG
docker system df | tee -a $LOG
echo "DONE $RUN_ID" | tee -a $LOG
