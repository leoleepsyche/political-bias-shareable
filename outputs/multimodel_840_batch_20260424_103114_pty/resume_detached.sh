#!/bin/zsh
set -euo pipefail

cd /Users/gengliu/political-bias-shareable/outputs/multimodel_840_batch_20260424_103114_pty

echo "[$(date '+%F %T')] RESUME detached batch"
caffeinate -dim zsh -c './run_remaining.sh 2>&1 | tee -a batch.log'
echo "[$(date '+%F %T')] RESUME detached batch finished"
