#!/bin/zsh
set -euo pipefail
cd /Users/gengliu/political-bias-shareable
PY=/Users/gengliu/anaconda3/envs/torch_m1/bin/python3
BATCH_ROOT=/Users/gengliu/political-bias-shareable/outputs/multimodel_840_batch_20260424_103114_pty
export PYTORCH_ENABLE_MPS_FALLBACK=1
MODELS=(
  'mistralai/Mistral-7B-Instruct-v0.2'
  'meta-llama/Meta-Llama-3-8B-Instruct'
)
slugify() {
  echo "$1" | tr '/:' '__' | tr '-' '_' | tr '.' '_'
}
for MODEL in "${MODELS[@]}"; do
  SLUG=$(slugify "$MODEL")
  OUTDIR="$BATCH_ROOT/$SLUG"
  mkdir -p "$OUTDIR"
  echo "[$(date '+%F %T')] START $MODEL -> $OUTDIR"
  "$PY" run_experiment.py \
    --model "$MODEL" \
    --input-csv data/ideoinst_clean/ideology_840_opinion.csv \
    --device mps \
    --languages en it fr es de \
    --control-method rfm \
    --prompt-format opinion \
    --coefs 0.0 0.8 1.5 2.5 4.0 \
    --train-pairs 600 --val-pairs 120 --test-pairs 120 \
    --batch-size 1 --n-components 1 --rfm-iters 8 \
    --selection-metric auc \
    --max-new-tokens 30 \
    --seed 0 \
    --output-dir "$OUTDIR"
  echo "[$(date '+%F %T')] DONE  $MODEL"
done
