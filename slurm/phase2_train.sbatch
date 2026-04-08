#!/bin/bash -l

#SBATCH --job-name=nanoalpha-train
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=artifacts/logs/slurm/%x-%j.out
#SBATCH --error=artifacts/logs/slurm/%x-%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT=/home/hpc/iwi5/iwi5232h/nano-alpha-llm-pretrain
PYTHON_BIN=${REPO_ROOT}/.py312/bin/python
PROJECT_ROOT=${REPO_ROOT}

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python environment not found at $PYTHON_BIN"
  exit 1
fi

module purge
module load python/3.12-conda

ENV_FILE=${REPO_ROOT}/.env
if [ -f "$ENV_FILE" ]; then
  set -a
  source "$ENV_FILE"
  set +a
fi

if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WANDB_API_KEY is not set. Add it to ${ENV_FILE} or export it before submission."
  exit 1
fi

export WANDB_PROJECT=nano-alph-llm
export WANDB_RUN_GROUP=phase2

cd "$PROJECT_ROOT"
mkdir -p artifacts/logs/slurm artifacts/logs/detailed checkpoints/nano-alpha-130m

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" -u - <<'PY'
import sys
import torch
print('sys.executable =', sys.executable)
print('torch =', torch.__version__)

required = [
  'torch',
  'transformers',
  'datasets',
  'tokenizers',
]
missing = []
for mod in required:
  try:
    __import__(mod)
  except Exception as exc:
    missing.append((mod, str(exc)))

if missing:
  print('Dependency preflight failed:')
  for mod, err in missing:
    print(f'  - {mod}: {err}')
  raise SystemExit(2)

print('Dependency preflight OK')
PY

{
echo "[$(date '+%F %T')] [train-sbatch] Starting phase2 training"
"$PYTHON_BIN" -u scripts/train.py \
  --train-file data/processed/phase1_v2_de_eval/train_corpus.jsonl \
  --val-file data/processed/phase1_v2_de_eval/val_corpus.jsonl \
  --tokenizer-file artifacts/tokenizer/nano_alpha_bpe.tokenizer.json \
  --output-dir checkpoints/nano-alpha-130m \
  --max-steps 5000 \
  --logging-steps 20 \
  --save-steps 500 \
  --eval-steps 500 \
  --train-batch-size 2 \
  --eval-batch-size 2 \
  --grad-accum-steps 16 \
  --seq-length 1024 \
  --learning-rate 3e-4 \
  --weight-decay 0.1 \
  --warmup-ratio 0.03 \
  --bf16 \
  --report-to wandb
echo "[$(date '+%F %T')] [train-sbatch] Finished phase2 training"
} 2>&1 | tee artifacts/logs/detailed/train_informer_${SLURM_JOB_ID}.txt
