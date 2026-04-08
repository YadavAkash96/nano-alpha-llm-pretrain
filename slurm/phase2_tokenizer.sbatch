#!/bin/bash -l

#SBATCH --job-name=nanoalpha-tokenizer
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
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

cd "$PROJECT_ROOT"
mkdir -p artifacts/logs/slurm artifacts/logs/detailed artifacts/tokenizer

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" -u - <<'PY'
import sys
print('sys.executable =', sys.executable)

required = [
  'tokenizers',
  'datasets',
  'json',
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
echo "[$(date '+%F %T')] [tokenizer-sbatch] Starting tokenizer training"
"$PYTHON_BIN" -u scripts/train_tokenizer.py \
  --train-file data/processed/phase1_v2_de_eval/train_corpus.jsonl \
  --vocab-size 32000 \
  --min-frequency 2 \
  --progress-every 1 \
  --output-dir artifacts/tokenizer \
  --name nano_alpha_bpe
echo "[$(date '+%F %T')] [tokenizer-sbatch] Finished tokenizer training"
} 2>&1 | tee artifacts/logs/detailed/tokenizer_informer_${SLURM_JOB_ID}.txt
