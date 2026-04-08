#!/bin/bash -l

#SBATCH --job-name=nanoalpha-phase3-eval
#SBATCH --partition=rtx3080
#SBATCH --gres=gpu:rtx3080:1
#SBATCH --cpus-per-task=8
#SBATCH --time=16:00:00
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
mkdir -p artifacts/logs/slurm artifacts/logs/detailed artifacts/eval/phase3_intrinsic

# Optional positional args:
# 1) checkpoints_dir
# 2) eval_file
# 3) output_dir
# 4) max_eval_tokens (default: 200000)
# 5) max_samples (default: 0)
# Optional env var:
# PHASE3_EXTRA_ARGS="--seq-length 1024 --num-bins 15 --selective-points 20 --domain-key lang"
CHECKPOINTS_DIR=${1:-checkpoints/nano-alpha-130m-v100}
EVAL_FILE=${2:-data/processed/phase1_v2_de_eval/val_corpus.jsonl}
OUT_DIR=${3:-artifacts/eval/phase3_intrinsic}
MAX_EVAL_TOKENS=${4:-200000}
MAX_SAMPLES=${5:-0}
EXTRA_ARGS=${PHASE3_EXTRA_ARGS:-}

if [ ! -d "$CHECKPOINTS_DIR" ]; then
  echo "Checkpoints directory not found: $CHECKPOINTS_DIR"
  exit 2
fi

if [ ! -f "$EVAL_FILE" ]; then
  echo "Eval file not found: $EVAL_FILE"
  exit 2
fi

if [ ! -f "artifacts/tokenizer/nano_alpha_bpe.tokenizer.json" ]; then
  echo "Tokenizer file not found: artifacts/tokenizer/nano_alpha_bpe.tokenizer.json"
  exit 2
fi

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" -u - <<'PY'
import sys
import torch

print('sys.executable =', sys.executable)
print('torch =', torch.__version__)
print('torch CUDA runtime =', torch.version.cuda)

required = [
  'torch',
  'transformers',
  'matplotlib',
  'numpy',
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
  raise SystemExit(3)

if not torch.cuda.is_available():
  print('GPU preflight failed: torch.cuda.is_available() is False. Refusing CPU fallback in this SLURM profile.')
  raise SystemExit(4)

print('CUDA available =', torch.cuda.is_available())
print('CUDA device count =', torch.cuda.device_count())
print('CUDA device 0 =', torch.cuda.get_device_name(0))
print('Dependency preflight OK')
PY

{
echo "[$(date '+%F %T')] [phase3-sbatch] Starting intrinsic evaluation"
echo "[$(date '+%F %T')] [phase3-sbatch] checkpoints-dir: $CHECKPOINTS_DIR"
echo "[$(date '+%F %T')] [phase3-sbatch] eval-file: $EVAL_FILE"
echo "[$(date '+%F %T')] [phase3-sbatch] output-dir: $OUT_DIR"
echo "[$(date '+%F %T')] [phase3-sbatch] max-eval-tokens: $MAX_EVAL_TOKENS"
echo "[$(date '+%F %T')] [phase3-sbatch] max-samples: $MAX_SAMPLES"
echo "[$(date '+%F %T')] [phase3-sbatch] extra-args: ${EXTRA_ARGS:-<none>}"

"$PYTHON_BIN" -u scripts/phase3_intrinsic_eval.py \
  --checkpoints-dir "$CHECKPOINTS_DIR" \
  --eval-file "$EVAL_FILE" \
  --tokenizer-file artifacts/tokenizer/nano_alpha_bpe.tokenizer.json \
  --output-dir "$OUT_DIR" \
  --max-eval-tokens "$MAX_EVAL_TOKENS" \
  --max-samples "$MAX_SAMPLES" \
  --device cuda \
  $EXTRA_ARGS

echo "[$(date '+%F %T')] [phase3-sbatch] Finished intrinsic evaluation"
} 2>&1 | tee artifacts/logs/detailed/phase3_intrinsic_eval_${SLURM_JOB_ID}.txt
