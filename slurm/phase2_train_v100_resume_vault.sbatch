#!/bin/bash -l

#SBATCH --job-name=nanoalpha-train-v100-resume-vault
#SBATCH --partition=v100
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/home/vault/iwi5/iwi5232h/nano-alpha-llm-pretrain/artifacts/logs/slurm/%x-%j.out
#SBATCH --error=/home/vault/iwi5/iwi5232h/nano-alpha-llm-pretrain/artifacts/logs/slurm/%x-%j.err
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
set -euo pipefail
export PYTHONUNBUFFERED=1

REPO_ROOT=/home/hpc/iwi5/iwi5232h/nano-alpha-llm-pretrain
PYTHON_BIN=${REPO_ROOT}/.py312/bin/python

# High-capacity writable root (default) for checkpoints + MLflow + logs.
VAULT_ROOT=${VAULT_ROOT:-/home/vault/iwi5/iwi5232h/nano-alpha-llm-pretrain}
OUTPUT_DIR=${OUTPUT_DIR:-${VAULT_ROOT}/checkpoints/nano-alpha-130m-v100}
MLFLOW_ROOT=${MLFLOW_ROOT:-${VAULT_ROOT}/mlruns}
LOG_ROOT=${LOG_ROOT:-${VAULT_ROOT}/artifacts/logs}

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

export WANDB_PROJECT=nano-alpha-llm
export WANDB_RUN_GROUP=phase2_v100
export WANDB_INIT_TIMEOUT=300

mkdir -p "$OUTPUT_DIR" "$MLFLOW_ROOT" "$LOG_ROOT/slurm" "$LOG_ROOT/detailed"

# MLflow defaults to vault file store unless overridden in environment.
LOCAL_MLFLOW_URI="file:${MLFLOW_ROOT}"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-$LOCAL_MLFLOW_URI}"
export MLFLOW_EXPERIMENT_NAME="${MLFLOW_EXPERIMENT_NAME:-nano-alpha-130m-phase2}"

# If a remote MLflow URI is configured but unreachable, fall back to local vault store.
if [[ "$MLFLOW_TRACKING_URI" =~ ^https?:// ]]; then
  if ! MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" "$PYTHON_BIN" - <<'PY'
import os
import urllib.error
import urllib.request

url = os.environ["MLFLOW_TRACKING_URI"]
try:
    urllib.request.urlopen(url, timeout=8)
except urllib.error.HTTPError:
    # Server reached but path may require auth/specific endpoint.
    pass
except Exception:
    raise SystemExit(1)
PY
  then
    echo "Remote MLflow URI unreachable; falling back to local file store: $LOCAL_MLFLOW_URI"
    export MLFLOW_TRACKING_URI="$LOCAL_MLFLOW_URI"
  fi
fi

# Usage:
# sbatch slurm/phase2_train_v100_resume_vault.sbatch \
#   /path/to/checkpoint-5000 60000 [wandb_run_id]
RESUME_CHECKPOINT_INPUT=${1:-}
NEW_MAX_STEPS=${2:-60000}
RESUME_ID=${3:-}

if [ -z "$RESUME_CHECKPOINT_INPUT" ]; then
  echo "Usage: sbatch phase2_train_v100_resume_vault.sbatch <checkpoint_path> [new_max_steps] [wandb_run_id]"
  echo "Example: sbatch phase2_train_v100_resume_vault.sbatch checkpoints/nano-alpha-130m-v100/checkpoint-5000 60000"
  exit 1
fi

if [ -d "$RESUME_CHECKPOINT_INPUT" ]; then
  RESUME_CHECKPOINT=$(cd "$RESUME_CHECKPOINT_INPUT" && pwd)
elif [ -d "$REPO_ROOT/$RESUME_CHECKPOINT_INPUT" ]; then
  RESUME_CHECKPOINT=$(cd "$REPO_ROOT/$RESUME_CHECKPOINT_INPUT" && pwd)
elif [ -d "$OUTPUT_DIR/$(basename "$RESUME_CHECKPOINT_INPUT")" ]; then
  RESUME_CHECKPOINT=$(cd "$OUTPUT_DIR/$(basename "$RESUME_CHECKPOINT_INPUT")" && pwd)
else
  echo "Checkpoint directory not found: $RESUME_CHECKPOINT_INPUT"
  echo "Looked in:"
  echo "  - $RESUME_CHECKPOINT_INPUT"
  echo "  - $REPO_ROOT/$RESUME_CHECKPOINT_INPUT"
  echo "  - $OUTPUT_DIR/$(basename "$RESUME_CHECKPOINT_INPUT")"
  exit 1
fi

cd "$REPO_ROOT"

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" -u - <<'PY'
import sys

import torch

print("sys.executable =", sys.executable)
print("torch =", torch.__version__)

required = [
    "torch",
    "transformers",
    "datasets",
    "tokenizers",
]
missing = []
for mod in required:
    try:
        __import__(mod)
    except Exception as exc:
        missing.append((mod, str(exc)))

if missing:
    print("Dependency preflight failed:")
    for mod, err in missing:
        print(f"  - {mod}: {err}")
    raise SystemExit(2)

if not torch.cuda.is_available():
    print("GPU preflight failed: torch.cuda.is_available() is False. Refusing CPU fallback.")
    raise SystemExit(3)

print("CUDA available =", torch.cuda.is_available())
print("CUDA device count =", torch.cuda.device_count())
print("CUDA device 0 =", torch.cuda.get_device_name(0))

print("Dependency preflight OK")
PY

REPORT_TO=mlflow,wandb
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WANDB_API_KEY is not set; using MLflow-only logging for this run."
  REPORT_TO=mlflow
elif ! "$PYTHON_BIN" - <<'PY'
import urllib.request

try:
    urllib.request.urlopen("https://api.wandb.ai", timeout=8)
except Exception:
    raise SystemExit(1)
PY
then
  echo "W&B endpoint unreachable from compute node; falling back to MLflow-only logging for this run."
  REPORT_TO=mlflow
fi

RESUME_ARGS="--resume-from-checkpoint $RESUME_CHECKPOINT"
if [ -n "$RESUME_ID" ]; then
  RESUME_ARGS="$RESUME_ARGS --resume-id $RESUME_ID"
fi

{
echo "[$(date '+%F %T')] [train-sbatch] Resuming phase2 training on V100 (vault-backed)"
echo "[$(date '+%F %T')] [train-sbatch] Checkpoint: $RESUME_CHECKPOINT"
echo "[$(date '+%F %T')] [train-sbatch] New max steps: $NEW_MAX_STEPS"
echo "[$(date '+%F %T')] [train-sbatch] Output dir: $OUTPUT_DIR"
echo "[$(date '+%F %T')] [train-sbatch] Report to: $REPORT_TO"
echo "[$(date '+%F %T')] [train-sbatch] MLflow URI: ${MLFLOW_TRACKING_URI}"
echo "[$(date '+%F %T')] [train-sbatch] MLflow experiment: ${MLFLOW_EXPERIMENT_NAME}"
"$PYTHON_BIN" -u scripts/train.py \
  --train-file data/processed/phase1_v2_de_eval/train_corpus.jsonl \
  --val-file data/processed/phase1_v2_de_eval/val_corpus.jsonl \
  --tokenizer-file artifacts/tokenizer/nano_alpha_bpe.tokenizer.json \
  --output-dir "$OUTPUT_DIR" \
  --max-steps "$NEW_MAX_STEPS" \
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
  --fp16 \
  --report-to "$REPORT_TO" \
  --mlflow-tracking-uri "$MLFLOW_TRACKING_URI" \
  --mlflow-experiment-name "$MLFLOW_EXPERIMENT_NAME" \
  $RESUME_ARGS
echo "[$(date '+%F %T')] [train-sbatch] Finished phase2 training on V100"
} 2>&1 | tee "$LOG_ROOT/detailed/train_informer_v100_resume_vault_${SLURM_JOB_ID}.txt"
