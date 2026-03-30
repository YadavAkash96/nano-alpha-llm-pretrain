# Prepare Data Runbook

This document explains how to reproduce the dataset preparation stage using [scripts/prepare_data.py](scripts/prepare_data.py).

## Goal

Build a training corpus for DE and EN pretraining and export a separate evaluation file.

## Prerequisites

1. Conda environment exists at ./.py312 in this repository
2. Hugging Face token is set in [.env](.env):

HF_TOKEN=your_token_here

## Environment activation

1. module load python/3.12-conda
2. source "$(conda info --base)/etc/profile.d/conda.sh"
3. conda activate ./.py312

## Smoke test command

python scripts/prepare_data.py \
  --train-source opus100 \
  --de-max-docs 5000 \
  --en-max-docs 5000 \
  --min-chars 30 \
  --eval-max-samples 100 \
  --eval-dataset CohereForAI/Global-MMLU \
  --eval-config de \
  --eval-split test \
  --eval-output-name eval_mcq_global_mmlu_de.jsonl \
  --val-ratio 0.02 \
  --output-dir data/processed/smoke_v3

## Full training corpus build

Use this before MinHash.

python scripts/prepare_data.py \
  --train-source wiki \
  --wiki-dataset wikimedia/wikipedia \
  --wiki-date 20231101 \
  --de-max-docs 50000 \
  --en-max-docs 50000 \
  --min-chars 200 \
  --eval-max-samples 500 \
  --eval-dataset CohereForAI/Global-MMLU \
  --eval-config de \
  --eval-split test \
  --eval-output-name eval_mcq_global_mmlu_de.jsonl \
  --val-ratio 0.02 \
  --output-dir data/processed/phase1_v1_de_eval

## Scale-up run to 100k plus 100k

Run only after phase1_v1_de_eval completes and disk/time are acceptable.

python scripts/prepare_data.py \
  --train-source wiki \
  --wiki-dataset wikimedia/wikipedia \
  --wiki-date 20231101 \
  --de-max-docs 100000 \
  --en-max-docs 100000 \
  --min-chars 200 \
  --eval-max-samples 1000 \
  --eval-dataset CohereForAI/Global-MMLU \
  --eval-config de \
  --eval-split test \
  --eval-output-name eval_mcq_global_mmlu_de.jsonl \
  --val-ratio 0.02 \
  --output-dir data/processed/phase1_v2_de_eval

## Output files per run

1. train_corpus.jsonl
2. val_corpus.jsonl
3. eval output file from --eval-output-name
4. prepare_data_report.json

## Freeze canonical dataset snapshot

After selecting the canonical folder (for example phase1_v2_de_eval), freeze it before MinHash and training.

1. Generate checksums:

sha256sum data/processed/phase1_v2_de_eval/train_corpus.jsonl \
  data/processed/phase1_v2_de_eval/val_corpus.jsonl \
  data/processed/phase1_v2_de_eval/eval_mcq_global_mmlu_de.jsonl \
  data/processed/phase1_v2_de_eval/prepare_data_report.json \
  > data/processed/phase1_v2_de_eval/SHA256SUMS.txt

2. Create a manifest with command and git revision:

{
  echo "run_date=$(date -Iseconds)"
  echo "git_commit=$(git rev-parse --short HEAD)"
  echo "dataset_dir=data/processed/phase1_v2_de_eval"
  echo "prepare_command=python scripts/prepare_data.py --train-source wiki --wiki-dataset wikimedia/wikipedia --wiki-date 20231101 --de-max-docs 100000 --en-max-docs 100000 --min-chars 200 --eval-max-samples 1000 --eval-dataset CohereForAI/Global-MMLU --eval-config de --eval-split test --eval-output-name eval_mcq_global_mmlu_de.jsonl --val-ratio 0.02 --output-dir data/processed/phase1_v2_de_eval"
} > data/processed/phase1_v2_de_eval/FREEZE_MANIFEST.txt

3. Verify checksums later when reusing data:

sha256sum -c data/processed/phase1_v2_de_eval/SHA256SUMS.txt

## Disk and runtime decision gate for 100k plus 100k

1. Check output size from phase1_v1:
   du -sh data/processed/phase1_v1_de_eval
2. Check row counts:
   wc -l data/processed/phase1_v1_de_eval/train_corpus.jsonl data/processed/phase1_v1_de_eval/val_corpus.jsonl
3. If phase1_v1 size and runtime are acceptable for your quota and schedule, run phase1_v2.

## If you prefer local download then copy to cluster

Yes, you can download locally and copy artifacts.

Recommended copy target:

./data/processed/

After copying, verify integrity with:

1. wc -l on jsonl files
2. head -n 1 prepare_data_report.json
3. a quick Python read of first few lines

## Notes

1. Current script supports loading token values from [.env](.env).
2. MinHash contamination filtering is intentionally the next stage and should run after this dataset is frozen.

## Troubleshooting

1. Warning about missing token:
Set HF_TOKEN in [.env](.env). The script loads [.env](.env) automatically.

2. Dataset scripts no longer supported:
This environment uses a datasets version where some legacy script-based datasets fail. Prefer parquet-hosted datasets such as wikimedia/wikipedia and CohereForAI/Global-MMLU.

3. x_csqa not found:
Use CohereForAI/Global-MMLU with --eval-config de and --eval-split test.

4. Conda activation issue with source .py312/bin/activate:
This environment is a conda-prefix env. Use module load python/3.12-conda, source "$(conda info --base)/etc/profile.d/conda.sh", then conda activate ./.py312.

5. Do not overwrite frozen output:
Always write new runs to a new output directory name. Keep canonical frozen dataset immutable.
