# MinHash Contamination Filter Runbook

This document explains how to run MinHash-based contamination filtering using [scripts/minhash_filter.py](scripts/minhash_filter.py).

## Goal

Remove training documents that are highly similar to evaluation content and generate a contamination report.

## Input assumptions

1. Canonical frozen dataset exists at [data/processed/phase1_v2_de_eval](data/processed/phase1_v2_de_eval).
2. Training file exists:
[data/processed/phase1_v2_de_eval/train_corpus.jsonl](data/processed/phase1_v2_de_eval/train_corpus.jsonl)
3. Eval file exists:
[data/processed/phase1_v2_de_eval/eval_mcq_global_mmlu_de.jsonl](data/processed/phase1_v2_de_eval/eval_mcq_global_mmlu_de.jsonl)

## Environment activation

1. module load python/3.12-conda
2. source "$(conda info --base)/etc/profile.d/conda.sh"
3. conda activate ./.py312

## Smoke test command

python scripts/minhash_filter.py \
  --train-file data/processed/phase1_v2_de_eval/train_corpus.jsonl \
  --eval-file data/processed/phase1_v2_de_eval/eval_mcq_global_mmlu_de.jsonl \
  --output-dir data/processed/phase1_v2_de_eval_minhash_smoke \
  --threshold 0.8 \
  --ngram-size 3 \
  --num-perm 128 \
  --progress-every 500 \
  --eval-progress-every 50 \
  --max-eval-samples 200 \
  --max-train-samples 5000

## Full filtering command

python scripts/minhash_filter.py \
  --train-file data/processed/phase1_v2_de_eval/train_corpus.jsonl \
  --eval-file data/processed/phase1_v2_de_eval/eval_mcq_global_mmlu_de.jsonl \
  --output-dir data/processed/phase1_v2_de_eval_minhash \
  --threshold 0.8 \
  --ngram-size 3 \
  --num-perm 128 \
  --progress-every 2000 \
  --eval-progress-every 100

## Output artifacts

1. [Filtered training corpus](data/processed/phase1_v2_de_eval_minhash/train_corpus.filtered.jsonl)
2. [Contamination report](data/processed/phase1_v2_de_eval_minhash/minhash_contamination_report.json)

## Verification

1. Check filtered line count and compare with original:
wc -l data/processed/phase1_v2_de_eval/train_corpus.jsonl data/processed/phase1_v2_de_eval_minhash/train_corpus.filtered.jsonl

2. Inspect summary in contamination report:
python - <<'PY'
import json
from pathlib import Path
p = Path('data/processed/phase1_v2_de_eval_minhash/minhash_contamination_report.json')
r = json.loads(p.read_text())
print(r['stats'])
PY

## Notes

1. Default threshold is 0.8 estimated Jaccard similarity via MinHash signatures.
2. Higher threshold keeps more data and removes only near-duplicates.
3. Lower threshold is stricter and may remove more borderline overlaps.
4. Run this once on the frozen dataset snapshot, then use filtered output for tokenizer/training.
5. Progress logs are enabled by default and print periodic throughput plus ETA-style updates.
6. If you want fewer logs, increase --progress-every and --eval-progress-every.
