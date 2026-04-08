# Phase 3: Intrinsic Evaluation

## Definition
Intrinsic evaluation measures the model as a probabilistic language model on held-out text using logits and token probabilities.

## What We Implement
1. Cross-entropy (NLL) and perplexity per checkpoint
2. Token-level correctness and confidence
3. Calibration metrics: ECE and Brier score
4. Reliability diagram plots
5. Selective prediction (risk-coverage) curves
6. Domain/language metric breakdown using metadata (for example `lang`)

## Why It Matters
1. Perplexity alone does not diagnose overconfidence.
2. Calibration quality is critical for safe downstream usage.
3. Checkpoint-level intrinsic metrics are cheap and stable compared to full benchmark suites.

## What It Evaluates
1. Distribution fit on held-out text
2. Confidence calibration against correctness
3. Abstention behavior under confidence thresholds
4. Domain-specific weakness patterns

## Evaluation Methods Table
| Method | Type | What it measures | Output artifact |
|---|---|---|---|
| Cross-Entropy / NLL | Intrinsic probabilistic metric | Average negative log-likelihood on held-out tokens | Metrics JSON, summary CSV |
| Perplexity | Intrinsic probabilistic metric | Exponentiated NLL, language-model fit quality | Metrics JSON, summary CSV |
| Token Accuracy | Intrinsic classification-style metric | Fraction of next-token top-1 predictions that match labels | Metrics JSON, summary CSV |
| Average Confidence | Confidence metric | Mean top-1 softmax confidence | Metrics JSON, summary CSV |
| ECE (Expected Calibration Error) | Calibration metric | Gap between confidence and empirical accuracy across bins | Metrics JSON, summary CSV |
| Brier Score | Calibration metric | Squared error between confidence and correctness | Metrics JSON, summary CSV |
| Reliability Diagram | Calibration visualization | Confidence vs observed accuracy by bins | PNG per checkpoint |
| Selective Prediction Curve (Risk-Coverage) | Uncertainty/abstention analysis | Risk as coverage increases under confidence-based selection | PNG per checkpoint |
| Domain/Language Breakdown | Slice analysis | NLL/PPL/accuracy per metadata key (for example `lang`) | Metrics JSON per checkpoint |

## Runtime and Execution Mode
1. Preferred execution target: V100 GPU via SLURM job.
2. CPU execution is supported for smoke tests but much slower for full checkpoint sweeps.
3. Use the scheduler script in the training repo:
	[Phase 3 V100 SLURM](../../nano-alpha-llm-pretrain/slurm/phase3_intrinsic_eval_v100.sbatch)

## Failure-Risk Notes (Compared to Training)
1. This phase is evaluation-only (forward pass only): no optimizer state updates, no gradient steps, and no checkpoint writes.
2. The SLURM job performs explicit preflight checks before starting:
	1. Python executable exists
	2. required packages import successfully
	3. CUDA is available on allocated node
	4. checkpoints directory, eval file, and tokenizer file exist
3. This reduces runtime failure risk compared to long training jobs, but does not eliminate failures caused by cluster/node issues.

## Dependency Note
1. Phase 3 uses already expected dependencies in the project environment:
	1. torch
	2. transformers
	3. numpy
	4. matplotlib
2. No additional package installation is required if the existing project environment is intact.

## Reproducibility: Parameter Reference
### A) SLURM positional parameters
| Position | Name | Default | Meaning |
|---|---|---|---|
| 1 | checkpoints_dir | `checkpoints/nano-alpha-130m-v100` | Root directory containing `checkpoint-*` and optional `final` |
| 2 | eval_file | `data/processed/phase1_v2_de_eval/val_corpus.jsonl` | JSONL evaluation corpus |
| 3 | output_dir | `artifacts/eval/phase3_intrinsic` | Output directory for JSON/CSV/PNG artifacts |
| 4 | max_eval_tokens | `200000` | Token cap per checkpoint (0 = no cap) |
| 5 | max_samples | `0` | Maximum JSONL rows to read (0 = all) |

### B) Python script parameters currently passed by the SLURM job
| Flag | Value Source | Purpose |
|---|---|---|
| `--checkpoints-dir` | Positional arg 1 | Select checkpoint set to evaluate |
| `--eval-file` | Positional arg 2 | Select validation corpus |
| `--tokenizer-file` | Fixed path in SLURM script | Tokenizer fallback for checkpoints without tokenizer files |
| `--output-dir` | Positional arg 3 | Where metrics/plots are written |
| `--max-eval-tokens` | Positional arg 4 | Runtime vs coverage control |
| `--max-samples` | Positional arg 5 | Quick subset evaluation |
| `--device` | Fixed `cuda` in SLURM script | Forces GPU evaluation profile |

### C) Optional advanced args (not fixed by default)
1. `--seq-length` (default 1024)
2. `--num-bins` (default 15)
3. `--selective-points` (default 20)
4. `--text-key` (default `text`)
5. `--domain-key` (default `lang`)

For reproducibility, record all non-default values in run notes or alongside evaluation artifacts.

## Inputs and Outputs
Inputs:
1. Checkpoint directory (`checkpoint-*` and `final`)
2. Validation JSONL with `text` and optional domain key

Outputs:
1. Per-checkpoint metrics JSON
2. Reliability diagram PNG
3. Selective prediction curve PNG
4. Aggregate CSV summary across checkpoints
