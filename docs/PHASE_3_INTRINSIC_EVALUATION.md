# Phase 3 Intrinsic Evaluation

This phase evaluates each training checkpoint using intrinsic language modeling metrics and uncertainty metrics.

## Main methods
1. Cross entropy and perplexity
2. Token accuracy and average confidence
3. ECE and Brier score
4. Reliability diagrams
5. Selective prediction risk coverage curves
6. Domain or language breakdown when metadata is available

## Entry points
1. Script: `scripts/phase3_intrinsic_eval.py`
2. SLURM job: `slurm/phase3_intrinsic_eval_v100.sbatch`

## Outputs
1. Per checkpoint JSON metric files
2. Summary CSV and JSON
3. PNG plots for reliability and selective prediction
