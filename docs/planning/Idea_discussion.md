# Nano-Alpha Project Plan (Restructured)

## Why This Restructure
The earlier plan treated evaluation as a narrow post-training check. This version defines evaluation as a first-class pipeline stage with explicit intrinsic metrics, benchmark suites, checkpoint dynamics, and reporting standards.

## Pipeline Overview
1. Phase 0: Environment and dependency setup
2. Phase 1: Data preparation, quality control, contamination filtering
3. Phase 2: Pre-training and checkpointing
4. Phase 3: Intrinsic evaluation (implemented first)
5. Phase 4: Benchmark evaluation (lm-eval-harness + task suites)
6. Phase 5: Pythia-style checkpoint dynamics analysis
7. Phase 6: Reporting and cross-run comparison

## Phase 0: Environment and Dependency Setup
Definition: Establish a reproducible runtime for GPU training and evaluation.

What we implement:
1. Isolated Python environment pinned to compatible package versions
2. Core dependencies for training, evaluation, calibration, and tracking
3. Environment validation (GPU availability, CUDA compatibility, package import preflight)

Why:
1. Avoid version drift and hidden reproducibility failures
2. Ensure training and evaluation scripts run consistently across sessions/jobs

What it evaluates:
1. Environment readiness for all downstream phases
2. Compatibility of PyTorch/CUDA stack with HPC hardware

Key dependencies:
1. torch
2. transformers
3. datasets
4. tokenizers
5. accelerate
6. wandb
7. mlflow
8. matplotlib
9. scikit-learn
10. datasketch

## Phase 1: Data Preparation and Contamination Control
Definition: Build a cleaned train/validation split and verify contamination controls against evaluation-oriented data.

What we implement:
1. Data ingestion and canonical JSONL outputs
2. Dataset QA reports and checksums/manifests
3. MinHash-based contamination filtering between train and evaluation-like sets
4. Reproducible split generation and metadata logging

Why:
1. Data quality dominates small-model outcomes
2. Leakage from eval-like content can invalidate benchmark claims
3. Reproducible data artifacts are required for trustworthy comparisons across runs

What it evaluates:
1. Integrity and consistency of source data
2. Leakage risk level and filtering effectiveness
3. Stability of corpus preparation pipeline

## Phase 2: Pre-Training and Checkpointing
Definition: Train the 130M Llama-style model from scratch with regular checkpointing and robust resume capability.

What we implement:
1. Tokenizer training and fixed tokenizer artifacts
2. Training loop with HF Trainer on prepared corpus
3. Step-based checkpointing (dense enough for analysis and resume)
4. MLflow/W&B compatible logging with HPC-aware fallback behavior
5. Resume workflow from latest checkpoint with total-step targeting

Why:
1. Checkpoints are needed both for fault tolerance and evaluation dynamics
2. Step-based tracking enables consistent phase-3/4/5 comparisons
3. Robust resume flow is mandatory under scheduler walltime limits

What it evaluates:
1. Optimization progress via train/eval loss traces
2. Throughput and stability under cluster constraints
3. Quality of produced checkpoint set for later evaluation phases

## Non-Negotiable Evaluation Principles
1. Evaluation is multi-metric, not just loss.
2. Intermediate checkpoints are required, not optional.
3. Benchmarking must include general and German-focused tasks.
4. Results must support comparison across future runs and model variants.

## Phase 3: Intrinsic Evaluation (Current Implementation Target)
Definition: Evaluate the model as a probabilistic language model on held-out text, without generation-first shortcuts.

What we implement:
1. Per-checkpoint cross-entropy and perplexity
2. Token-level accuracy and confidence extraction from logits
3. Calibration metrics: ECE, Brier score
4. Reliability diagrams
5. Selective prediction and abstention curves
6. Domain/language breakdown (when metadata like `lang` is present)

Why:
1. Perplexity alone can hide overconfidence or calibration failures.
2. Intrinsic uncertainty quality must be measured before downstream benchmarks.
3. This gives stable, low-cost checkpoint-level diagnostics.

What it evaluates:
1. Fit to held-out distribution (NLL/PPL)
2. Confidence alignment with correctness (ECE/Brier)
3. Risk-coverage behavior for abstention policies
4. Differences across domains/languages in the same evaluation set

## Phase 4: Benchmark Evaluation
Definition: Task-level evaluation on standardized benchmark suites using log-likelihood style scoring.

Primary runner:
1. lm-eval-harness

Target tasks:
1. MMLU
2. ARC
3. HellaSwag
4. TruthfulQA
5. IGEL (German-centric evaluation)
6. Additional multilingual/domain tasks as needed

Why:
1. Intrinsic metrics do not fully capture reasoning and factual behavior.
2. Standardized benchmarks enable external comparability.

What it evaluates:
1. Zero-/few-shot task performance
2. Robustness of capabilities across task families
3. Language-transfer behavior for EN/DE settings

## Phase 5: Pythia-Style Training Dynamics
Definition: Evaluate many checkpoints across training to analyze learning trajectories, not only the final checkpoint.

Checkpoint strategy:
1. Dense early checkpoints (e.g., 500, 1000, 1500, 2000, ...)
2. Sparser later checkpoints (e.g., every 2k-5k steps)
3. Keep at least one resume-safe latest checkpoint

Why:
1. Different tasks can peak at different training steps.
2. Final checkpoint is not always best checkpoint.

What it evaluates:
1. Metric evolution by step
2. Inflection points and diminishing returns
3. Trade-offs between calibration and task performance

## Phase 6: Reporting and Comparison
Definition: Produce consistent artifacts for checkpoint-level and run-level decisions.

What we implement:
1. Checkpoint-vs-metric tables
2. Best checkpoint per metric/task
3. Curves for PPL, CE, ECE, benchmark scores
4. Run cards summarizing data, hyperparameters, and outcomes

Why:
1. Enables reproducible model selection.
2. Supports clear communication to team and reviewers.

What it evaluates:
1. Relative quality of checkpoints
2. Reproducibility and traceability across runs
3. Decision-readiness for deployment/further training

## Phase Documents
1. [Phase 0 Environment Setup](phases/PHASE_0_ENVIRONMENT_SETUP.md)
2. [Phase 1 Data Preparation](phases/PHASE_1_DATA_PREPARATION.md)
3. [Phase 2 Pre-Training and Checkpointing](phases/PHASE_2_PRETRAINING_AND_CHECKPOINTING.md)
4. [Phase 3 Intrinsic Evaluation](phases/PHASE_3_INTRINSIC_EVALUATION.md)
5. [Phase 4 Benchmark Evaluation](phases/PHASE_4_BENCHMARK_EVALUATION.md)
6. [Phase 5 Checkpoint Dynamics](phases/PHASE_5_CHECKPOINT_DYNAMICS.md)
7. [Phase 6 Reporting and Comparison](phases/PHASE_6_REPORTING_AND_COMPARISON.md)

## Current Status
1. Planning updated for Phase 0-6 scope.
2. Phase 3 implementation added in training repository as:
   [Phase 3 evaluator script](../nano-alpha-llm-pretrain/scripts/phase3_intrinsic_eval.py)
3. Next implementation targets:
   1. Phase 4 lm-eval integration
   2. Phase 5 checkpoint sweep orchestration
   3. Phase 6 consolidated reporting outputs
