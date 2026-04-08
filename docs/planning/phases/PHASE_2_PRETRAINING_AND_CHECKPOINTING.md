# Phase 2: Pre-Training and Checkpointing

## Definition
Train the Nano-Alpha causal language model from scratch and produce a checkpoint series for resume and evaluation.

## What We Implement
1. Tokenizer training and fixed tokenizer artifacts
2. HF Trainer-based pretraining loop
3. Step-based checkpointing schedule
4. Experiment tracking (MLflow/W&B with fallback)
5. Resume flow from last checkpoint with total-step continuation

## Why It Matters
1. Checkpoints are needed for fault tolerance under walltime constraints.
2. Intermediate checkpoints enable Phase 3/4/5 analyses.
3. Standardized training outputs improve cross-run comparability.

## What It Evaluates
1. Optimization behavior (loss progression)
2. Training throughput and stability
3. Checkpoint quality and usability for downstream evaluation

## Typical Outputs
1. checkpoint-* directories
2. final model directory
3. trainer_state and run metrics
4. tracking artifacts and logs
