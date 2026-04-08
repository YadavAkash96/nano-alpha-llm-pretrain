# Phase 4: Benchmark Evaluation

## Definition
Benchmark evaluation measures task-level behavior on standardized suites using consistent prompting/scoring.

## What We Implement
1. lm-eval-harness runner integration
2. Task subsets: MMLU, ARC, HellaSwag, TruthfulQA, IGEL
3. Checkpoint-by-checkpoint benchmark execution
4. Artifact logging for prompt format, shot count, and scores

## Why It Matters
1. Intrinsic scores cannot fully predict task behavior.
2. Standard tasks provide external comparability.
3. Enables robust model selection for target use cases.

## What It Evaluates
1. Reasoning and factual consistency
2. Task generalization across domains
3. German-focused and multilingual behavior
4. Metric trajectories across checkpoints
