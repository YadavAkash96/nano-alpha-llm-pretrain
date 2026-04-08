# Phase 5: Checkpoint Dynamics (Pythia-Style)

## Definition
Checkpoint dynamics analysis tracks how metrics evolve during training rather than only scoring the final model.

## What We Implement
1. Scheduled checkpoint set (dense early, sparse later)
2. Automated sweep of intrinsic + benchmark metrics
3. Joint trend analysis across metrics and tasks

## Why It Matters
1. Best checkpoints differ by task/metric.
2. Final checkpoint may overfit some objectives.
3. Dynamics reveal scaling behavior and training efficiency.

## What It Evaluates
1. Inflection points and diminishing returns
2. Trade-offs between calibration and benchmark quality
3. Stability of progress across further runs
