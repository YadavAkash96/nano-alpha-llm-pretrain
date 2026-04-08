# V100 Training Resume Guide

This guide explains how to continue training from a checkpoint and monitor progress on W&B.

## Quick Start: Resume Training

After the current training finishes (or at any checkpoint), resume from the **LAST checkpoint**:

```bash
# Example: if current job max_steps=5000, use checkpoint-5000
sbatch slurm/phase2_train_v100_resume.sbatch checkpoints/nano-alpha-130m-v100/checkpoint-5000 10000
```

This resumes from step 5000 and trains to step 10000 total (adding 5000 more steps).

**Key point:** The second argument is your **new total max_steps**, not additional steps.

## What Each Parameter Means

1. **checkpoint_path**: Directory of your checkpoint (required)
   - Use the LAST checkpoint from the previous run
   - Example from current job: `checkpoints/nano-alpha-130m-v100/checkpoint-5000`
   - This contains model, optimizer, scheduler, and RNG state

2. **new_max_steps**: TOTAL training steps (not additional) (required)
   - This is the final target, not what you're adding
   - If previous job did max_steps=5000, set this to your new goal (e.g., 10000, 30000, 60000)
   - The trainer will resume from step 5000 and run until reaching this number

3. **wandb_run_id** (optional): Continue in same W&B run
   - If provided, logs append to existing run (unified charts)
   - If omitted, creates a new separate run
   - Find run ID at https://wandb.ai/ay1820098/nano-alpha-llm or in job logs

## Finding Your W&B Run

Your W&B project:
- **Project**: `nano-alpha-llm`
- **Run group**: `phase2_v100`

On W&B dashboard:
1. Go to https://wandb.ai/ay1820098/nano-alpha-llm
2. Look for runs in the "phase2_v100" group
3. Run name defaults to "nano-alpha-130m-phase2"

## Why I Don't See Training Data in W&B

Possible reasons and fixes:

1. **Network timeout on compute node** (most common):
   - The script detects this automatically
   - It falls back to `--report-to none`
   - Data won't sync to W&B during training
   - You can manually sync logs later:
   ```bash
   wandb sync checkpoints/nano-alpha-130m-v100/
   ```

2. **W&B run is in offline mode**:
   - Force online mode:
   ```bash
   wandb online
   ```

3. **New run created instead of resumed**:
   - Next submission, pass the `--resume-id` parameter

## How to Use Existing Run

To continue in the SAME W&B run (same charts/history):

1. Find your run ID from W&B dashboard
2. Submit resume with the ID:
```bash
sbatch slurm/phase2_train_v100_resume.sbatch checkpoints/nano-alpha-130m-v100/checkpoint-1000 10000 <run_id>
```

Logs will append to the same run instead of starting a new one.

## Checkpoint Format

Each checkpoint includes:
- `model.safetensors`: Model weights
- `optimizer.pt`: Optimizer state (for resumed training)
- `scheduler.pt`: Learning rate scheduler state
- `trainer_state.json`: Training metadata
- `rng_state.pth`: Random number generator state

All are needed for correct resume.

## Multi-Stage Training Example

Current state:
- Running job: ~1200 steps done
- Target: 80k steps total (for ~2.6B tokens)

Plan:
```bash
# Stage 1 (current): ~1200 steps, likely to complete
# Stage 2: Resume to 30k steps
sbatch slurm/phase2_train_v100_resume.sbatch checkpoints/nano-alpha-130m-v100/checkpoint-1000 30000

# Stage 3: Resume to 60k steps (if loss still improving)
sbatch slurm/phase2_train_v100_resume.sbatch checkpoints/nano-alpha-130m-v100/checkpoint-<latest> 60000

# Stage 4: Resume to 80k steps (final, if needed)
sbatch slurm/phase2_train_v100_resume.sbatch checkpoints/nano-alpha-130m-v100/checkpoint-<latest> 80000
```

Each stage adds more training and computes more tokens (see TRAINING_QA.md for token budgets).

## Troubleshooting Resume

**Error: "Checkpoint directory not found"**
- Verify checkpoint path exists: `ls checkpoints/nano-alpha-130m-v100/checkpoint-XXX`
- Use full path or relative path from repo root

**Training doesn't resume (starts at step 0)**
- Trainer may auto-detect if using same output-dir
- Resume script validates checkpoint and passes resume flag to trainer

**Loss jumps or training looks wrong**
- Check that new_max_steps > current steps
- Verify you're using same model config, tokenizer, and data path

