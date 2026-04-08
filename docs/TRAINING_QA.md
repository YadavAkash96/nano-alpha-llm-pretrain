# Training Q&A (Living Document)

This document captures high-value questions and answers about model scaling, token budgets, and parameter tuning for this project.

Last updated: 2026-03-31

## Q1) What does `max_steps=5000` mean?
It means 5000 optimizer updates, not 5000 samples and not 5000 epochs.

With current config:
- per-device batch size = 2
- GPUs = 1
- gradient accumulation = 16
- sequence length = 1024

Effective sequences per optimizer step:
- `2 * 1 * 16 = 32`

Tokens per optimizer step:
- `32 * 1024 = 32768`

Total tokens for 5000 steps:
- `5000 * 32768 = 163,840,000` (~0.164B tokens)

## Q2) How are training tokens computed exactly?
Use:

`total_tokens = steps * per_device_batch_size * num_gpus * grad_accum_steps * seq_length`

Important distinction:
- Processed tokens: all tokens used in optimization (includes repeats across epochs).
- Unique dataset tokens: one-pass token count of your corpus after chunking.

## Q3) How many epochs is our current run equivalent to?
Training is step-capped, so epoch count is derived.

From live run (job 1561850):
- inferred step ~= 1200
- logged epoch ~= 0.1187
- inferred steps/epoch ~= 10,110

So:
- 5000 steps ~= `5000 / 10110 = 0.49` epoch (about half an epoch)

## Q4) Why can model quality still be limited even if loss drops fast?
Early loss drop is expected and good, but strong general capability usually needs a larger total token budget.

At current setup:
- 0.164B tokens (5000 steps) is a short pretraining phase for a 130M model.
- Expect improvements in local fluency/style.
- Expect weaker robustness and long-range reliability versus longer training.

## Q5) For this V100 setup, what is the best quality-first strategy while staying at 130M?
Recommendation:
- Keep 130M model.
- Train longer via checkpoint continuation.
- Increase token budget before increasing model size.
- Keep fp16 on V100.

Suggested milestones:
- 10k steps -> 0.328B tokens
- 30k steps -> 0.983B tokens
- 60k steps -> 1.966B tokens
- 80k steps -> 2.621B tokens

## Q6) What is the estimated walltime for these milestones on current throughput?
Measured from live run around step 1200:
- average ~1.4217 sec/step

Projected total training time:
- 5k steps -> ~1.97h
- 10k steps -> ~3.95h
- 30k steps -> ~11.85h
- 60k steps -> ~23.69h
- 80k steps -> ~31.59h

Notes:
- Real runtime can vary with queue delays, periodic eval/save overhead, and node variability.
- Since current job time limit is 24h, 60k is near boundary and 80k requires multi-job continuation.

## Q7) Should we prioritize more data or more steps?
Both help, but order matters:
1. First increase steps (processed token budget) to reach at least ~1B tokens.
2. In parallel, improve/add high-quality data to increase unique tokens and reduce overfitting on repeats.
3. Only then evaluate whether larger model size is worth the extra compute.

## Q8) What are practical next submissions for better learning?
A staged plan:
- Stage A: finish current 5k baseline.
- Stage B: continue from latest checkpoint to 30k steps total.
- Stage C: continue to 60k if loss/eval still improving.
- Stage D: optional 80k if gains continue and compute budget allows.

Use checkpoints every 500 steps for safe resume under queue interruptions.

## Q9) How do I resume training from a checkpoint across multiple SLURM jobs?
Use the [resume sbatch template](../slurm/phase2_train_v100_resume.sbatch):

```bash
# After current job finishes, resume from its LAST checkpoint
sbatch slurm/phase2_train_v100_resume.sbatch \
  checkpoints/nano-alpha-130m-v100/checkpoint-5000 \
  10000
```

Parameters:
- **Arg 1**: Last checkpoint directory (required)
  - Always use the FINAL checkpoint from the previous run
  - Example: if previous job had max_steps=5000, use `checkpoint-5000`
  - This contains full optimizer/scheduler/RNG state
- **Arg 2**: New total max_steps (required)
  - This is your final target, **NOT incremental**
  - If resuming from step 5000 and want 10k total steps: pass `10000` (will add 5k more)
  - If wanting 30k total steps: pass `30000` (will add 25k more)
- **Arg 3** (optional): W&B run ID
  - If provided, logs append to same run (unified charts)
  - If omitted, creates a new run (separate charts)

Resume capabilities preserved:
- Optimizer and scheduler state restored from checkpoint
- RNG (random number generator) state restored for reproducibility
- Gradient accumulation counters reset properly
- Learning rate schedule continues appropriately from previous position

See [TRAINING_RESUME.md](./TRAINING_RESUME.md) for multi-stage planning, examples, and W&B integration.

## Q10) Why don't I see training logs on W&B immediately, and how do I monitor progress?
Training logs are usually offline or delayed due to HPC network latency.

Root cause:
- Compute nodes on HPC cluster often cannot reliably reach external APIs (api.wandb.ai).
- W&B client experiences timeout (~8 seconds) and automatically falls back to `--report-to none`.
- Data syncs only after job completes, via `wandb sync` command.

What to expect:
- **During training**: Logs written to local checkpoint directory (e.g., `checkpoints/nano-alpha-130m-v100/logs/`)
- **After job finishes**: Manually sync with:
  ```bash
  wandb sync checkpoints/nano-alpha-130m-v100/
  ```
- **On W&B dashboard** (after sync): Charts appear at https://wandb.ai/ay1820098/nano-alpha-llm
  - Look for project=`nano-alpha-llm`, group=`phase2_v100`

Checking training health without W&B:
- Monitor loss via `tail -f` in job stdout/stderr log file
- Check GPU usage: `nvidia-smi` (should show ~32GB used on V100)
- Verify checkpoint saves: `ls -lah checkpoints/nano-alpha-130m-v100/` every few minutes
- Check trainer state for loss: `cat checkpoints/nano-alpha-130m-v100/trainer_state.json | grep -A5 training_loss`

Optional: Force online mode (not recommended on unreliable networks):
- Remove the W&B fallback check in sbatch to keep --report-to wandb
- Risk: job hangs waiting for W&B timeout if network is down
- Safer: stick with fallback and manual sync after completion

## Q11) Does setting a higher max_steps require more resources per step, or only more total runtime?
`max_steps` controls only how long training runs (number of optimizer updates). It does not increase per-step memory demand.

Per-step feasibility depends on your batch/sequence configuration and model size:
- per-device batch size = 2
- gradient accumulation = 16
- sequence length = 1024
- tokens per optimizer step = `2 * 16 * 1024 = 32768` (single GPU)

If one step fits and runs stably on the V100, increasing `max_steps` mostly increases walltime and total consumed GPU-hours.

Two independent stop conditions always apply:
- Training reaches `max_steps`
- SLURM job reaches walltime limit (24h) or is terminated

So for `max_steps=60000`:
- It will continue toward 60k only while resources remain allocated.
- If walltime ends first, training stops early.
- Resume from the most recent saved checkpoint and continue in a new job.

Checkpoint implication with `save_steps=500`:
- At interruption, you resume from the latest `checkpoint-<multiple_of_500>`.
- You can lose up to 499 unsaved steps since the previous checkpoint.

## Q12) Do we need intermediate checkpoints for LLM evaluation (lm-harness, MMLU, HellaSwag), not just loss curves?
Yes. For LLM quality analysis, intermediate checkpoints are important because:
- Loss alone does not capture downstream task performance.
- Benchmarks can improve non-linearly across training, and best quality may occur before final step.
- Different tasks (e.g., MMLU vs HellaSwag) can peak at different checkpoints.

Practical retention strategy under storage limits:
- Keep dense checkpoints early: every 500 steps up to 5k/10k.
- Keep sparse checkpoints later: every 2k to 5k steps (e.g., 10k, 15k, 20k, 30k, 40k, 50k, 60k).
- Always keep the latest checkpoint and final model.

For evaluation-only archives, keep model/tokenizer files; optimizer states are only needed for resume:
- Keep: `model.safetensors`, `config.json`, `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`
- Optional for reporting: `trainer_state.json`, `training_args.bin`
- Drop for eval-only storage savings: `optimizer.pt`, `scheduler.pt`, `scaler.pt`, `rng_state.pth`

Suggested benchmark cadence:
- Run lightweight benchmark subset at each retained checkpoint.
- Run full benchmark suite on top-3 checkpoints selected by lightweight scores.
- Log benchmark outputs into MLflow as artifacts for comparison tables.

---

## Add New Q&A Entries
When adding new entries, prefer questions about:
- token accounting and throughput
- tuning knobs and trade-offs
- expected quality at different budgets
- stability and reproducibility on HPC
- checkpoint resume and W&B integration
