#!/usr/bin/env python3
"""Phase 3 intrinsic evaluation across checkpoints.

Computes per-checkpoint intrinsic metrics from logits:
- Cross-entropy / NLL and perplexity
- Token-level accuracy
- Calibration metrics (ECE, Brier)
- Reliability diagram
- Selective prediction (risk-coverage) curve
- Optional domain/language breakdown using a metadata key (default: lang)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 intrinsic evaluation")
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("checkpoints/nano-alpha-130m-v100"),
        help="Directory containing checkpoint-* subdirectories and optional final/",
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=Path("data/processed/phase1_v2_de_eval/val_corpus.jsonl"),
        help="Validation JSONL with at least a text field",
    )
    parser.add_argument(
        "--tokenizer-file",
        type=Path,
        default=Path("artifacts/tokenizer/nano_alpha_bpe.tokenizer.json"),
        help="Tokenizer JSON fallback if checkpoint tokenizer is missing",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/eval/phase3_intrinsic"),
        help="Output directory for metrics and plots",
    )
    parser.add_argument("--text-key", type=str, default="text", help="JSON key containing text")
    parser.add_argument("--domain-key", type=str, default="lang", help="JSON key for domain/language breakdown")
    parser.add_argument("--seq-length", type=int, default=1024, help="Chunk length for evaluation")
    parser.add_argument("--max-samples", type=int, default=0, help="Limit number of JSONL rows (0 = all)")
    parser.add_argument(
        "--max-eval-tokens",
        type=int,
        default=200000,
        help="Cap evaluated next-token positions per checkpoint (0 = no cap)",
    )
    parser.add_argument("--num-bins", type=int, default=15, help="Bin count for ECE/reliability")
    parser.add_argument(
        "--selective-points",
        type=int,
        default=20,
        help="Number of coverage points for selective prediction curve",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Evaluation device",
    )
    parser.add_argument(
        "--min-checkpoint-step",
        type=int,
        default=0,
        help="Only evaluate checkpoint-N where N >= this value (default: 0)",
    )
    parser.add_argument(
        "--max-checkpoint-step",
        type=int,
        default=0,
        help="Only evaluate checkpoint-N where N <= this value (0 means no upper bound)",
    )
    parser.add_argument(
        "--presentation-stride",
        type=int,
        default=2000,
        help="Step gap used to create a trimmed presentation summary (default: 2000)",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def discover_checkpoints(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Checkpoints directory not found: {root}")

    ckpts = [p for p in root.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]

    def step_of(path: Path) -> int:
        try:
            return int(path.name.split("-")[-1])
        except Exception:
            return -1

    ckpts.sort(key=step_of)

    final_dir = root / "final"
    if final_dir.exists() and final_dir.is_dir():
        ckpts.append(final_dir)

    if not ckpts:
        raise RuntimeError(f"No checkpoints found under {root}")
    return ckpts


def iter_jsonl_rows(path: Path, max_rows: int) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if max_rows > 0 and idx >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def fallback_tokenizer(tokenizer_file: Path) -> PreTrainedTokenizerFast:
    tok = PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))
    tok.add_special_tokens(
        {
            "pad_token": "<pad>",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
        }
    )
    tok.model_max_length = int(1e12)
    return tok


def load_tokenizer(ckpt_dir: Path, tokenizer_file: Path):
    try:
        return AutoTokenizer.from_pretrained(str(ckpt_dir), use_fast=True, local_files_only=True)
    except Exception:
        return fallback_tokenizer(tokenizer_file)


def safe_exp(value: float) -> float:
    # Avoid overflow in pathological cases.
    return math.exp(min(value, 50.0))


def reliability_bins(
    conf: torch.Tensor,
    corr: torch.Tensor,
    num_bins: int,
    bin_count: np.ndarray,
    bin_conf_sum: np.ndarray,
    bin_corr_sum: np.ndarray,
) -> None:
    idx = torch.clamp((conf * num_bins).long(), max=num_bins - 1)
    for b in range(num_bins):
        mask = idx == b
        if not torch.any(mask):
            continue
        c = int(mask.sum().item())
        bin_count[b] += c
        bin_conf_sum[b] += float(conf[mask].sum().item())
        bin_corr_sum[b] += float(corr[mask].sum().item())


def selective_curve(conf_all: np.ndarray, corr_all: np.ndarray, points: int) -> List[Dict[str, float]]:
    if conf_all.size == 0:
        return []
    order = np.argsort(-conf_all)
    corr_sorted = corr_all[order]
    n = corr_sorted.size

    out: List[Dict[str, float]] = []
    for cov in np.linspace(0.05, 1.0, points):
        k = max(1, int(round(cov * n)))
        acc = float(corr_sorted[:k].mean())
        out.append(
            {
                "coverage": float(k / n),
                "accuracy": acc,
                "risk": float(1.0 - acc),
            }
        )
    return out


def plot_reliability(bin_count: np.ndarray, bin_conf_sum: np.ndarray, bin_corr_sum: np.ndarray, out_png: Path) -> None:
    nonzero = bin_count > 0
    if not np.any(nonzero):
        return

    avg_conf = np.zeros_like(bin_conf_sum, dtype=float)
    avg_acc = np.zeros_like(bin_corr_sum, dtype=float)
    avg_conf[nonzero] = bin_conf_sum[nonzero] / bin_count[nonzero]
    avg_acc[nonzero] = bin_corr_sum[nonzero] / bin_count[nonzero]

    centers = (np.arange(bin_count.size) + 0.5) / bin_count.size

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="perfect calibration")
    ax.plot(centers[nonzero], avg_acc[nonzero], marker="o", linewidth=2, label="model")
    ax.set_title("Reliability Diagram")
    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_selective(curve: List[Dict[str, float]], out_png: Path) -> None:
    if not curve:
        return
    x = [p["coverage"] for p in curve]
    y = [p["risk"] for p in curve]

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(x, y, marker="o", linewidth=2)
    ax.set_title("Selective Prediction Curve")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk (1 - accuracy)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def checkpoint_step(name: str) -> int:
    if name == "final":
        return 10**9
    if name.startswith("checkpoint-"):
        try:
            return int(name.split("-")[-1])
        except Exception:
            return 10**9 - 1
    return 10**9 - 2


def checkpoint_step_or_none(name: str) -> int | None:
    if name.startswith("checkpoint-"):
        try:
            return int(name.split("-")[-1])
        except Exception:
            return None
    return None


def filter_checkpoints_by_step(ckpts: List[Path], min_step: int, max_step: int) -> List[Path]:
    filtered: List[Path] = []
    for ckpt in ckpts:
        step = checkpoint_step_or_none(ckpt.name)
        if step is None:
            # Keep non-step aliases like final by default.
            filtered.append(ckpt)
            continue
        if step < min_step:
            continue
        if max_step > 0 and step > max_step:
            continue
        filtered.append(ckpt)
    return filtered


def build_presentation_rows(rows: List[Dict[str, object]], stride: int) -> List[Dict[str, object]]:
    if not rows:
        return []

    stride = max(1, stride)
    sorted_rows = sorted(rows, key=lambda r: checkpoint_step(str(r["checkpoint"])))

    out: List[Dict[str, object]] = []
    first = sorted_rows[0]
    out.append(first)

    for row in sorted_rows[1:-1]:
        name = str(row["checkpoint"])
        step = checkpoint_step_or_none(name)
        if step is None:
            continue
        if step % stride == 0:
            out.append(row)

    if len(sorted_rows) > 1:
        last = sorted_rows[-1]
        if out[-1]["checkpoint"] != last["checkpoint"]:
            out.append(last)

    return out


def write_summary_md(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "| Checkpoint | Perplexity | Token Accuracy | ECE | Brier |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {checkpoint} | {perplexity:.4f} | {token_accuracy:.4f} | {ece:.4f} | {brier:.4f} |".format(
                checkpoint=row["checkpoint"],
                perplexity=float(row["perplexity"]),
                token_accuracy=float(row["token_accuracy"]),
                ece=float(row["ece"]),
                brier=float(row["brier"]),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def normalize_series(values: List[float], higher_is_better: bool) -> List[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if abs(high - low) < 1e-12:
        return [0.5 for _ in values]
    scaled = [(value - low) / (high - low) for value in values]
    if higher_is_better:
        return scaled
    return [1.0 - value for value in scaled]


def plot_checkpoint_comparison(summary_rows: List[Dict[str, object]], out_png: Path) -> None:
    if not summary_rows:
        return

    rows = sorted(summary_rows, key=lambda r: checkpoint_step(str(r["checkpoint"])))
    labels = [str(r["checkpoint"]) for r in rows]
    x = list(range(len(rows)))

    ppl = normalize_series([float(r["perplexity"]) for r in rows], higher_is_better=False)
    acc = normalize_series([float(r["token_accuracy"]) for r in rows], higher_is_better=True)
    ece = normalize_series([float(r["ece"]) for r in rows], higher_is_better=False)
    brier = normalize_series([float(r["brier"]) for r in rows], higher_is_better=False)

    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.plot(x, ppl, marker="o", linewidth=2.2, color="#2563eb", label="Perplexity trend")
    ax.plot(x, acc, marker="o", linewidth=2.2, color="#16a34a", label="Token accuracy trend")
    ax.plot(x, ece, marker="o", linewidth=2.2, color="#f59e0b", label="ECE trend")
    ax.plot(x, brier, marker="o", linewidth=2.2, color="#ef4444", label="Brier trend")

    ax.set_title("Checkpoint comparison during learning", fontsize=15)
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Normalized score, higher is better")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.22)
    ax.legend(loc="best", frameon=True)

    ax.text(
        0.01,
        0.02,
        "Metrics are normalized per series so they can share one clean comparison plot.",
        transform=ax.transAxes,
        fontsize=9,
        color="#475569",
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def evaluate_checkpoint(
    ckpt_dir: Path,
    args: argparse.Namespace,
    rows: List[Dict[str, object]],
    device: torch.device,
) -> Dict[str, object]:
    tokenizer = load_tokenizer(ckpt_dir, args.tokenizer_file)

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt_dir),
        local_files_only=True,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    total_nll = 0.0
    total_tokens = 0
    total_correct = 0
    total_conf_sum = 0.0
    brier_sum = 0.0

    domain_stats = defaultdict(lambda: {"nll": 0.0, "tokens": 0, "correct": 0})

    bin_count = np.zeros(args.num_bins, dtype=np.int64)
    bin_conf_sum = np.zeros(args.num_bins, dtype=np.float64)
    bin_corr_sum = np.zeros(args.num_bins, dtype=np.float64)

    selective_conf: List[float] = []
    selective_corr: List[int] = []

    evaluated_rows = 0

    with torch.no_grad():
        for row in rows:
            text = str(row.get(args.text_key, "") or "").strip()
            if not text:
                continue

            domain = str(row.get(args.domain_key, "unknown") or "unknown")
            encoded = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
            if len(encoded) < 2:
                continue

            for start in range(0, len(encoded) - 1, args.seq_length):
                token_chunk = encoded[start : start + args.seq_length]
                if len(token_chunk) < 2:
                    continue

                input_ids = torch.tensor([token_chunk], dtype=torch.long, device=device)
                logits = model(input_ids).logits[:, :-1, :]
                labels = input_ids[:, 1:]

                log_probs = F.log_softmax(logits.float(), dim=-1)
                true_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

                n_tokens = labels.numel()
                nll = float((-true_log_probs).sum().item())

                top_logp, top_idx = torch.max(log_probs, dim=-1)
                conf = torch.exp(top_logp)
                correct = top_idx.eq(labels)
                corr_float = correct.float()

                total_nll += nll
                total_tokens += n_tokens
                total_correct += int(correct.sum().item())
                total_conf_sum += float(conf.sum().item())
                brier_sum += float(((conf - corr_float) ** 2).sum().item())

                domain_stats[domain]["nll"] += nll
                domain_stats[domain]["tokens"] += n_tokens
                domain_stats[domain]["correct"] += int(correct.sum().item())

                reliability_bins(
                    conf=conf.view(-1).cpu(),
                    corr=corr_float.view(-1).cpu(),
                    num_bins=args.num_bins,
                    bin_count=bin_count,
                    bin_conf_sum=bin_conf_sum,
                    bin_corr_sum=bin_corr_sum,
                )

                selective_conf.extend(conf.view(-1).detach().cpu().tolist())
                selective_corr.extend(correct.view(-1).detach().cpu().int().tolist())

                if args.max_eval_tokens > 0 and total_tokens >= args.max_eval_tokens:
                    break

            evaluated_rows += 1
            if args.max_eval_tokens > 0 and total_tokens >= args.max_eval_tokens:
                break

    if total_tokens == 0:
        raise RuntimeError(f"No evaluable tokens for checkpoint: {ckpt_dir}")

    avg_nll = total_nll / total_tokens
    ppl = safe_exp(avg_nll)
    acc = total_correct / total_tokens
    avg_conf = total_conf_sum / total_tokens
    brier = brier_sum / total_tokens

    ece = 0.0
    for i in range(args.num_bins):
        if bin_count[i] == 0:
            continue
        bin_acc = bin_corr_sum[i] / bin_count[i]
        bin_conf = bin_conf_sum[i] / bin_count[i]
        ece += (bin_count[i] / total_tokens) * abs(bin_acc - bin_conf)

    domain_metrics = {}
    for domain, st in sorted(domain_stats.items()):
        if st["tokens"] == 0:
            continue
        dnll = st["nll"] / st["tokens"]
        domain_metrics[domain] = {
            "tokens": st["tokens"],
            "avg_nll": dnll,
            "perplexity": safe_exp(dnll),
            "token_accuracy": st["correct"] / st["tokens"],
        }

    conf_arr = np.array(selective_conf, dtype=np.float64)
    corr_arr = np.array(selective_corr, dtype=np.float64)
    sel_curve = selective_curve(conf_arr, corr_arr, points=args.selective_points)

    return {
        "checkpoint": ckpt_dir.name,
        "tokens": total_tokens,
        "evaluated_rows": evaluated_rows,
        "avg_nll": avg_nll,
        "perplexity": ppl,
        "token_accuracy": acc,
        "avg_confidence": avg_conf,
        "ece": ece,
        "brier": brier,
        "domain_breakdown": domain_metrics,
        "reliability_bins": {
            "count": bin_count.tolist(),
            "confidence_sum": bin_conf_sum.tolist(),
            "correct_sum": bin_corr_sum.tolist(),
        },
        "selective_curve": sel_curve,
    }


def write_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "checkpoint",
                "tokens",
                "evaluated_rows",
                "avg_nll",
                "perplexity",
                "token_accuracy",
                "avg_confidence",
                "ece",
                "brier",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in writer.fieldnames})


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    if not args.eval_file.exists():
        raise FileNotFoundError(f"Eval file not found: {args.eval_file}")
    if not args.tokenizer_file.exists():
        raise FileNotFoundError(f"Tokenizer file not found: {args.tokenizer_file}")

    rows = list(iter_jsonl_rows(args.eval_file, args.max_samples))
    if not rows:
        raise RuntimeError("No rows loaded from eval JSONL")

    ckpts = discover_checkpoints(args.checkpoints_dir)
    ckpts = filter_checkpoints_by_step(ckpts, args.min_checkpoint_step, args.max_checkpoint_step)
    if not ckpts:
        raise RuntimeError(
            "No checkpoints match filters: "
            f"min_step={args.min_checkpoint_step}, max_step={args.max_checkpoint_step}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, object]] = []

    print(f"Device: {device}")
    print(f"Checkpoints discovered: {len(ckpts)}")
    for ckpt in ckpts:
        print(f"Evaluating {ckpt.name} ...")
        metrics = evaluate_checkpoint(ckpt, args, rows, device)

        per_ckpt_json = args.output_dir / f"{ckpt.name}_intrinsic_metrics.json"
        per_ckpt_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        plot_reliability(
            np.array(metrics["reliability_bins"]["count"], dtype=np.int64),
            np.array(metrics["reliability_bins"]["confidence_sum"], dtype=np.float64),
            np.array(metrics["reliability_bins"]["correct_sum"], dtype=np.float64),
            args.output_dir / f"{ckpt.name}_reliability.png",
        )
        plot_selective(metrics["selective_curve"], args.output_dir / f"{ckpt.name}_selective_curve.png")

        summary_rows.append(
            {
                "checkpoint": metrics["checkpoint"],
                "tokens": metrics["tokens"],
                "evaluated_rows": metrics["evaluated_rows"],
                "avg_nll": metrics["avg_nll"],
                "perplexity": metrics["perplexity"],
                "token_accuracy": metrics["token_accuracy"],
                "avg_confidence": metrics["avg_confidence"],
                "ece": metrics["ece"],
                "brier": metrics["brier"],
            }
        )

        print(
            f"  tokens={metrics['tokens']} ppl={metrics['perplexity']:.4f} "
            f"acc={metrics['token_accuracy']:.4f} ece={metrics['ece']:.4f}"
        )

    summary_json = args.output_dir / "phase3_intrinsic_summary.json"
    summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    summary_csv = args.output_dir / "phase3_intrinsic_summary.csv"
    write_summary_csv(summary_csv, summary_rows)

    summary_md = args.output_dir / "phase3_intrinsic_summary.md"
    write_summary_md(summary_md, summary_rows)

    comparison_png = args.output_dir / "phase3_checkpoint_comparison.png"
    plot_checkpoint_comparison(summary_rows, comparison_png)

    presentation_rows = build_presentation_rows(summary_rows, stride=args.presentation_stride)
    presentation_json = args.output_dir / "phase3_intrinsic_summary_presentation.json"
    presentation_json.write_text(json.dumps(presentation_rows, indent=2), encoding="utf-8")

    presentation_csv = args.output_dir / "phase3_intrinsic_summary_presentation.csv"
    write_summary_csv(presentation_csv, presentation_rows)

    presentation_md = args.output_dir / "phase3_intrinsic_summary_presentation.md"
    write_summary_md(presentation_md, presentation_rows)

    presentation_png = args.output_dir / "phase3_checkpoint_comparison_presentation.png"
    plot_checkpoint_comparison(presentation_rows, presentation_png)

    print(f"Wrote summary JSON: {summary_json}")
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote summary Markdown table: {summary_md}")
    print(f"Wrote checkpoint comparison plot: {comparison_png}")
    print(f"Wrote presentation JSON: {presentation_json}")
    print(f"Wrote presentation CSV: {presentation_csv}")
    print(f"Wrote presentation Markdown table: {presentation_md}")
    print(f"Wrote presentation checkpoint comparison plot: {presentation_png}")


if __name__ == "__main__":
    main()
