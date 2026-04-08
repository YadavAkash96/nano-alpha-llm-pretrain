#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def load_log_history(trainer_state_path: Path):
    with trainer_state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    return state.get("log_history", []), state


def split_metrics(log_history):
    train_steps = []
    train_loss = []
    learning_rate = []
    grad_norm = []

    eval_steps = []
    eval_loss = []
    eval_runtime = []
    eval_sps = []

    for row in log_history:
        step = row.get("step")
        if step is None:
            continue

        if "loss" in row:
            train_steps.append(step)
            train_loss.append(float(row["loss"]))
            learning_rate.append(float(row.get("learning_rate", "nan")))
            grad_norm.append(float(row.get("grad_norm", "nan")))

        if "eval_loss" in row:
            eval_steps.append(step)
            eval_loss.append(float(row["eval_loss"]))
            eval_runtime.append(float(row.get("eval_runtime", "nan")))
            eval_sps.append(float(row.get("eval_samples_per_second", "nan")))

    return {
        "train_steps": train_steps,
        "train_loss": train_loss,
        "learning_rate": learning_rate,
        "grad_norm": grad_norm,
        "eval_steps": eval_steps,
        "eval_loss": eval_loss,
        "eval_runtime": eval_runtime,
        "eval_samples_per_second": eval_sps,
    }


def write_csv(out_csv: Path, metrics):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("kind,step,value\n")
        for s, v in zip(metrics["train_steps"], metrics["train_loss"]):
            f.write(f"train_loss,{s},{v}\n")
        for s, v in zip(metrics["train_steps"], metrics["learning_rate"]):
            f.write(f"learning_rate,{s},{v}\n")
        for s, v in zip(metrics["train_steps"], metrics["grad_norm"]):
            f.write(f"grad_norm,{s},{v}\n")
        for s, v in zip(metrics["eval_steps"], metrics["eval_loss"]):
            f.write(f"eval_loss,{s},{v}\n")
        for s, v in zip(metrics["eval_steps"], metrics["eval_runtime"]):
            f.write(f"eval_runtime,{s},{v}\n")
        for s, v in zip(metrics["eval_steps"], metrics["eval_samples_per_second"]):
            f.write(f"eval_samples_per_second,{s},{v}\n")


def plot_png(out_png: Path, metrics, eval_interval: int):
    import matplotlib.pyplot as plt

    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(metrics["train_steps"], metrics["train_loss"], label="train_loss", linewidth=1.5)
    ax.plot(metrics["eval_steps"], metrics["eval_loss"], label="eval_loss", marker="o", linestyle="-", linewidth=1.8)

    # Visual stage markers: evaluation happens at regular boundaries.
    if metrics["train_steps"] and eval_interval > 0:
        max_step = max(metrics["train_steps"])
        x = eval_interval
        while x <= max_step:
            ax.axvline(x, color="gray", alpha=0.12, linewidth=0.9)
            x += eval_interval

    ax.set_title("Training and Validation Loss vs Step")
    ax.set_xlabel("Global step")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def build_summary(metrics, state):
    summary = {
        "global_step": int(state.get("global_step") or 0),
        "eval_steps": int(state.get("eval_steps") or 0),
        "train_points": len(metrics["train_steps"]),
        "eval_points": len(metrics["eval_steps"]),
        "last_train_loss": metrics["train_loss"][-1] if metrics["train_loss"] else None,
        "last_eval_loss": metrics["eval_loss"][-1] if metrics["eval_loss"] else None,
        "best_eval_loss": min(metrics["eval_loss"]) if metrics["eval_loss"] else None,
        "best_eval_step": None,
    }
    if metrics["eval_loss"]:
        idx = min(range(len(metrics["eval_loss"])), key=lambda i: metrics["eval_loss"][i])
        summary["best_eval_step"] = metrics["eval_steps"][idx]
    return summary


def write_summary(out_json: Path, summary):
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def plot_interactive_html(out_html: Path, metrics, summary, eval_interval: int):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    out_html.parent.mkdir(parents=True, exist_ok=True)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Loss Curves",
            "Learning Rate and Gradient Norm",
            "Validation Runtime and Throughput",
        ),
        specs=[[{"secondary_y": False}], [{"secondary_y": True}], [{"secondary_y": True}]],
    )

    fig.add_trace(
        go.Scatter(
            x=metrics["train_steps"],
            y=metrics["train_loss"],
            mode="lines",
            name="train_loss",
            line={"width": 2},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=metrics["eval_steps"],
            y=metrics["eval_loss"],
            mode="lines+markers",
            name="eval_loss",
            marker={"size": 7},
            line={"width": 2},
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=metrics["train_steps"],
            y=metrics["learning_rate"],
            mode="lines",
            name="learning_rate",
            line={"width": 2},
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=metrics["train_steps"],
            y=metrics["grad_norm"],
            mode="lines",
            name="grad_norm",
            line={"width": 1.5, "dash": "dot"},
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=metrics["eval_steps"],
            y=metrics["eval_runtime"],
            mode="lines+markers",
            name="eval_runtime_sec",
            marker={"size": 7},
            line={"width": 2},
        ),
        row=3,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=metrics["eval_steps"],
            y=metrics["eval_samples_per_second"],
            mode="lines+markers",
            name="eval_samples_per_sec",
            marker={"size": 7},
            line={"width": 2, "dash": "dot"},
        ),
        row=3,
        col=1,
        secondary_y=True,
    )

    # Stage markers: eval boundaries every eval_interval steps.
    if metrics["train_steps"] and eval_interval > 0:
        max_step = max(metrics["train_steps"])
        x = eval_interval
        while x <= max_step:
            fig.add_vline(x=x, line_width=1, line_dash="dot", line_color="rgba(120,120,120,0.25)")
            x += eval_interval

    best_eval = (
        f"best eval_loss={summary['best_eval_loss']:.4f} at step={summary['best_eval_step']}"
        if summary["best_eval_loss"] is not None
        else "best eval_loss=n/a"
    )

    fig.update_layout(
        title=(
            f"Training Dashboard | global_step={summary['global_step']} | "
            f"train points={summary['train_points']} | eval points={summary['eval_points']} | {best_eval}"
        ),
        height=980,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.03, "x": 0.0},
        margin={"l": 70, "r": 60, "t": 100, "b": 60},
    )

    fig.update_xaxes(title_text="Global step", row=3, col=1)
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="Learning rate", row=2, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Grad norm", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Eval runtime (sec)", row=3, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Eval samples/sec", row=3, col=1, secondary_y=True)

    fig.write_html(str(out_html), include_plotlyjs=True, full_html=True)


def main():
    parser = argparse.ArgumentParser(description="Plot training and validation curves from HF trainer_state.json")
    parser.add_argument("--trainer-state", type=Path, required=True, help="Path to trainer_state.json")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/plots"), help="Output directory")
    args = parser.parse_args()

    log_history, state = load_log_history(args.trainer_state)
    metrics = split_metrics(log_history)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "loss_curves.csv"
    write_csv(csv_path, metrics)

    summary = build_summary(metrics, state)
    summary_path = out_dir / "summary_metrics.json"
    write_summary(summary_path, summary)

    eval_interval = int(state.get("eval_steps") or 0)

    png_path = out_dir / "loss_curves.png"
    plotted = False
    try:
        plot_png(png_path, metrics, eval_interval=eval_interval)
        plotted = True
    except Exception as exc:
        # CSV is still useful even if matplotlib is unavailable.
        print(f"Plot skipped: {exc}")

    print(f"Wrote CSV: {csv_path}")
    if plotted:
        print(f"Wrote PNG: {png_path}")

    html_path = out_dir / "training_dashboard.html"
    wrote_html = False
    try:
        plot_interactive_html(html_path, metrics, summary, eval_interval=eval_interval)
        wrote_html = True
    except Exception as exc:
        print(f"Interactive HTML skipped: {exc}")

    print(f"Wrote summary JSON: {summary_path}")
    if wrote_html:
        print(f"Wrote interactive HTML: {html_path}")

    print(f"Train points: {len(metrics['train_steps'])}")
    print(f"Eval points: {len(metrics['eval_steps'])}")
    if metrics["eval_steps"]:
        print(f"Eval interval observed: ~{metrics['eval_steps'][1] - metrics['eval_steps'][0] if len(metrics['eval_steps']) > 1 else eval_interval} steps")


if __name__ == "__main__":
    main()
