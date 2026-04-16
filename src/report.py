"""Generate benchmark comparison reports.

Produces:
  - comparison_table.csv — all experiments × all metrics
  - loss_curves.png — overlaid training loss curves
  - visual_grid.png — same test images, predictions from all models
  - precision_recall.png — overlaid P/R curves
  - benchmark_summary.md — formatted summary
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"


def generate_comparison_table(all_results: dict) -> pd.DataFrame:
    """Build a comparison table from evaluation results."""
    rows = []
    for name, res in all_results.items():
        std = res.get("standard", {})
        speed = res.get("speed", {})
        tiny = res.get("tiny", {})
        row = {
            "experiment": name,
            "mAP50": round(std.get("metrics/mAP50(B)", 0), 4),
            "mAP50-95": round(std.get("metrics/mAP50-95(B)", 0), 4),
            "precision": round(std.get("metrics/precision(B)", 0), 4),
            "recall": round(std.get("metrics/recall(B)", 0), 4),
            "f1": round(
                2 * std.get("metrics/precision(B)", 0) * std.get("metrics/recall(B)", 0)
                / max(std.get("metrics/precision(B)", 0) + std.get("metrics/recall(B)", 0), 1e-8),
                4,
            ),
            "tiny_images": tiny.get("tiny_images", "?"),
            "tiny_ratio": round(tiny.get("tiny_ratio", 0), 3),
            "speed_ms": speed.get("mean_ms", "?"),
            "fps": speed.get("fps", "?"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("mAP50-95", ascending=False).reset_index(drop=True)
    return df


def plot_loss_curves(experiments: list[str]) -> None:
    """Overlay training loss curves from all experiments."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    loss_names = ["train/box_loss", "train/cls_loss", "train/dfl_loss"]
    titles = ["Box Loss", "Classification Loss", "DFL Loss"]

    for name in experiments:
        csv_path = ROOT / "runs" / name / "results.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        for ax, col, title in zip(axes, loss_names, titles):
            col = col.strip()
            if col in df.columns:
                ax.plot(df["epoch"], df[col], label=name, linewidth=1.5)
                ax.set_title(title)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")

    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = REPORTS_DIR / "loss_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_metric_curves(experiments: list[str]) -> None:
    """Overlay mAP and recall curves across epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics = ["metrics/mAP50(B)", "metrics/mAP50-95(B)"]
    titles = ["mAP@0.5", "mAP@0.5:0.95"]

    for name in experiments:
        csv_path = ROOT / "runs" / name / "results.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        for ax, col, title in zip(axes, metrics, titles):
            col = col.strip()
            if col in df.columns:
                ax.plot(df["epoch"], df[col], label=name, linewidth=1.5)
                ax.set_title(title)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Metric")

    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = REPORTS_DIR / "metric_curves.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_visual_grid(experiments: list[str]) -> None:
    """Show predictions from all models on the same test images."""
    import random
    from ultralytics import YOLO

    # Pick 6 random test images (consistent seed)
    test_imgs = sorted((ROOT / "data" / "images" / "test").glob("*"))
    if not test_imgs:
        print("  WARNING: No test images found, skipping visual grid")
        return

    rng = random.Random(42)
    samples = rng.sample(test_imgs, min(6, len(test_imgs)))

    n_models = len(experiments)
    n_imgs = len(samples)
    fig, axes = plt.subplots(n_imgs, n_models, figsize=(5 * n_models, 5 * n_imgs))
    if n_models == 1:
        axes = axes.reshape(-1, 1)
    if n_imgs == 1:
        axes = axes.reshape(1, -1)

    for col, name in enumerate(experiments):
        weights = ROOT / "runs" / name / "weights" / "best.pt"
        if not weights.exists():
            continue
        model = YOLO(str(weights))
        for row, img_path in enumerate(samples):
            results = model.predict(str(img_path), imgsz=640, verbose=False, conf=0.25)
            plot = results[0].plot()
            # BGR → RGB
            axes[row, col].imshow(plot[:, :, ::-1])
            axes[row, col].set_title(f"{name}\n{img_path.stem}", fontsize=8)
            axes[row, col].axis("off")

    plt.tight_layout()
    out = REPORTS_DIR / "visual_grid.png"
    plt.savefig(out, dpi=100)
    plt.close()
    print(f"  Saved: {out}")


def generate_summary_md(df: pd.DataFrame, all_results: dict) -> None:
    """Write a formatted markdown summary."""
    out = REPORTS_DIR / "benchmark_summary.md"

    lines = [
        "# UAV Detection Benchmark Results\n",
        "## Comparison Table\n",
        df.to_markdown(index=False),
        "\n",
        "## Key Findings\n",
    ]

    # Auto-generate findings
    if not df.empty:
        best_map = df.iloc[0]
        lines.append(f"- **Best mAP@0.5:0.95**: {best_map['experiment']} ({best_map['mAP50-95']})")
        best_recall = df.sort_values("recall", ascending=False).iloc[0]
        lines.append(f"- **Best Recall**: {best_recall['experiment']} ({best_recall['recall']})")
        best_speed = df.sort_values("speed_ms").iloc[0]
        lines.append(f"- **Fastest**: {best_speed['experiment']} ({best_speed['speed_ms']} ms/img)")
        lines.append(f"- **Tiny object images in test set**: {best_map['tiny_images']} ({best_map['tiny_ratio']*100:.1f}%)")

    lines.extend([
        "\n## Generated Artifacts\n",
        "- `comparison_table.csv` — full metrics table",
        "- `loss_curves.png` — training loss overlay",
        "- `metric_curves.png` — mAP progression overlay",
        "- `visual_grid.png` — side-by-side predictions",
        "",
    ])

    out.write_text("\n".join(lines))
    print(f"  Saved: {out}")


def generate_report(all_results: dict, config: dict) -> None:
    """Generate all report artifacts."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    experiments = [
        name for name in config["experiments"]
        if (ROOT / "runs" / name / "weights").exists()
    ]

    print("\n  Generating comparison table...")
    df = generate_comparison_table(all_results)
    df.to_csv(REPORTS_DIR / "comparison_table.csv", index=False)
    print(f"  Saved: {REPORTS_DIR / 'comparison_table.csv'}")
    print()
    print(df.to_string(index=False))

    print("\n  Generating loss curves...")
    plot_loss_curves(experiments)

    print("  Generating metric curves...")
    plot_metric_curves(experiments)

    print("  Generating visual grid...")
    plot_visual_grid(experiments)

    print("  Generating summary...")
    generate_summary_md(df, all_results)

    print(f"\n  All reports saved to: {REPORTS_DIR}")
