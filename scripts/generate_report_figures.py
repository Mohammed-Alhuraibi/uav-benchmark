#!/usr/bin/env python3
"""Generate publication-quality figures for TeknoFest SAVASAN report.

Produces 6 figures in report_figures/ directory at 300 DPI.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path
import csv

OUT = Path("report_figures")
OUT.mkdir(exist_ok=True)
DPI = 300

# ── Data ──────────────────────────────────────────────────────────────
MODELS = ["YOLOv11s\nBaseline", "YOLOv11s+P2\n(Ours)", "YOLO26s", "YOLO26n"]
MODELS_SHORT = ["YOLOv11s", "YOLOv11s+P2", "YOLO26s", "YOLO26n"]
COLORS = ["#5B9BD5", "#2ECC71", "#E67E22", "#9B59B6"]
WINNER_IDX = 1  # YOLOv11s+P2

METRICS = {
    "mAP@0.5":     [0.9843, 0.9943, 0.9939, 0.9920],
    "mAP@0.5:0.95":[0.8343, 0.8365, 0.8505, 0.8155],
    "Precision":    [0.9914, 0.9937, 0.9803, 0.9708],
    "Recall":       [0.9813, 0.9972, 0.9835, 0.9661],
    "F1-Score":     [0.9864, 0.9955, 0.9819, 0.9684],
}
FPS = [30.8, 30.3, 29.9, 30.3]


def fig1_bar_chart():
    """Figure 1: Grouped bar chart comparing all 4 models."""
    fig, ax = plt.subplots(figsize=(10, 5.5))

    metric_names = list(METRICS.keys())
    x = np.arange(len(metric_names))
    n_models = len(MODELS)
    width = 0.18
    offsets = np.arange(n_models) - (n_models - 1) / 2

    for i, (model, color) in enumerate(zip(MODELS_SHORT, COLORS)):
        values = [METRICS[m][i] for m in metric_names]
        bars = ax.bar(
            x + offsets[i] * width, values, width,
            label=model, color=color, edgecolor="white", linewidth=0.5,
            zorder=3,
        )
        # Bold the winner bars
        if i == WINNER_IDX:
            for bar in bars:
                bar.set_edgecolor("#1a1a1a")
                bar.set_linewidth(1.5)

    # Value labels on winner bars only
    winner_values = [METRICS[m][WINNER_IDX] for m in metric_names]
    for j, val in enumerate(winner_values):
        ax.text(
            x[j] + offsets[WINNER_IDX] * width, val + 0.003,
            f"{val:.3f}", ha="center", va="bottom", fontsize=7,
            fontweight="bold", color=COLORS[WINNER_IDX],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Performance Comparison — UAV Detection Benchmark", fontsize=13, fontweight="bold")
    ax.set_ylim(0.78, 1.02)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(OUT / "fig1_model_comparison_bar.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [1/6] fig1_model_comparison_bar.png")


def fig2_radar_chart():
    """Figure 2: Radar/spider chart for multi-dimensional comparison."""
    categories = ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall", "F1-Score"]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for i, (model, color) in enumerate(zip(MODELS_SHORT, COLORS)):
        values = [METRICS[cat][i] for cat in categories]
        values += values[:1]

        lw = 2.5 if i == WINNER_IDX else 1.5
        alpha_fill = 0.15 if i == WINNER_IDX else 0.05
        ls = "-" if i == WINNER_IDX else "--"

        ax.plot(angles, values, "o-", linewidth=lw, label=model, color=color, linestyle=ls, markersize=4)
        ax.fill(angles, values, alpha=alpha_fill, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0.80, 1.0)
    ax.set_yticks([0.85, 0.90, 0.95, 1.0])
    ax.set_yticklabels(["0.85", "0.90", "0.95", "1.00"], fontsize=8, color="gray")
    ax.set_title("Multi-Metric Model Profile", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="lower right", bbox_to_anchor=(1.25, 0), fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT / "fig2_model_radar.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [2/6] fig2_model_radar.png")


def fig3_training_curves():
    """Figure 3: Training convergence — loss and mAP over epochs."""
    # Read results.csv
    csv_path = Path("reports(1)/runs/yolo11s_p2/results.csv")
    epochs, box_loss, cls_loss, dfl_loss = [], [], [], []
    map50, map50_95, prec, rec = [], [], [], []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            box_loss.append(float(row["train/box_loss"]))
            cls_loss.append(float(row["train/cls_loss"]))
            dfl_loss.append(float(row["train/dfl_loss"]))
            map50.append(float(row["metrics/mAP50(B)"]))
            map50_95.append(float(row["metrics/mAP50-95(B)"]))
            prec.append(float(row["metrics/precision(B)"]))
            rec.append(float(row["metrics/recall(B)"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: Training losses
    ax1 = axes[0]
    ax1.plot(epochs, box_loss, "-", color="#E74C3C", linewidth=1.5, label="Box Loss")
    ax1.plot(epochs, cls_loss, "-", color="#3498DB", linewidth=1.5, label="Class Loss")
    ax1.plot(epochs, dfl_loss, "-", color="#2ECC71", linewidth=1.5, label="DFL Loss")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.set_title("Training Loss Convergence", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right: Validation metrics
    ax2 = axes[1]
    ax2.plot(epochs, map50, "-", color="#2ECC71", linewidth=2, label="mAP@0.5")
    ax2.plot(epochs, map50_95, "-", color="#E67E22", linewidth=2, label="mAP@0.5:0.95")
    ax2.plot(epochs, prec, "--", color="#5B9BD5", linewidth=1.2, label="Precision", alpha=0.8)
    ax2.plot(epochs, rec, "--", color="#9B59B6", linewidth=1.2, label="Recall", alpha=0.8)
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Score", fontsize=10)
    ax2.set_title("Validation Metrics Progression", fontsize=12, fontweight="bold")
    ax2.set_ylim(0.5, 1.02)
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(alpha=0.3)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("YOLOv11s+P2 Training — 50 Epochs on MMFW-UAV Dataset", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_training_convergence.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [3/6] fig3_training_convergence.png")


def fig4_pipeline_architecture():
    """Figure 4: Detection pipeline flowchart."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4)
    ax.axis("off")

    # Stage definitions: (x, label, sublabel, color)
    stages = [
        (0.3,  "Camera\nCapture",     "IMX708\n2304x1296\n@56fps",      "#5DADE2"),
        (2.6,  "Preprocess",          "CLAHE\n(optional)\nResize",       "#48C9B0"),
        (4.9,  "YOLO\nDetection",     "YOLOv11s+P2\n640x640\nINT8",     "#2ECC71"),
        (7.2,  "ByteTrack\nTracker",  "Persistent IDs\nRe-identification","#F4D03F"),
        (9.5,  "Lock-on\nState Machine","4s timer\n200ms dropout\ntolerance","#E67E22"),
        (11.8, "Telemetry\nOutput",   "2Hz max\nPosition report\nto GCS","#E74C3C"),
    ]

    box_w = 1.9
    box_h = 1.6
    box_y = 1.8

    for x, label, sublabel, color in stages:
        # Main box
        rect = FancyBboxPatch(
            (x, box_y), box_w, box_h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="#2C3E50",
            linewidth=1.5, alpha=0.85,
        )
        ax.add_patch(rect)

        # Main label
        ax.text(
            x + box_w / 2, box_y + box_h * 0.65,
            label, ha="center", va="center",
            fontsize=9, fontweight="bold", color="#1a1a1a",
        )

        # Sub label (below box)
        ax.text(
            x + box_w / 2, box_y - 0.35,
            sublabel, ha="center", va="top",
            fontsize=7, color="#555555", style="italic",
        )

    # Arrows between stages
    arrow_y = box_y + box_h / 2
    for i in range(len(stages) - 1):
        x_start = stages[i][0] + box_w
        x_end = stages[i + 1][0]
        ax.annotate(
            "", xy=(x_end, arrow_y), xytext=(x_start, arrow_y),
            arrowprops=dict(
                arrowstyle="-|>", color="#2C3E50",
                lw=2, mutation_scale=15,
            ),
        )

    # Title
    ax.text(
        7, 3.8, "Detection and Lock-on Pipeline",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#2C3E50",
    )

    # Confidence thresholds annotation
    ax.annotate(
        "det_conf = 0.35\n(all detections)",
        xy=(5.85, box_y), xytext=(5.5, 0.5),
        fontsize=7, color="#2ECC71", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#2ECC71", lw=1),
    )
    ax.annotate(
        "lock_conf = 0.75\n(lock-on only)",
        xy=(10.3, box_y), xytext=(10.0, 0.5),
        fontsize=7, color="#E67E22", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#E67E22", lw=1),
    )

    fig.tight_layout()
    fig.savefig(OUT / "fig4_pipeline_architecture.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [4/6] fig4_pipeline_architecture.png")


def fig5_visual_servoing():
    """Figure 5: Visual servoing — image error to attitude control."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [1.2, 1]})

    # ── Left: Camera frame with error vectors ──
    ax = axes[0]
    ax.set_xlim(0, 640)
    ax.set_ylim(0, 640)
    ax.invert_yaxis()
    ax.set_aspect("equal")

    # Camera frame border
    frame = mpatches.FancyBboxPatch(
        (10, 10), 620, 620,
        boxstyle="round,pad=5", facecolor="#E8F4FD",
        edgecolor="#2C3E50", linewidth=2,
    )
    ax.add_patch(frame)

    # Image center crosshair
    cx, cy = 320, 320
    ax.plot([cx - 30, cx + 30], [cy, cy], "-", color="#7f8c8d", lw=1.5, alpha=0.6)
    ax.plot([cx, cx], [cy - 30, cy + 30], "-", color="#7f8c8d", lw=1.5, alpha=0.6)
    ax.plot(cx, cy, "o", color="#7f8c8d", markersize=6, alpha=0.6)
    ax.text(cx + 35, cy - 10, "Image\nCenter", fontsize=8, color="#7f8c8d")

    # Target UAV bbox (offset from center)
    tx, ty = 440, 200  # target center
    tw, th = 100, 40   # target size
    target_rect = mpatches.FancyBboxPatch(
        (tx - tw/2, ty - th/2), tw, th,
        boxstyle="round,pad=2", facecolor="#2ECC71",
        edgecolor="#27AE60", linewidth=2, alpha=0.7,
    )
    ax.add_patch(target_rect)
    ax.text(tx, ty, "UAV", ha="center", va="center", fontsize=9, fontweight="bold", color="#1a1a1a")
    ax.plot(tx, ty, "+", color="#E74C3C", markersize=12, markeredgewidth=2)

    # Error vectors
    # Horizontal error (ex)
    ax.annotate(
        "", xy=(tx, cy), xytext=(cx, cy),
        arrowprops=dict(arrowstyle="-|>", color="#E74C3C", lw=2.5),
    )
    ax.text((cx + tx) / 2, cy + 20, r"$e_x$", fontsize=14, fontweight="bold",
            color="#E74C3C", ha="center")

    # Vertical error (ey)
    ax.annotate(
        "", xy=(tx, ty), xytext=(tx, cy),
        arrowprops=dict(arrowstyle="-|>", color="#3498DB", lw=2.5),
    )
    ax.text(tx + 20, (cy + ty) / 2, r"$e_y$", fontsize=14, fontweight="bold",
            color="#3498DB", ha="left")

    # Diagonal dashed line
    ax.plot([cx, tx], [cy, ty], "--", color="#95a5a6", lw=1, alpha=0.5)

    ax.set_title("Camera Frame — Error Extraction", fontsize=12, fontweight="bold")
    ax.set_xlabel("Pixel X", fontsize=9)
    ax.set_ylabel("Pixel Y", fontsize=9)

    # ── Right: Control mapping table ──
    ax2 = axes[1]
    ax2.axis("off")

    # Title
    ax2.text(0.5, 0.95, "Image Error to Attitude Control",
             ha="center", va="top", fontsize=12, fontweight="bold",
             transform=ax2.transAxes)

    # Mapping boxes
    mappings = [
        ("$e_x > 0$", "Target is RIGHT", "Roll RIGHT\n(or Yaw RIGHT)", "#E74C3C", 0.82),
        ("$e_x < 0$", "Target is LEFT",  "Roll LEFT\n(or Yaw LEFT)",   "#E74C3C", 0.66),
        ("$e_y > 0$", "Target is BELOW", "Pitch DOWN",                  "#3498DB", 0.50),
        ("$e_y < 0$", "Target is ABOVE", "Pitch UP",                    "#3498DB", 0.34),
    ]

    for error_eq, meaning, command, color, y_pos in mappings:
        # Error box
        err_box = FancyBboxPatch(
            (0.02, y_pos - 0.04), 0.22, 0.10,
            boxstyle="round,pad=0.01", facecolor="#f0f0f0",
            edgecolor=color, linewidth=1.5,
            transform=ax2.transAxes,
        )
        ax2.add_patch(err_box)
        ax2.text(0.13, y_pos + 0.01, error_eq, ha="center", va="center",
                 fontsize=11, transform=ax2.transAxes)

        # Arrow
        ax2.annotate(
            "", xy=(0.50, y_pos + 0.01), xytext=(0.26, y_pos + 0.01),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5),
            xycoords=ax2.transAxes, textcoords=ax2.transAxes,
        )

        # Meaning
        ax2.text(0.38, y_pos + 0.01, meaning, ha="center", va="center",
                 fontsize=7, color="#555", transform=ax2.transAxes, style="italic")

        # Command box
        cmd_box = FancyBboxPatch(
            (0.52, y_pos - 0.04), 0.38, 0.10,
            boxstyle="round,pad=0.01", facecolor=color,
            edgecolor=color, linewidth=1.5, alpha=0.2,
            transform=ax2.transAxes,
        )
        ax2.add_patch(cmd_box)
        ax2.text(0.71, y_pos + 0.01, command, ha="center", va="center",
                 fontsize=9, fontweight="bold", transform=ax2.transAxes)

    # Speed control section
    ax2.plot([0.05, 0.90], [0.22, 0.22], "-", color="#95a5a6", lw=0.8,
             transform=ax2.transAxes)
    ax2.text(0.5, 0.18, "Speed Control from BBox Size",
             ha="center", va="center", fontsize=10, fontweight="bold",
             color="#2C3E50", transform=ax2.transAxes)

    speed_mappings = [
        ("BBox LARGE", "Target close", "Reduce speed", "#E67E22", 0.10),
        ("BBox SMALL", "Target far",   "Increase speed", "#27AE60", 0.02),
    ]
    for label, meaning, cmd, color, y_pos in speed_mappings:
        ax2.text(0.13, y_pos + 0.01, label, ha="center", va="center",
                 fontsize=9, transform=ax2.transAxes, fontweight="bold", color=color)
        ax2.text(0.38, y_pos + 0.01, meaning, ha="center", va="center",
                 fontsize=7, color="#555", transform=ax2.transAxes, style="italic")
        ax2.annotate(
            "", xy=(0.50, y_pos + 0.01), xytext=(0.26, y_pos + 0.01),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2),
            xycoords=ax2.transAxes, textcoords=ax2.transAxes,
        )
        ax2.text(0.71, y_pos + 0.01, cmd, ha="center", va="center",
                 fontsize=9, fontweight="bold", transform=ax2.transAxes)

    fig.tight_layout()
    fig.savefig(OUT / "fig5_visual_servoing.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [5/6] fig5_visual_servoing.png")


def fig6_p2_head_architecture():
    """Figure 6: P2 detection head — why stride-4 matters for small objects."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    titles = ["Standard 3-Scale Head (P3/P4/P5)", "Our P2-Enhanced 4-Scale Head (P2/P3/P4/P5)"]
    configs = [
        # (scale_name, feature_size, stride, receptive, color, min_obj_label)
        [
            ("P3", "80x80",   8,  "#5DADE2", "~8 px"),
            ("P4", "40x40",  16,  "#3498DB", "~16 px"),
            ("P5", "20x20",  32,  "#2471A3", "~32 px"),
        ],
        [
            ("P2", "160x160",  4, "#2ECC71", "~4 px"),
            ("P3", "80x80",    8, "#5DADE2", "~8 px"),
            ("P4", "40x40",   16, "#3498DB", "~16 px"),
            ("P5", "20x20",   32, "#2471A3", "~32 px"),
        ],
    ]

    for idx, (ax, title, scales) in enumerate(zip(axes, titles, configs)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

        n = len(scales)
        total_h = 8
        gap = 0.3
        box_h = (total_h - (n - 1) * gap) / n
        x_start = 1
        box_w = 8

        for i, (name, feat_size, stride, color, min_obj) in enumerate(scales):
            y = 9 - (i + 1) * (box_h + gap) + gap
            # Scale width proportional to feature map size
            scale_factor = [0.4, 0.55, 0.75, 1.0] if n == 4 else [0.55, 0.75, 1.0]
            w = box_w * scale_factor[n - 1 - i]
            x = x_start + (box_w - w) / 2

            alpha = 0.9 if (name == "P2" and idx == 1) else 0.6
            lw = 2.5 if (name == "P2" and idx == 1) else 1

            rect = FancyBboxPatch(
                (x, y), w, box_h,
                boxstyle="round,pad=0.08",
                facecolor=color, edgecolor="#2C3E50",
                linewidth=lw, alpha=alpha,
            )
            ax.add_patch(rect)

            # Labels
            label = f"{name}  |  {feat_size}  |  stride {stride}"
            ax.text(x + w / 2, y + box_h / 2 + 0.15, label,
                    ha="center", va="center", fontsize=9, fontweight="bold")
            ax.text(x + w / 2, y + box_h / 2 - 0.35, f"Min detectable: {min_obj}",
                    ha="center", va="center", fontsize=7, color="#333", style="italic")

        # Arrow and label for P2
        if idx == 1:
            # Highlight P2
            ax.text(9.5, 9 - (box_h + gap) + gap + box_h / 2,
                    "NEW",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="#2ECC71",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#2ECC71", alpha=0.2))

        # "Input 640x640" label at top
        ax.text(5, 9.5, "Input: 640 x 640", ha="center", va="center",
                fontsize=9, color="#7f8c8d")

    # Bottom annotation
    fig.text(
        0.5, 0.01,
        "P2 head adds a 160x160 feature map that detects objects as small as ~4 pixels — "
        "critical for distant UAV targets",
        ha="center", fontsize=10, style="italic", color="#2C3E50",
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(OUT / "fig6_p2_head_architecture.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("  [6/6] fig6_p2_head_architecture.png")


if __name__ == "__main__":
    print("Generating report figures...\n")
    fig1_bar_chart()
    fig2_radar_chart()
    fig3_training_curves()
    fig4_pipeline_architecture()
    fig5_visual_servoing()
    fig6_p2_head_architecture()
    print(f"\nAll figures saved to {OUT}/")
