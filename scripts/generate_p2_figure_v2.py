#!/usr/bin/env python3
"""P2 head figure v2 — shows parallel branches, not stacked sequence."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

DPI = 300

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
fig.subplots_adjust(wspace=0.35)

for idx, (ax, title, has_p2) in enumerate(zip(
    axes,
    ["Standard 3-Scale Head (P3/P4/P5)", "Our P2-Enhanced 4-Scale Head"],
    [False, True],
)):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

    # ── Backbone column ──
    bb_x = 0.5
    bb_w = 2.0

    # Backbone box
    bb = FancyBboxPatch(
        (bb_x, 1.8), bb_w, 7.0,
        boxstyle="round,pad=0.15",
        facecolor="#D6E4F0", edgecolor="#2C3E50", linewidth=1.5,
    )
    ax.add_patch(bb)

    # Input label above backbone
    ax.text(bb_x + bb_w / 2, 9.3, "Input\n640 x 640",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#2C3E50")

    # Backbone label below
    ax.text(bb_x + bb_w / 2, 1.1, "Backbone (CNN)",
            ha="center", va="center", fontsize=9, fontweight="bold", color="#2C3E50")

    # Layer positions inside backbone
    layer_positions = {
        "Layer 2": 7.6,
        "Layer 3": 6.1,
        "Layer 4": 4.6,
        "Layer 5": 3.1,
    }

    # Draw layer labels inside backbone
    for layer_name, y_pos in layer_positions.items():
        if not has_p2 and layer_name == "Layer 2":
            color = "#AAAAAA"
        else:
            color = "#2C3E50"
        ax.text(bb_x + bb_w / 2, y_pos, layer_name,
                ha="center", va="center", fontsize=8, color=color, alpha=0.7)

    # ── Detection heads ──
    head_x = 5.0
    box_w = 4.5       # same width for all — text always fits
    box_h = 1.05

    if has_p2:
        heads = [
            ("P2", "160x160", "stride 4", "~4 px",  7.6, "#2ECC71", True),
            ("P3", "80x80",   "stride 8", "~8 px",  6.1, "#5DADE2", False),
            ("P4", "40x40",   "stride 16","~16 px", 4.6, "#3498DB", False),
            ("P5", "20x20",   "stride 32","~32 px", 3.1, "#2471A3", False),
        ]
    else:
        heads = [
            ("P3", "80x80",   "stride 8", "~8 px",  6.1, "#5DADE2", False),
            ("P4", "40x40",   "stride 16","~16 px", 4.6, "#3498DB", False),
            ("P5", "20x20",   "stride 32","~32 px", 3.1, "#2471A3", False),
        ]

    for name, feat, stride, min_det, y, color, is_new in heads:
        # Dot on backbone edge
        ax.plot(bb_x + bb_w, y, "o", color=color, markersize=7, zorder=5)

        # Arrow from backbone to head
        ax.annotate(
            "", xy=(head_x, y), xytext=(bb_x + bb_w + 0.05, y),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=2),
        )

        # Detection head box
        alpha = 0.9 if is_new else 0.6
        lw = 2.5 if is_new else 1.2
        head_box = FancyBboxPatch(
            (head_x, y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.1",
            facecolor=color, edgecolor="#2C3E50",
            linewidth=lw, alpha=alpha,
        )
        ax.add_patch(head_box)

        # Main label (centered in box)
        ax.text(head_x + box_w / 2, y + 0.15,
                f"{name}  |  {feat}  |  {stride}",
                ha="center", va="center", fontsize=9.5, fontweight="bold",
                color="#1a1a1a")
        # Sub label
        ax.text(head_x + box_w / 2, y - 0.25,
                f"Min detectable: {min_det}",
                ha="center", va="center", fontsize=7.5, color="#333",
                style="italic")

        # NEW badge
        if is_new:
            ax.text(head_x + box_w + 0.4, y, "NEW",
                    ha="left", va="center", fontsize=11, fontweight="bold",
                    color="#27AE60",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#D5F5E3",
                              edgecolor="#27AE60", linewidth=1.5))

    # Grayed-out Layer 2 for standard model
    if not has_p2:
        y2 = layer_positions["Layer 2"]
        ax.plot(bb_x + bb_w, y2, "o", color="#CCCCCC", markersize=7, zorder=5)
        ax.plot([bb_x + bb_w + 0.1, bb_x + bb_w + 1.5], [y2, y2],
                "--", color="#CCCCCC", lw=1.2)
        ax.text(bb_x + bb_w + 1.7, y2, "not used",
                ha="left", va="center", fontsize=8, color="#999999",
                style="italic")

    # "PARALLEL" annotation on right side
    if has_p2:
        ax.text(head_x + box_w / 2, 1.5, "All 4 heads run in PARALLEL",
                ha="center", va="center", fontsize=9, color="#7f8c8d",
                style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F0F0",
                          edgecolor="#CCCCCC", linewidth=0.8))

# Bottom caption
fig.text(0.5, 0.01,
         "Each detection head taps into a different backbone layer simultaneously "
         "- P2 connects to an early high-resolution layer that was previously unused",
         ha="center", fontsize=10, style="italic", color="#2C3E50")

fig.tight_layout(rect=[0, 0.05, 1, 0.95])
fig.savefig("report_figures/fig6_p2_head_architecture.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("Saved: report_figures/fig6_p2_head_architecture.png")
