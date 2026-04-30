#!/usr/bin/env python3
"""Generate a professional table figure for the model comparison."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 300

models = ["YOLOv11s", "YOLOv11s + P2\n(Selected)", "YOLO26s", "YOLO26n"]

col_labels = ["Model", "mAP@0.5", "mAP\n@0.5:0.95", "Precision", "Recall", "F1",  "FPS",
              "Advantage", "Disadvantage"]

cell_text = [
    ["YOLOv11s",
     "0.984", "0.834", "0.991", "0.981", "0.986", "30.8",
     "Fastest model;\nmature and simple to deploy",
     "3-scale head misses\nsmallest targets"],
    ["YOLOv11s\n+ P2",
     "0.994", "0.837", "0.994", "0.997", "0.996", "30.3",
     "Highest recall + precision;\ndetects targets down to ~4px",
     "Slightly more parameters\nthan baseline"],
    ["YOLO26s",
     "0.994", "0.851", "0.980", "0.984", "0.982", "29.9",
     "Best localization accuracy;\nNMS-free architecture",
     "Lower precision raises\nfalse lock-on risk (-30pt)"],
    ["YOLO26n",
     "0.992", "0.816", "0.971", "0.966", "0.968", "30.3",
     "Smallest model;\nultra-low power friendly",
     "Weakest overall;\ninsufficient for competition"],
]

fig, ax = plt.subplots(figsize=(14, 5))
ax.axis("off")

table = ax.table(
    cellText=cell_text,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)

n_cols = len(col_labels)
n_rows = len(cell_text)

# Column widths — wider for text columns
col_widths = [0.08, 0.06, 0.06, 0.06, 0.06, 0.05, 0.05, 0.22, 0.22]
row_height = 0.18

for j in range(n_cols):
    for i in range(n_rows + 1):  # +1 for header
        cell = table[i, j]
        cell.set_width(col_widths[j])
        cell.set_height(row_height)

# ── Header styling ──
for j in range(n_cols):
    cell = table[0, j]
    cell.set_facecolor("#1B2A4A")
    cell.set_text_props(color="white", fontweight="bold", fontsize=9)
    cell.set_edgecolor("#1B2A4A")
    cell.set_linewidth(0)
    cell.set_height(0.16)

# ── Data rows ──
WINNER = 1  # row index (0-based)

for i in range(n_rows):
    ri = i + 1  # table row index (header = 0)
    is_winner = (i == WINNER)

    for j in range(n_cols):
        cell = table[ri, j]
        cell.set_edgecolor("#E5E7EB")
        cell.set_linewidth(0.7)

        # Row background
        if is_winner:
            cell.set_facecolor("#DCFCE7")  # light green
        else:
            cell.set_facecolor("#FFFFFF" if i % 2 else "#F9FAFB")

        # Default text
        cell.set_text_props(fontsize=9)

        # Model name column
        if j == 0:
            cell.get_text().set_ha("left")
            if is_winner:
                cell.set_text_props(fontsize=9.5, fontweight="bold", color="#15803D")
            else:
                cell.set_text_props(fontsize=9.5, fontweight="bold", color="#1E3A5F")

        # Metric columns (1-6): bold the best value per column
        # Best per column: mAP50→row1,2 tie; mAP50-95→row2; P→row1; R→row1; F1→row1; FPS→row0
        best_map = {1: [1, 2], 2: [2], 3: [1], 4: [1], 5: [1], 6: [0]}
        if j in best_map and i in best_map[j]:
            color = "#15803D" if is_winner else "#1E3A5F"
            cell.set_text_props(fontsize=9.5, fontweight="bold", color=color)

        # Advantage column
        if j == 7:
            cell.get_text().set_ha("left")
            cell.set_text_props(
                fontsize=8.5,
                color="#15803D" if is_winner else "#166534",
                fontweight="bold" if is_winner else "normal",
            )

        # Disadvantage column
        if j == 8:
            cell.get_text().set_ha("left")
            cell.set_text_props(
                fontsize=8.5,
                color="#B91C1C" if not is_winner else "#92400E",
                fontweight="bold" if is_winner else "normal",
            )

# Title
ax.set_title(
    "Comparison of Evaluated Detection Models on MMFW-UAV Dataset",
    fontsize=14, fontweight="bold", color="#1B2A4A",
    pad=12,
)

fig.tight_layout(pad=1.5)
fig.savefig("report_figures/fig_model_comparison_table.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("Saved: report_figures/fig_model_comparison_table.png")
