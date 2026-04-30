#!/usr/bin/env python3
"""Generate a clear model comparison grid: 3 images × 4 models.

Picks small, medium, and large target images to show where models differ.
Shows ground-truth (green) vs prediction (colored) with confidence scores.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

DPI = 300
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "dataset"
OUT = ROOT / "report_figures"

# Models in order
MODELS = [
    ("YOLOv11s", ROOT / "reports(1)" / "runs" / "yolo11s_baseline" / "weights" / "best.pt"),
    ("YOLOv11s + P2", ROOT / "reports(1)" / "runs" / "yolo11s_p2" / "weights" / "best.pt"),
    ("YOLO26s", ROOT / "reports(1)" / "runs" / "yolo26s" / "weights" / "best.pt"),
    ("YOLO26n", ROOT / "reports(1)" / "runs" / "yolo26n" / "weights" / "best.pt"),
]

# Strategic image picks: small, medium, large target
SAMPLES = [
    ("000314.jpg", "Small target\n(~24×19 px)"),
    ("000018.jpg", "Medium target\n(~58×56 px)"),
    ("000448.jpg", "Large target\n(~978×516 px)"),
]

# Colors for each model (BGR for cv2, then we convert)
MODEL_COLORS_RGB = [
    (93, 173, 226),    # blue
    (46, 204, 113),    # green (P2 — winner)
    (241, 196, 15),    # yellow
    (231, 76, 60),     # red
]


def load_gt(img_name):
    """Load ground-truth bounding boxes in pixel coords."""
    lbl = img_name.replace(".jpg", ".txt").replace(".png", ".txt")
    # Check test first, then train, then val
    for split in ["test", "train", "val"]:
        lbl_path = DATA / "labels" / split / lbl
        img_path = DATA / "images" / split / img_name
        if img_path.exists():
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            bboxes = []
            if lbl_path.exists():
                with open(lbl_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 5:
                            cx, cy, bw, bh = map(float, parts[1:])
                            x1 = int((cx - bw / 2) * w)
                            y1 = int((cy - bh / 2) * h)
                            x2 = int((cx + bw / 2) * w)
                            y2 = int((cy + bh / 2) * h)
                            bboxes.append((x1, y1, x2, y2))
            return img, bboxes
    return None, []


def draw_gt_box(img, bboxes):
    """Draw ground truth with dashed green outline."""
    out = img.copy()
    for (x1, y1, x2, y2) in bboxes:
        # Dashed effect: draw segments
        for start in range(x1, x2, 12):
            end = min(start + 6, x2)
            cv2.line(out, (start, y1), (end, y1), (0, 200, 0), 2)
            cv2.line(out, (start, y2), (end, y2), (0, 200, 0), 2)
        for start in range(y1, y2, 12):
            end = min(start + 6, y2)
            cv2.line(out, (x1, start), (x1, end), (0, 200, 0), 2)
            cv2.line(out, (x2, start), (x2, end), (0, 200, 0), 2)
    return out


def draw_pred_box(img, x1, y1, x2, y2, conf, color, thickness=2):
    """Draw prediction box with confidence label."""
    out = img.copy()
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    # Confidence label
    label = f"{conf:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
    # Background for text
    cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(out, label, (x1 + 3, y1 - 5), font, font_scale,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def add_zoom_inset(ax, img, bbox, zoom_factor=3):
    """Add a zoomed inset showing the target area."""
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    # Expand crop region
    pad = max((x2 - x1), (y2 - y1)) * 1.5
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    crop_x1 = max(0, int(cx - pad))
    crop_y1 = max(0, int(cy - pad))
    crop_x2 = min(w, int(cx + pad))
    crop_y2 = min(h, int(cy + pad))
    crop = img[crop_y1:crop_y2, crop_x1:crop_x2]

    if crop.size == 0:
        return

    # Place inset in bottom-right
    inset_ax = ax.inset_axes([0.55, 0.02, 0.43, 0.43])
    inset_ax.imshow(crop)
    inset_ax.set_xticks([])
    inset_ax.set_yticks([])
    for spine in inset_ax.spines.values():
        spine.set_edgecolor("#FFD700")
        spine.set_linewidth(2)

    # Draw indicator rectangle on main image
    rect = plt.Rectangle((crop_x1, crop_y1), crop_x2 - crop_x1, crop_y2 - crop_y1,
                          fill=False, edgecolor="#FFD700", linewidth=1.5, linestyle="--")
    ax.add_patch(rect)


# ── Build the figure ──
n_rows = len(SAMPLES)
n_cols = len(MODELS) + 1  # +1 for ground truth column
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 11))

# Load models
loaded_models = []
for name, path in MODELS:
    print(f"Loading {name}...")
    model = YOLO(str(path))
    loaded_models.append((name, model))

for row, (img_name, row_label) in enumerate(SAMPLES):
    img, gt_bboxes = load_gt(img_name)
    if img is None:
        print(f"WARNING: {img_name} not found")
        continue

    is_small = row == 0  # add zoom inset for small targets

    # Column 0: Ground Truth
    ax = axes[row, 0]
    gt_img = draw_gt_box(img, gt_bboxes)
    ax.imshow(gt_img)
    ax.set_xticks([])
    ax.set_yticks([])
    if row == 0:
        ax.set_title("Ground Truth", fontsize=12, fontweight="bold",
                      color="#2C3E50", pad=10)
    ax.set_ylabel(row_label, fontsize=10, fontweight="bold",
                  rotation=0, labelpad=75, va="center")

    if is_small and gt_bboxes:
        add_zoom_inset(ax, gt_img, gt_bboxes[0])

    # Columns 1-4: Each model's prediction
    for col, (model_name, model) in enumerate(loaded_models):
        ax = axes[row, col + 1]
        color = MODEL_COLORS_RGB[col]

        # Run inference
        results = model.predict(str(DATA / "images" / "train" / img_name),
                                imgsz=640, verbose=False, conf=0.25)

        # Start with GT boxes (dashed green)
        disp = draw_gt_box(img, gt_bboxes)

        # Draw predictions
        boxes = results[0].boxes
        detected = False
        if len(boxes) > 0:
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
                conf = boxes.conf[i].cpu().item()
                disp = draw_pred_box(disp, xyxy[0], xyxy[1], xyxy[2], xyxy[3],
                                     conf, color, thickness=3)
                detected = True

        ax.imshow(disp)
        ax.set_xticks([])
        ax.set_yticks([])

        if row == 0:
            # Header with model name
            header_color = "#15803D" if col == 1 else "#2C3E50"
            star = " \u2605" if col == 1 else ""
            ax.set_title(f"{model_name}{star}", fontsize=12,
                         fontweight="bold", color=header_color, pad=10)

        # Add zoom inset for small targets
        if is_small and gt_bboxes:
            add_zoom_inset(ax, disp, gt_bboxes[0])

        # MISS label if nothing detected
        if not detected:
            ax.text(0.5, 0.5, "MISS", fontsize=28, ha="center", va="center",
                    transform=ax.transAxes, color="red", fontweight="bold",
                    alpha=0.7,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="red", linewidth=2, alpha=0.8))

# Legend at bottom
fig.text(0.5, 0.01,
         "Green dashed = ground truth  |  Colored solid = model prediction with confidence  |  "
         "Yellow inset = zoomed view of target area",
         ha="center", fontsize=10, style="italic", color="#555")

fig.suptitle("Detection Comparison Across Models — MMFW-UAV Dataset",
             fontsize=15, fontweight="bold", color="#1B2A4A", y=0.98)

fig.tight_layout(rect=[0.06, 0.03, 1, 0.95])
fig.savefig(OUT / "fig10_model_visual_comparison.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("\nSaved: report_figures/fig10_model_visual_comparison.png")
