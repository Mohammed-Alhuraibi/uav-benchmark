#!/usr/bin/env python3
"""Visualize each augmentation transform on real dataset images.

Generates a grid: rows = sample images, columns = original + each transform.
Uses the same settings from configs/augmentation.yaml.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import random

random.seed(42)
np.random.seed(42)

DPI = 300
DATA_DIR = "data/dataset"
IMG_DIR = os.path.join(DATA_DIR, "images", "train")
LBL_DIR = os.path.join(DATA_DIR, "labels", "train")
OUT_DIR = "report_figures"

# Three sample images: small target, medium target, large target
SAMPLES = [
    ("000314.jpg", "Small target"),
    ("000018.jpg", "Medium target"),
    ("000436.jpg", "Large target"),
]


def load_image_and_bbox(img_name):
    """Load image and YOLO-format bounding box."""
    img = cv2.imread(os.path.join(IMG_DIR, img_name))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    lbl_name = img_name.replace(".jpg", ".txt").replace(".png", ".txt")
    lbl_path = os.path.join(LBL_DIR, lbl_name)
    bboxes = []
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cx, cy, bw, bh = map(float, parts[1:])
                    # Convert normalized to pixel coords
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    bboxes.append((x1, y1, x2, y2))
    return img, bboxes, (h, w)


def draw_bbox(img, bboxes, color=(0, 255, 0), thickness=2):
    """Draw bounding boxes on image copy."""
    out = img.copy()
    for (x1, y1, x2, y2) in bboxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        # Small crosshair at center
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        sz = max(6, min(20, (x2 - x1) // 4))
        cv2.line(out, (cx - sz, cy), (cx + sz, cy), color, 1)
        cv2.line(out, (cx, cy - sz), (cx, cy + sz), color, 1)
    return out


# ── Augmentation functions (matching configs/augmentation.yaml) ──

def aug_fliplr(img, bboxes, hw):
    """Horizontal flip (fliplr: 0.5)"""
    h, w = hw
    flipped = cv2.flip(img, 1)
    new_bboxes = [(w - x2, y1, w - x1, y2) for (x1, y1, x2, y2) in bboxes]
    return flipped, new_bboxes


def aug_scale(img, bboxes, hw):
    """Random scale (scale: 0.5) — zoom out by factor 0.5"""
    h, w = hw
    scale = 0.6  # show a visible zoom-out
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))
    # Place on gray canvas
    canvas = np.full_like(img, 114)  # YOLO uses gray=114 for padding
    y_off = (h - new_h) // 2
    x_off = (w - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    new_bboxes = [
        (int(x1 * scale) + x_off, int(y1 * scale) + y_off,
         int(x2 * scale) + x_off, int(y2 * scale) + y_off)
        for (x1, y1, x2, y2) in bboxes
    ]
    return canvas, new_bboxes


def aug_translate(img, bboxes, hw):
    """Random translate (translate: 0.1) — shift by 10%"""
    h, w = hw
    tx, ty = int(w * 0.1), int(h * -0.08)
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted = cv2.warpAffine(img, M, (w, h), borderValue=(114, 114, 114))
    new_bboxes = [
        (max(0, x1 + tx), max(0, y1 + ty),
         min(w, x2 + tx), min(h, y2 + ty))
        for (x1, y1, x2, y2) in bboxes
    ]
    return shifted, new_bboxes


def aug_hsv_h(img, bboxes, hw):
    """HSV hue shift (hsv_h: 0.015)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + 0.015 * 180) % 180
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return result, bboxes


def aug_hsv_s(img, bboxes, hw):
    """HSV saturation (hsv_s: 0.5)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.4, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return result, bboxes


def aug_hsv_v(img, bboxes, hw):
    """HSV brightness (hsv_v: 0.3)"""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.7, 0, 255)  # darken
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    return result, bboxes


def aug_mosaic(imgs_bboxes, target_size):
    """4-image mosaic (mosaic: 0.5) — combine 4 images into one."""
    h, w = target_size
    canvas = np.full((h, w, 3), 114, dtype=np.uint8)
    h2, w2 = h // 2, w // 2
    all_bboxes = []

    for i, (img, bboxes) in enumerate(imgs_bboxes[:4]):
        resized = cv2.resize(img, (w2, h2))
        sy = float(h2) / img.shape[0]
        sx = float(w2) / img.shape[1]

        if i == 0:
            canvas[0:h2, 0:w2] = resized
            ox, oy = 0, 0
        elif i == 1:
            canvas[0:h2, w2:w] = resized
            ox, oy = w2, 0
        elif i == 2:
            canvas[h2:h, 0:w2] = resized
            ox, oy = 0, h2
        else:
            canvas[h2:h, w2:w] = resized
            ox, oy = w2, h2

        for (x1, y1, x2, y2) in bboxes:
            all_bboxes.append((
                int(x1 * sx) + ox, int(y1 * sy) + oy,
                int(x2 * sx) + ox, int(y2 * sy) + oy,
            ))

    return canvas, all_bboxes


# ── Build the figure ──

# Individual transforms (applied per-image)
transforms = [
    ("Horizontal\nFlip", aug_fliplr),
    ("Scale\n(zoom out)", aug_scale),
    ("Translate\n(shift)", aug_translate),
    ("HSV Hue\nShift", aug_hsv_h),
    ("HSV Saturation\nBoost", aug_hsv_s),
    ("HSV Brightness\n(darken)", aug_hsv_v),
]

n_rows = len(SAMPLES) + 1  # +1 for mosaic row
n_cols = 1 + len(transforms)  # original + transforms

fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 13))

# Row labels
row_labels = [s[1] for s in SAMPLES] + ["Mosaic\n(4 images)"]

# ── Per-image rows ──
for row_idx, (img_name, label) in enumerate(SAMPLES):
    img, bboxes, hw = load_image_and_bbox(img_name)

    # Column 0: Original
    ax = axes[row_idx, 0]
    ax.imshow(draw_bbox(img, bboxes, color=(0, 255, 0), thickness=3))
    ax.set_xticks([])
    ax.set_yticks([])
    if row_idx == 0:
        ax.set_title("Original", fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel(label, fontsize=10, fontweight="bold", rotation=0,
                  labelpad=70, va="center")

    # Columns 1-6: Each transform
    for col_idx, (t_name, t_func) in enumerate(transforms):
        ax = axes[row_idx, col_idx + 1]
        aug_img, aug_bboxes = t_func(img, bboxes, hw)
        ax.imshow(draw_bbox(aug_img, aug_bboxes, color=(255, 165, 0), thickness=3))
        ax.set_xticks([])
        ax.set_yticks([])
        if row_idx == 0:
            ax.set_title(t_name, fontsize=10, fontweight="bold", pad=8)

# ── Mosaic row ──
# Load 4 random train images for mosaic
mosaic_imgs = []
all_train = sorted(os.listdir(IMG_DIR))
chosen = random.sample(all_train[:200], 4)
for fname in chosen:
    mimg, mbboxes, mhw = load_image_and_bbox(fname)
    mosaic_imgs.append((mimg, mbboxes))

# Use the first image's size as target
target_h, target_w = mosaic_imgs[0][0].shape[:2]
mosaic_result, mosaic_bboxes = aug_mosaic(mosaic_imgs, (target_h, target_w))

# Show the 4 source images in first 4 columns
mosaic_row = len(SAMPLES)
for ci in range(n_cols):
    ax = axes[mosaic_row, ci]
    if ci == 0:
        # Show first source image
        ax.imshow(draw_bbox(mosaic_imgs[0][0], mosaic_imgs[0][1],
                            color=(0, 255, 0), thickness=3))
        ax.set_ylabel("Mosaic\n(4 images)", fontsize=10, fontweight="bold",
                      rotation=0, labelpad=70, va="center")
        ax.set_title("", fontsize=10)
    elif ci <= 3:
        # Show remaining source images
        ax.imshow(draw_bbox(mosaic_imgs[ci][0], mosaic_imgs[ci][1],
                            color=(0, 255, 0), thickness=3))
    elif ci == 4:
        # Arrow placeholder
        ax.text(0.5, 0.5, "→", fontsize=40, ha="center", va="center",
                transform=ax.transAxes, color="#2C3E50")
        ax.set_facecolor("#F5F5F5")
    elif ci == 5:
        # Mosaic result
        ax.imshow(draw_bbox(mosaic_result, mosaic_bboxes,
                            color=(255, 165, 0), thickness=3))
        ax.set_title("", fontsize=10)
        # Add "MOSAIC RESULT" label
        ax.text(0.5, -0.05, "Mosaic Result", fontsize=10, fontweight="bold",
                ha="center", va="top", transform=ax.transAxes, color="#E67E22")
    else:
        ax.axis("off")
        continue

    ax.set_xticks([])
    ax.set_yticks([])

# Add bottom label for mosaic row
axes[mosaic_row, 0].text(0.5, -0.08, "Source 1", fontsize=8, ha="center",
                          va="top", transform=axes[mosaic_row, 0].transAxes)
for ci in range(1, 4):
    axes[mosaic_row, ci].text(0.5, -0.08, f"Source {ci+1}", fontsize=8,
                               ha="center", va="top",
                               transform=axes[mosaic_row, ci].transAxes)

# Legend
fig.text(0.5, 0.01,
         "Green box = original annotation  |  Orange box = transformed annotation  |  "
         "Settings from configs/augmentation.yaml",
         ha="center", fontsize=10, style="italic", color="#555")

fig.suptitle("Augmentation Transforms Applied to MMFW-UAV Dataset Samples",
             fontsize=15, fontweight="bold", color="#1B2A4A", y=0.98)

fig.tight_layout(rect=[0.06, 0.03, 1, 0.95])
fig.savefig(os.path.join(OUT_DIR, "fig9_augmentation_examples.png"),
            dpi=DPI, bbox_inches="tight")
plt.close(fig)
print("Saved: report_figures/fig9_augmentation_examples.png")
