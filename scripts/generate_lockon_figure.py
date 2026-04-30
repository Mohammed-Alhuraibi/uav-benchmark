#!/usr/bin/env python3
"""Lock-on visualization — clean version.

Shows only:
  - Detection bounding box (cyan)
  - White line from frame center to target center
  - Blue normalized deviation vector (-1 to 1) next to the line
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import cv2
import numpy as np
from ultralytics import YOLO

DPI = 300

model = YOLO("reports(1)/runs/yolo11s_p2/weights/best.pt")

SAMPLES = [
    ("001464.jpg", "Lock-on Example 1"),
    ("001392.jpg", "Lock-on Example 2"),
    ("002427.jpg", "Lock-on Example 3"),
]


def draw_lockon(ax, img_name):
    img_path = f"data/dataset/images/train/{img_name}"
    results = model.predict(img_path, imgsz=640, verbose=False, conf=0.35)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W = img.shape[:2]

    box = results[0].boxes
    xyxy = box.xyxy[0].cpu().numpy()
    det_x1, det_y1, det_x2, det_y2 = xyxy
    det_cx = (det_x1 + det_x2) / 2
    det_cy = (det_y1 + det_y2) / 2
    det_w = det_x2 - det_x1
    det_h = det_y2 - det_y1

    frame_cx = W / 2
    frame_cy = H / 2

    # Normalized deviation: -1 to 1
    vx = (det_cx - frame_cx) / (W / 2)     # right = +, left = -
    vy = -(det_cy - frame_cy) / (H / 2)    # up = +, down = -

    ax.imshow(img)

    # AV: Target Hit Area (yellow dashed) — 25% side margins, 10% top/bottom
    av_x = W * 0.25
    av_y = H * 0.10
    av_w = W * 0.50
    av_h = H * 0.80
    ax.add_patch(Rectangle((av_x, av_y), av_w, av_h, fill=False,
                 edgecolor="#F1C40F", linewidth=2, linestyle="-",
                 alpha=0.7, zorder=10))

    # Detection bbox (cyan)
    ax.add_patch(Rectangle((det_x1, det_y1), det_w, det_h, fill=False,
                 edgecolor="#E74C3C", linewidth=2.5, zorder=12))

    # Frame center crosshair (small, subtle)
    cs = min(W, H) * 0.012
    ax.plot([frame_cx - cs, frame_cx + cs], [frame_cy, frame_cy],
            color="white", lw=1.5, zorder=15)
    ax.plot([frame_cx, frame_cx], [frame_cy - cs, frame_cy + cs],
            color="white", lw=1.5, zorder=15)
    ax.plot(frame_cx, frame_cy, "o", color="white", ms=4, zorder=16)

    # Target center dot
    ax.plot(det_cx, det_cy, "o", color="#E74C3C", ms=5, zorder=15)
    ax.plot(det_cx, det_cy, "o", color="white", ms=2.5, zorder=16)

    # White line: center → target
    ax.plot([frame_cx, det_cx], [frame_cy, det_cy],
            color="white", lw=2, zorder=13, alpha=0.9)

    # Blue vector label next to the line midpoint
    mid_x = (frame_cx + det_cx) / 2
    mid_y = (frame_cy + det_cy) / 2

    # Offset the label perpendicular to the line so it doesn't sit on top
    dx = det_cx - frame_cx
    dy = det_cy - frame_cy
    length = max((dx**2 + dy**2)**0.5, 1)
    # Perpendicular direction (rotated 90°)
    perp_x = -dy / length * W * 0.03
    perp_y = dx / length * H * 0.03

    ax.text(mid_x + perp_x, mid_y + perp_y,
            f"({vx:.2f}, {vy:.2f})",
            ha="center", va="center",
            fontsize=11, fontweight="bold", color="#4FC3F7", zorder=20,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black",
                      alpha=0.6, edgecolor="none"))

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")


for i, (img_name, subtitle) in enumerate(SAMPLES):
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    print(f"Processing {img_name}...")
    draw_lockon(ax, img_name)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    out = f"report_figures/fig11_lockon_geometry_{i+1}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved: {out}")
