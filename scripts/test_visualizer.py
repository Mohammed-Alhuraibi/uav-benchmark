"""Smoke test for src.sim.visualizer.HUDRenderer.

Renders four canonical lock-on phases against a frame extracted from the
P-51 simulation clip. Dumps PNGs to /tmp/hud_*.png for visual inspection.

Run:
    python scripts/test_visualizer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from src.sim.controller import ControlCommand
from src.sim.state import HUDState
from src.sim.visualizer import HUDLayout, HUDRenderer

ROOT = Path(__file__).resolve().parent.parent
CLIP = ROOT / "data" / "sim" / "clips" / "p51_dogfight.mp4"
OUT = Path("/tmp")


def grab_frame(t_seconds: float = 5.0) -> np.ndarray:
    """Grab a single frame from the simulation clip — falls back to a
    synthetic gradient if the clip isn't available."""
    if CLIP.exists():
        cap = cv2.VideoCapture(str(CLIP))
        cap.set(cv2.CAP_PROP_POS_MSEC, t_seconds * 1000)
        ok, frame = cap.read()
        cap.release()
        if ok:
            return frame
    # fallback gradient
    h, w = 680, 720
    grad = np.linspace(60, 200, w, dtype=np.uint8)
    return np.stack([np.tile(grad, (h, 1))] * 3, axis=-1).astype(np.uint8)


def make_detection(cx, cy, w, h, conf, lockable, tid=1, meets_size=True):
    return {
        "bbox": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
        "confidence": conf,
        "bbox_w_ratio": w / 720,
        "bbox_h_ratio": h / 680,
        "meets_size": meets_size,
        "lockable": lockable,
        "track_id": tid,
    }


def make_state(phase: str, frame_w=720, frame_h=680) -> HUDState:
    """One canonical state per lock-on phase."""
    cx, cy, w, h = 0.55 * frame_w, 0.50 * frame_h, 60, 32

    if phase == "searching":
        return HUDState(
            timestamp_s=12 * 3600 + 5 * 60 + 23.456,
            frame_idx=42, fps=58.7, inference_ms=14.3,
            detections=[],
            lock_state={"locked": False, "progress": 0.0, "elapsed_s": 0.0,
                        "dropout_s": 0.0, "target_id": None, "in_zone": False},
        )

    if phase == "tracking":
        det = make_detection(cx, cy, w, h, conf=0.86, lockable=True, tid=1)
        # Kalman bbox slightly offset from target — produces non-trivial spec error
        kbbox = (cx - w / 2 - 4, cy - h / 2 + 2, cx + w / 2 - 4, cy + h / 2 + 2)
        ctrl = ControlCommand(
            roll=0.18, pitch=-0.05, yaw=0.07, throttle=0.62,
            error_x_norm=0.12, error_y_norm=0.0, size_ratio=0.083,
        )
        return HUDState(
            timestamp_s=12 * 3600 + 5 * 60 + 24.137,
            frame_idx=80, fps=57.2, inference_ms=15.1,
            detections=[det],
            kalman_bbox=kbbox,
            lock_state={"locked": False, "progress": 0.42, "elapsed_s": 1.68,
                        "dropout_s": 0.0, "target_id": 1, "in_zone": True},
            control=ctrl,
            spec_error_px=(4.0, -2.0),
            spec_envelope_half_px=(w / 2, h / 2),
            in_envelope=True,
        )

    if phase == "dropout":
        kbbox = (cx - w / 2 + 8, cy - h / 2 - 1, cx + w / 2 + 8, cy + h / 2 - 1)
        ctrl = ControlCommand(
            roll=0.21, pitch=-0.06, yaw=0.08, throttle=0.64,
            error_x_norm=0.16, error_y_norm=-0.01, size_ratio=0.083,
        )
        return HUDState(
            timestamp_s=12 * 3600 + 5 * 60 + 25.213,
            frame_idx=100, fps=56.4, inference_ms=14.9,
            detections=[],
            kalman_bbox=kbbox,
            lock_state={"locked": False, "progress": 0.65, "elapsed_s": 2.60,
                        "dropout_s": 0.087, "target_id": 1, "in_zone": False},
            control=ctrl,
            spec_error_px=(8.0, 1.0),
            spec_envelope_half_px=(w / 2, h / 2),
            in_envelope=True,
        )

    if phase == "locked":
        det = make_detection(cx, cy, w, h, conf=0.93, lockable=True, tid=1)
        kbbox = (cx - w / 2 - 1, cy - h / 2 + 1, cx + w / 2 - 1, cy + h / 2 + 1)
        ctrl = ControlCommand(
            roll=0.04, pitch=-0.01, yaw=0.02, throttle=0.58,
            error_x_norm=0.02, error_y_norm=0.0, size_ratio=0.083,
        )
        return HUDState(
            timestamp_s=12 * 3600 + 5 * 60 + 27.000,
            frame_idx=180, fps=56.0, inference_ms=14.7,
            detections=[det],
            kalman_bbox=kbbox,
            lock_state={"locked": True, "progress": 1.0, "elapsed_s": 4.05,
                        "dropout_s": 0.0, "target_id": 1, "in_zone": True},
            control=ctrl,
            spec_error_px=(1.0, -1.0),
            spec_envelope_half_px=(w / 2, h / 2),
            in_envelope=True,
        )

    raise ValueError(f"unknown phase: {phase}")


def main() -> int:
    frame = grab_frame()
    fh, fw = frame.shape[:2]
    print(f"source frame: {fw}x{fh}")
    renderer = HUDRenderer(fw, fh)
    expected_size = renderer.output_size  # (W, H)
    print(f"expected composite: {expected_size[0]}x{expected_size[1]}")

    for phase in ["searching", "tracking", "dropout", "locked"]:
        state = make_state(phase, fw, fh)
        composite = renderer.render(frame, state)

        assert composite.shape == (fh, expected_size[0], 3), \
            f"composite shape wrong for {phase}: {composite.shape}"

        out = OUT / f"hud_{phase}.png"
        cv2.imwrite(str(out), composite)
        print(f"   {phase:9s} -> {out} ({composite.shape[1]}x{composite.shape[0]})")

    print("\nAll renders succeeded. Inspect /tmp/hud_*.png visually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
