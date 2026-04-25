"""Per-frame simulation state — the schema that flows through the pipeline.

Lives in its own module (rather than visualizer.py) so headless consumers
(figure generation, log analysis) can import it without pulling in OpenCV.

The pipeline composes a HUDState each frame from InferencePipeline,
LockOnStateMachine, ConstantVelocityKalman, and VisualServoController
outputs. Both HUDRenderer and FrameLogger read it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.sim.controller import ControlCommand


@dataclass
class HUDState:
    """All inputs needed for a single frame's render + log."""
    # Time / frame identity
    timestamp_s: float            # wall-clock-style time in seconds (for server time field)
    frame_idx: int
    fps: float
    inference_ms: float

    # Detection + tracking
    detections: list[dict] = field(default_factory=list)  # raw output of InferencePipeline.predict()
    kalman_bbox: tuple[float, float, float, float] | None = None

    # Lock-on state machine output (dict from LockOnStateMachine.update())
    lock_state: dict[str, Any] = field(default_factory=dict)

    # Controller output — None when no target locked
    control: ControlCommand | None = None

    # Spec envelope: (err_x_px, err_y_px) and (half_target_w_px, half_target_h_px).
    # in_envelope = |err_x| <= half_w AND |err_y| <= half_h.
    spec_error_px: tuple[float, float] | None = None
    spec_envelope_half_px: tuple[float, float] | None = None
    in_envelope: bool = False

    # Zone definitions (fractions of frame W/H). Defaults match deployment.yaml.
    ah_w_frac: float = 0.35
    ah_h_frac: float = 0.50
    av_w_frac: float = 0.50      # 1 - 2*0.25
    av_h_frac: float = 0.80      # 1 - 2*0.10

    # Cosmetic
    model_name: str = "yolo11s_p2"


def phase_from_lock_state(lock_state: dict[str, Any]) -> str:
    """Derive a human-readable phase label from the lock-state-machine dict.

    Single source of truth — both the visualizer and the logger derive
    phase names from this so they can never drift apart.
    """
    if lock_state.get("locked"):
        return "LOCKED"
    if lock_state.get("dropout_s", 0.0) > 0:
        return "DROPOUT"
    if lock_state.get("progress", 0.0) > 0:
        return "TRACKING"
    return "SEARCHING"
