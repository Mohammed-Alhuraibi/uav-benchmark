"""Visual servo controller.

Converts a target bbox in the camera frame into roll / pitch / yaw / throttle
commands. The pipeline feeds it the Kalman-smoothed bbox (not the raw YOLO
bbox) so commanded surfaces don't jitter on detection noise.

Conventions:
  - Image origin top-left, x right, y down.
  - error_x = target_cx - W/2  (positive = target right of center)
  - error_y = target_cy - H/2  (positive = target below center)

  - roll  ∈ [-1, +1], positive = bank right.
  - pitch ∈ [-1, +1], positive = nose up.
  - yaw   ∈ [-1, +1], positive = nose right.
  - throttle ∈ [0, 1], positive feedback when target appears too small.

Note: this controller does *not* compute the spec-validity envelope
(AH center vs target center, ≤ ½ target dim per axis). That's a separate
concern handled by the pipeline before deciding to send a lock packet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PIDGains:
    kp: float
    ki: float = 0.0
    kd: float = 0.0
    out_min: float = -1.0
    out_max: float = 1.0
    integral_min: float = -1.0
    integral_max: float = 1.0
    # Optional first-order rate limiter on the output. None = unlimited.
    # Units: output-units per second. e.g. 6.0 means the output can swing
    # at most 6.0 between a step of dt=1.0s, so on a 60fps loop the per-frame
    # change is capped at 6.0/60 = 0.1.
    slew_limit_per_s: float | None = None


class PID:
    """Single-axis PID with integral clamping and optional slew limiting."""

    def __init__(self, gains: PIDGains):
        self.gains = gains
        self._integral = 0.0
        self._prev_error: float | None = None
        self._prev_output = 0.0  # for slew limiting

    def reset(self) -> None:
        self._integral = 0.0
        self._prev_error = None
        self._prev_output = 0.0

    def step(self, error: float, dt: float) -> float:
        if dt < 0:
            raise ValueError(
                f"PID dt={dt} negative — caller bug; timestamps must be "
                "monotonically non-decreasing."
            )

        if dt > 0:
            # Anti-windup: clamp the integral term itself, not the final output.
            # Stops the I-term from continuing to grow during long saturation.
            self._integral += error * dt
            self._integral = max(
                self.gains.integral_min,
                min(self.gains.integral_max, self._integral),
            )

            if self._prev_error is None:
                derivative = 0.0
            else:
                derivative = (error - self._prev_error) / dt
            self._prev_error = error
        else:
            # dt == 0 — no time advance. Reuse the last integrator and skip
            # the derivative term (undefined over zero time).
            derivative = 0.0

        out = (
            self.gains.kp * error
            + self.gains.ki * self._integral
            + self.gains.kd * derivative
        )
        out = max(self.gains.out_min, min(self.gains.out_max, out))

        # Slew (rate) limiter on the OUTPUT — separate from output saturation.
        # Disabled when dt == 0 so a tester doing back-to-back same-time calls
        # doesn't get stuck on the slew bound forever.
        if self.gains.slew_limit_per_s is not None and dt > 0:
            max_step = self.gains.slew_limit_per_s * dt
            lo = self._prev_output - max_step
            hi = self._prev_output + max_step
            out = max(lo, min(hi, out))

        self._prev_output = out
        return out


@dataclass
class ControllerConfig:
    """Per-axis gains and target-size setpoint.

    Defaults assume normalized errors in [-1, +1] (i.e., divided by half-frame).
    Kp ≈ 1.0 means "100% of error → full deflection." Tuned conservatively;
    bump kp up if commanded surfaces look sluggish in the demo.
    """
    roll: PIDGains = field(default_factory=lambda: PIDGains(kp=0.8, ki=0.05, kd=0.10))
    pitch: PIDGains = field(default_factory=lambda: PIDGains(kp=0.7, ki=0.05, kd=0.10))
    yaw: PIDGains = field(default_factory=lambda: PIDGains(kp=0.3, ki=0.02, kd=0.05))
    throttle: PIDGains = field(default_factory=lambda: PIDGains(
        kp=2.0, ki=0.10, kd=0.0,
        out_min=-0.5, out_max=0.5,           # commanded delta around base
        integral_min=-0.5, integral_max=0.5,
    ))
    target_size_ratio: float = 0.10   # desired max-axis ratio of target / frame
    throttle_base: float = 0.55       # nominal cruise

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ControllerConfig":
        """Build from a parsed YAML dict. Missing axes fall back to defaults."""
        defaults = cls()
        return cls(
            roll=PIDGains(**d["roll"]) if "roll" in d else defaults.roll,
            pitch=PIDGains(**d["pitch"]) if "pitch" in d else defaults.pitch,
            yaw=PIDGains(**d["yaw"]) if "yaw" in d else defaults.yaw,
            throttle=PIDGains(**d["throttle"]) if "throttle" in d else defaults.throttle,
            target_size_ratio=d.get("target_size_ratio", defaults.target_size_ratio),
            throttle_base=d.get("throttle_base", defaults.throttle_base),
        )


@dataclass
class ControlCommand:
    """One frame's commanded surfaces + the underlying error."""
    roll: float
    pitch: float
    yaw: float
    throttle: float
    error_x_norm: float        # in [-1, +1], frame-half units
    error_y_norm: float        # in [-1, +1], frame-half units
    size_ratio: float          # max(target_w/W, target_h/H) — same metric as min_bbox_ratio


class VisualServoController:
    """Three-axis attitude + throttle controller for nose-camera centering."""

    def __init__(self, config: ControllerConfig | None = None):
        self.cfg = config or ControllerConfig()
        self._roll = PID(self.cfg.roll)
        self._pitch = PID(self.cfg.pitch)
        self._yaw = PID(self.cfg.yaw)
        self._throttle = PID(self.cfg.throttle)

    def reset(self) -> None:
        """Clear all integrators — call when lock is lost or target switched."""
        self._roll.reset()
        self._pitch.reset()
        self._yaw.reset()
        self._throttle.reset()

    def step(
        self,
        target_bbox: tuple[float, float, float, float],
        frame_w: int,
        frame_h: int,
        dt: float,
    ) -> ControlCommand:
        """Compute a control command for the given (smoothed) target bbox.

        bbox is (x1, y1, x2, y2) in pixel coordinates of the original frame.
        """
        x1, y1, x2, y2 = target_bbox
        target_cx = (x1 + x2) / 2
        target_cy = (y1 + y2) / 2
        target_w = x2 - x1
        target_h = y2 - y1

        # Normalize errors to [-1, +1] using half-frame as the unit.
        ex = (target_cx - frame_w / 2) / (frame_w / 2)
        ey = (target_cy - frame_h / 2) / (frame_h / 2)

        # Aircraft control:
        #   error right (ex>0) -> roll right (positive) and yaw right (positive)
        #   error down  (ey>0, image-y down) -> nose down -> pitch negative
        roll_cmd = self._roll.step(ex, dt)
        yaw_cmd = self._yaw.step(ex, dt)
        pitch_cmd = self._pitch.step(-ey, dt)

        # Throttle: same metric as competition's min_bbox_ratio (max axis ratio).
        # If actual size < setpoint we want to close in (positive throttle delta).
        size_ratio = max(target_w / frame_w, target_h / frame_h)
        size_error = self.cfg.target_size_ratio - size_ratio
        throttle_delta = self._throttle.step(size_error, dt)
        # Final clamp to physical [0, 1] range — separate from the PID's
        # out_min/out_max, which bound the *delta* relative to throttle_base.
        throttle_cmd = max(0.0, min(1.0, self.cfg.throttle_base + throttle_delta))

        return ControlCommand(
            roll=roll_cmd,
            pitch=pitch_cmd,
            yaw=yaw_cmd,
            throttle=throttle_cmd,
            error_x_norm=ex,
            error_y_norm=ey,
            size_ratio=size_ratio,
        )
