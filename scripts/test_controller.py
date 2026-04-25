"""Smoke test for src.sim.controller.

Verifies sign conventions and steady-state behavior:
  1. Centered target -> all surfaces ~0.
  2. Target right of center -> positive roll and yaw.
  3. Target below center -> negative pitch (nose down).
  4. Small target -> throttle above base; large target -> throttle below base.
  5. PID anti-windup: output saturates and stays saturated, doesn't grow.
  6. reset() — after reset, fresh small error gives small output (no leftover I).
  7. Negative dt raises ValueError.
  8. Slew limiter caps per-step output change.
  9. ControllerConfig.from_dict round-trip.

Run:
    python scripts/test_controller.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sim.controller import (
    ControllerConfig,
    PID,
    PIDGains,
    VisualServoController,
)


def bbox(cx: float, cy: float, w: float, h: float) -> tuple:
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def run_steady(controller: VisualServoController, target_bbox, W, H, n=120, dt=1 / 60):
    cmd = None
    for _ in range(n):
        cmd = controller.step(target_bbox, W, H, dt)
    return cmd


def test_centered() -> None:
    print("[test 1] centered target -> all surfaces ~0")
    W, H = 720, 680
    ctrl = VisualServoController()
    target = bbox(W / 2, H / 2, 50, 50)
    cmd = run_steady(ctrl, target, W, H)
    print(f"   roll={cmd.roll:+.3f} pitch={cmd.pitch:+.3f} yaw={cmd.yaw:+.3f}")
    assert abs(cmd.roll) < 0.05 and abs(cmd.pitch) < 0.05 and abs(cmd.yaw) < 0.05, \
        f"surfaces not near zero: {cmd}"
    print("   OK")


def test_target_right() -> None:
    print("[test 2] target at 80% of W (ex=+0.6) -> roll+yaw positive")
    W, H = 720, 680
    ctrl = VisualServoController()
    target = bbox(W * 0.8, H / 2, 50, 50)
    cmd = run_steady(ctrl, target, W, H, n=10)
    print(f"   ex={cmd.error_x_norm:+.3f} roll={cmd.roll:+.3f} yaw={cmd.yaw:+.3f}")
    assert cmd.error_x_norm > 0
    assert cmd.roll > 0 and cmd.yaw > 0
    print("   OK")


def test_target_below() -> None:
    print("[test 3] target below center -> pitch negative (nose down)")
    W, H = 720, 680
    ctrl = VisualServoController()
    target = bbox(W / 2, H * 0.85, 50, 50)
    cmd = run_steady(ctrl, target, W, H, n=10)
    print(f"   ey={cmd.error_y_norm:+.3f} pitch={cmd.pitch:+.3f}")
    assert cmd.error_y_norm > 0
    assert cmd.pitch < 0, "pitch should be negative when target is below center"
    print("   OK")


def test_throttle_size_feedback() -> None:
    print("[test 4] small target -> throttle up, big target -> throttle down")
    W, H = 720, 680
    base = ControllerConfig().throttle_base

    small = run_steady(VisualServoController(), bbox(W / 2, H / 2, 20, 20), W, H, n=30)
    big = run_steady(VisualServoController(), bbox(W / 2, H / 2, 170, 170), W, H, n=30)
    print(f"   small size={small.size_ratio:.3f} throttle={small.throttle:.3f}"
          f"  base={base}")
    print(f"   big   size={big.size_ratio:.3f} throttle={big.throttle:.3f}")
    assert small.throttle > base, "small target should increase throttle"
    assert big.throttle < base, "big target should decrease throttle"
    print("   OK")


def test_anti_windup_observable() -> None:
    """Sustained saturation should plateau, not keep growing."""
    print("[test 5] PID output saturates and stays saturated")
    pid = PID(PIDGains(kp=0.0, ki=10.0, integral_max=1.0, integral_min=-1.0,
                       out_max=1.0, out_min=-1.0))
    out_after_100s = None
    out_after_1s = None
    for i in range(10_000):
        out = pid.step(error=1.0, dt=0.01)
        if i == 100:
            out_after_1s = out
        if i == 9_999:
            out_after_100s = out
    print(f"   out @ 1s={out_after_1s:.4f}  @ 100s={out_after_100s:.4f}")
    assert out_after_1s == 1.0, "output should already be saturated by 1s"
    assert out_after_100s == 1.0, "output should still be 1.0 — no runaway"
    print("   OK")


def test_reset_observable() -> None:
    """After reset, a small fresh error must produce a small output."""
    print("[test 6] reset clears integrator (observable via output)")
    pid = PID(PIDGains(kp=0.1, ki=2.0, integral_max=1.0, integral_min=-1.0))
    # Drive sustained large error to load the integrator
    for _ in range(500):
        pid.step(error=1.0, dt=0.01)
    pre_reset = pid.step(error=0.0, dt=0.01)  # zero-error reading with loaded I
    pid.reset()
    post_reset = pid.step(error=0.0, dt=0.01)
    print(f"   zero-error output before reset={pre_reset:.4f} after reset={post_reset:.4f}")
    assert pre_reset > 0.5, "integrator should produce non-trivial output pre-reset"
    assert abs(post_reset) < 1e-6, "after reset, zero error must give zero output"
    print("   OK")


def test_negative_dt_raises() -> None:
    print("[test 7] negative dt raises ValueError")
    pid = PID(PIDGains(kp=1.0))
    pid.step(error=0.5, dt=0.01)
    try:
        pid.step(error=0.5, dt=-0.01)
    except ValueError as e:
        print(f"   raised: {e}")
    else:
        raise AssertionError("expected ValueError on negative dt")
    print("   OK")


def test_slew_limiter() -> None:
    """A step input should ramp at <= slew_limit_per_s."""
    print("[test 8] slew limiter caps per-step output change")
    pid = PID(PIDGains(kp=10.0, slew_limit_per_s=4.0))  # 4 unit/s, dt=0.01 -> 0.04/step
    outs = []
    for _ in range(50):
        outs.append(pid.step(error=1.0, dt=0.01))
    # First step from 0 should be capped at 0.04
    assert abs(outs[0] - 0.04) < 1e-9, f"first step should be 0.04, got {outs[0]}"
    # Step deltas should never exceed 0.04
    deltas = [outs[i] - outs[i-1] for i in range(1, len(outs))]
    assert max(abs(d) for d in deltas) <= 0.04 + 1e-9, "slew limit violated mid-run"
    # Eventually plateau at output cap (1.0)
    assert outs[-1] == 1.0, f"should reach saturation, got {outs[-1]}"
    print(f"   first step={outs[0]:.4f}  max delta={max(abs(d) for d in deltas):.4f}"
          f"  final={outs[-1]:.4f}")
    print("   OK")


def test_from_dict() -> None:
    print("[test 9] ControllerConfig.from_dict")
    d = {
        "roll": {"kp": 1.5, "ki": 0.2},
        "pitch": {"kp": 1.0},
        "throttle_base": 0.6,
    }
    cfg = ControllerConfig.from_dict(d)
    assert cfg.roll.kp == 1.5 and cfg.roll.ki == 0.2
    assert cfg.pitch.kp == 1.0
    assert cfg.throttle_base == 0.6
    # Missing axes fall back to defaults
    assert cfg.yaw.kp == ControllerConfig().yaw.kp
    print(f"   roll.kp={cfg.roll.kp}  pitch.kp={cfg.pitch.kp}"
          f"  yaw.kp={cfg.yaw.kp} (default)  throttle_base={cfg.throttle_base}")
    print("   OK")


if __name__ == "__main__":
    test_centered()
    test_target_right()
    test_target_below()
    test_throttle_size_feedback()
    test_anti_windup_observable()
    test_reset_observable()
    test_negative_dt_raises()
    test_slew_limiter()
    test_from_dict()
    print("\nAll tests passed.")
