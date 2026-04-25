"""Smoke test for src.sim.kalman.

Scenarios:
  1. Static target — Kalman estimate should beat raw noise.
  2. Constant-velocity target — filter learns velocity, predicts accurately
     across a 0.5s dropout (predict_only mode).
  3. Reset clears state.
  4. Negative-dt input raises ValueError.
  5. Update before any predict_only works (auto-init records timestamp).

Run:
    python scripts/test_kalman.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.sim.kalman import ConstantVelocityKalman


def bbox(cx: float, cy: float, w: float = 50, h: float = 50) -> tuple:
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def center(b: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def test_static() -> None:
    print("[test 1] static target — filter vs raw noise")
    kf = ConstantVelocityKalman()
    rng = np.random.default_rng(0)
    t = 0.0
    dt = 1 / 60
    true_cx, true_cy = 320.0, 240.0
    sigma = 1.5
    raw_errs, kf_errs = [], []
    for i in range(180):  # 3 s at 60 fps
        t += dt
        m_cx = true_cx + rng.normal(0, sigma)
        m_cy = true_cy + rng.normal(0, sigma)
        kf.update(bbox(m_cx, m_cy), t)
        if i > 30:  # discard burn-in
            raw_errs.append(np.hypot(m_cx - true_cx, m_cy - true_cy))
            cx, cy = center(kf.bbox)
            kf_errs.append(np.hypot(cx - true_cx, cy - true_cy))
    raw_rmse = np.sqrt(np.mean(np.square(raw_errs)))
    kf_rmse = np.sqrt(np.mean(np.square(kf_errs)))
    print(f"   raw RMSE={raw_rmse:.2f}px  kalman RMSE={kf_rmse:.2f}px")
    assert kf_rmse < raw_rmse, "Kalman should reduce noise vs raw measurements"
    print("   OK")


def test_constant_velocity_with_dropout() -> None:
    print("[test 2] constant-velocity target with 0.5s dropout")
    kf = ConstantVelocityKalman()
    rng = np.random.default_rng(1)
    dt = 1 / 60
    t = 0.0
    cx, cy = 200.0, 240.0
    vx, vy = 80.0, 30.0  # px/s

    # 1 s of measurements to learn velocity
    for _ in range(60):
        t += dt
        cx += vx * dt
        cy += vy * dt
        kf.update(bbox(cx + rng.normal(0, 1.5), cy + rng.normal(0, 1.5)), t)

    est_vx, est_vy = kf.velocity
    print(f"   learned velocity=({est_vx:.1f},{est_vy:.1f}) true=({vx},{vy})")
    assert abs(est_vx - vx) < 8 and abs(est_vy - vy) < 8, \
        f"velocity off: estimated ({est_vx},{est_vy}) vs true ({vx},{vy})"

    # 0.5 s dropout — predict only
    for _ in range(30):
        t += dt
        cx += vx * dt
        cy += vy * dt
        kf.predict_only(t)

    pred_cx, pred_cy = center(kf.bbox)
    err = np.hypot(pred_cx - cx, pred_cy - cy)
    print(f"   after 0.5s dropout: predicted=({pred_cx:.1f},{pred_cy:.1f})"
          f" true=({cx:.1f},{cy:.1f}) err={err:.2f}px")
    assert err < 5.0, f"dropout prediction error too large: {err}"
    print("   OK")


def test_reset() -> None:
    print("[test 3] reset clears state")
    kf = ConstantVelocityKalman()
    kf.update(bbox(100, 100), 0.0)
    assert kf.initialized
    kf.reset()
    assert not kf.initialized
    assert kf.bbox is None
    assert kf.velocity is None
    print("   OK")


def test_negative_dt_raises() -> None:
    print("[test 4] negative dt raises ValueError")
    kf = ConstantVelocityKalman()
    kf.update(bbox(100, 100), 1.0)
    try:
        kf.update(bbox(101, 100), 0.5)  # going backwards
    except ValueError as e:
        print(f"   raised correctly: {e}")
    else:
        raise AssertionError("expected ValueError on backwards timestamp")
    print("   OK")


def test_update_before_predict_only() -> None:
    """The auto-init in update() must record _t so later predict_only works."""
    print("[test 5] update() then predict_only() works (auto-init records time)")
    kf = ConstantVelocityKalman()
    kf.update(bbox(100, 100), 0.0)
    kf.predict_only(0.1)  # would crash if _t wasn't set during init
    assert kf.bbox is not None
    print("   OK")


if __name__ == "__main__":
    test_static()
    test_constant_velocity_with_dropout()
    test_reset()
    test_negative_dt_raises()
    test_update_before_predict_only()
    print("\nAll tests passed.")
