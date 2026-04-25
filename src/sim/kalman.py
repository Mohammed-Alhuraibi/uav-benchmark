"""Constant-velocity Kalman filter for bbox tracking.

Drives the smoothed Lockdown Quadrilateral (AH) rectangle. The raw YOLO
bbox is the *measurement*; this filter produces the *predicted* bbox that
gets drawn on screen. During detection dropouts the filter keeps
predicting forward, so the AH rectangle continues to slide on motion
inertia until the target reappears.

State (6D): [cx, cy, w, h, vx, vy]
Observation (4D): [cx, cy, w, h]

Width and height are modelled as quasi-static (no velocity) since on a
nose-cam view they change slowly with range. Center has full velocity so
the filter can extrapolate during gaps.

Auto-recovery on long dropouts is the *caller's* responsibility — when
the lock-on state machine resets after exceeding its 200 ms tolerance,
it must also call kf.reset(). This filter has no maximum-age policy.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanConfig:
    """Tunable noise parameters. All in pixel units (or px/s for velocity)."""
    q_pos: float = 1.0      # process noise on cx, cy (per sqrt(s))
    q_size: float = 0.5     # process noise on w, h (per sqrt(s))
    q_vel: float = 20.0     # process noise on vx, vy (per sqrt(s))
    r_pos: float = 2.0      # measurement noise on cx, cy (px)
    r_size: float = 4.0     # measurement noise on w, h (px)
    init_pos_var: float = 25.0     # initial covariance for cx, cy
    init_size_var: float = 100.0   # initial covariance for w, h
    init_vel_var: float = 1e4      # initial covariance for vx, vy (large — unknown)
    min_size_px: float = 1.0       # clamp for w, h after update (avoids degenerate boxes)


class ConstantVelocityKalman:
    """Bbox state tracker with measurement and dropout-prediction modes.

    Typical loop per frame:

        if measurement_available:
            kf.update(bbox, t_now)        # predicts internally, then incorporates
        else:
            kf.predict_only(t_now)        # advance state without measurement

        smoothed_bbox = kf.bbox           # what to draw as AH

    The first update() initializes the filter — no need to seed manually.
    """

    _STATE_DIM = 6
    _OBS_DIM = 4

    def __init__(self, config: KalmanConfig | None = None):
        self.cfg = config or KalmanConfig()
        self._x: np.ndarray | None = None    # state vector (6,)
        self._P: np.ndarray | None = None    # covariance (6, 6)
        self._t: float | None = None         # timestamp of last advance
        self._H = np.zeros((self._OBS_DIM, self._STATE_DIM))
        self._H[0, 0] = 1.0  # observe cx
        self._H[1, 1] = 1.0  # observe cy
        self._H[2, 2] = 1.0  # observe w
        self._H[3, 3] = 1.0  # observe h
        self._R = np.diag([
            self.cfg.r_pos ** 2,
            self.cfg.r_pos ** 2,
            self.cfg.r_size ** 2,
            self.cfg.r_size ** 2,
        ])

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def initialized(self) -> bool:
        return self._x is not None

    @property
    def bbox(self) -> tuple[float, float, float, float] | None:
        """Smoothed bbox as (x1, y1, x2, y2). None until first update."""
        if not self.initialized:
            return None
        cx, cy, w, h = self._x[0], self._x[1], self._x[2], self._x[3]
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    @property
    def velocity(self) -> tuple[float, float] | None:
        """Estimated (vx, vy) in px/s, or None until initialized."""
        if not self.initialized:
            return None
        return (float(self._x[4]), float(self._x[5]))

    def reset(self) -> None:
        """Drop all state — call after lock-on resets."""
        self._x = None
        self._P = None
        self._t = None

    def update(self, bbox_xyxy: tuple[float, float, float, float], t: float) -> None:
        """Predict to time t, then incorporate the measurement bbox."""
        z = self._bbox_to_obs(bbox_xyxy)

        if not self.initialized:
            self._initialize(z, t)
            return

        self._advance_to(t)

        # Joseph-form would be P = (I-KH) P (I-KH)^T + K R K^T. The
        # simplified form below is fine here because P stays well-conditioned
        # at our Q/R magnitudes — but it relies on K being optimal; if you
        # ever change H or skip updates, switch to Joseph.
        y = z - self._H @ self._x
        S = self._H @ self._P @ self._H.T + self._R
        K = np.linalg.solve(S.T, (self._P @ self._H.T).T).T  # P H^T S^-1
        self._x = self._x + K @ y
        I = np.eye(self._STATE_DIM)
        self._P = (I - K @ self._H) @ self._P

        # Defensive: width/height must stay strictly positive even after
        # a noisy update.
        self._x[2] = max(self._x[2], self.cfg.min_size_px)
        self._x[3] = max(self._x[3], self.cfg.min_size_px)

    def predict_only(self, t: float) -> None:
        """Advance state to time t with no measurement (dropout frame).

        No-op if uninitialized — without a starting bbox we have nothing
        to predict from.
        """
        if not self.initialized:
            return
        self._advance_to(t)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _advance_to(self, t: float) -> None:
        dt = t - self._t
        if dt < 0:
            raise ValueError(
                f"Kalman timestamp went backwards: {t} < {self._t}. "
                "Caller bug — timestamps must be monotonically non-decreasing."
            )
        if dt == 0.0:
            return

        F = self._transition_matrix(dt)
        Q = self._process_noise(dt)

        self._x = F @ self._x
        self._P = F @ self._P @ F.T + Q
        self._t = t

    def _initialize(self, z: np.ndarray, t: float) -> None:
        self._x = np.zeros(self._STATE_DIM)
        self._x[:4] = z  # cx, cy, w, h; vx, vy left at 0
        self._P = np.diag([
            self.cfg.init_pos_var,
            self.cfg.init_pos_var,
            self.cfg.init_size_var,
            self.cfg.init_size_var,
            self.cfg.init_vel_var,
            self.cfg.init_vel_var,
        ])
        self._t = t

    @classmethod
    def _bbox_to_obs(cls, bbox_xyxy: tuple[float, float, float, float]) -> np.ndarray:
        x1, y1, x2, y2 = bbox_xyxy
        return np.array([
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            x2 - x1,
            y2 - y1,
        ])

    @classmethod
    def _transition_matrix(cls, dt: float) -> np.ndarray:
        F = np.eye(cls._STATE_DIM)
        F[0, 4] = dt  # cx += vx*dt
        F[1, 5] = dt  # cy += vy*dt
        return F

    def _process_noise(self, dt: float) -> np.ndarray:
        # Diagonal Q scaled by dt — variance grows linearly with elapsed time.
        return np.diag([
            (self.cfg.q_pos ** 2) * dt,
            (self.cfg.q_pos ** 2) * dt,
            (self.cfg.q_size ** 2) * dt,
            (self.cfg.q_size ** 2) * dt,
            (self.cfg.q_vel ** 2) * dt,
            (self.cfg.q_vel ** 2) * dt,
        ])
