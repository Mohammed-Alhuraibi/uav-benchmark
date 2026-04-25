"""Frame & telemetry logger for the simulation.

Two outputs from one logger:
  - CSV: one row per frame, fixed schema (FRAME_COLUMNS). Drop into pandas
    for the error-path and lock-timeline figures.
  - JSONL: multi-channel event stream for terminal piping. Channels:
        meta   1× at session start (config snapshot)
        frame  1× per frame (summary)
        det    1× per detection (only on frames with detections)
        ctrl   1× per frame when controller has output
        lock   only on phase transitions (SEARCHING/TRACKING/DROPOUT/LOCKED)

The CSV is the source of truth for figures; the JSONL is for live observation.
Either output can be disabled by passing None for its path.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, IO

from src.sim.state import HUDState, phase_from_lock_state


# Single source of truth for the CSV schema. Order is preserved.
FRAME_COLUMNS: tuple[str, ...] = (
    "frame_idx", "t_s", "fps", "inference_ms",
    "n_det", "n_lockable", "target_id",
    "target_x1", "target_y1", "target_x2", "target_y2", "target_conf",
    "kalman_x1", "kalman_y1", "kalman_x2", "kalman_y2",
    "locked", "progress", "elapsed_s", "dropout_s", "in_zone", "phase",
    "err_x_px", "err_y_px", "env_half_w_px", "env_half_h_px", "in_envelope",
    "roll", "pitch", "yaw", "throttle",
    "error_x_norm", "error_y_norm", "size_ratio",
)

DEFAULT_CHANNELS: frozenset[str] = frozenset({"meta", "frame", "det", "ctrl", "lock"})


def _best_lockable(dets: list[dict]) -> dict | None:
    """Highest-confidence lockable detection, or None."""
    lockable = [d for d in dets if d.get("lockable")]
    if not lockable:
        return None
    return max(lockable, key=lambda d: d.get("confidence", 0))


@dataclass
class LoggerConfig:
    csv_path: Path | None = None
    jsonl_path: Path | None = None
    channels: frozenset[str] = DEFAULT_CHANNELS


class FrameLogger:
    """Write per-frame CSV + multi-channel JSONL telemetry.

    Use as a context manager so output files always flush and close:

        with FrameLogger(LoggerConfig(csv_path=..., jsonl_path=...)) as log:
            log.write_meta(model="yolo11s_p2", source="...mp4", config={...})
            for frame in pipeline:
                log.write_frame(state)
    """

    def __init__(self, config: LoggerConfig):
        self.cfg = config
        self._csv_file: IO | None = None
        self._csv_writer: csv.DictWriter | None = None
        self._jsonl_file: IO | None = None
        self._prev_phase: str | None = None
        self._opened = False

    # ------------------------------------------------------------------
    # context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "FrameLogger":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def open(self) -> None:
        if self._opened:
            return
        if self.cfg.csv_path is not None:
            self.cfg.csv_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(self.cfg.csv_path, "w", newline="", buffering=1)
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=FRAME_COLUMNS)
            self._csv_writer.writeheader()
        if self.cfg.jsonl_path is not None:
            self.cfg.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            self._jsonl_file = open(self.cfg.jsonl_path, "w", buffering=1)
        self._opened = True

    def close(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        if self._jsonl_file is not None:
            self._jsonl_file.close()
            self._jsonl_file = None
        self._opened = False

    # ------------------------------------------------------------------
    # public writes
    # ------------------------------------------------------------------

    def write_meta(self, **payload: Any) -> None:
        """Single event at session start. Captures config snapshot."""
        record = {
            "ch": "meta",
            "started_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            **payload,
        }
        self._emit_jsonl(record)

    def write_frame(self, state: HUDState) -> None:
        """Write one frame's data to CSV (one row) and JSONL (multiple events)."""
        row = self._row_from_state(state)
        if self._csv_writer is not None:
            self._csv_writer.writerow(row)

        # JSONL events — gated by channel set
        t = round(state.timestamp_s, 4)

        # frame summary
        if "frame" in self.cfg.channels:
            self._emit_jsonl({
                "ch": "frame",
                "t": t,
                "idx": state.frame_idx,
                "fps": round(state.fps, 1),
                "inf_ms": round(state.inference_ms, 1),
                "n_det": len(state.detections),
                "n_lock": row["n_lockable"],
                "phase": row["phase"],
                "prog": round(row["progress"], 3) if row["progress"] is not None else 0,
                "in_zone": bool(row["in_zone"]),
            })

        # detection events
        if "det" in self.cfg.channels and state.detections:
            for d in state.detections:
                bbox = [round(v, 1) for v in d["bbox"]]
                self._emit_jsonl({
                    "ch": "det",
                    "t": t,
                    "id": d.get("track_id"),
                    "bbox": bbox,
                    "c": round(d.get("confidence", 0.0), 3),
                    "lockable": bool(d.get("lockable")),
                })

        # control event
        if "ctrl" in self.cfg.channels and state.control is not None:
            c = state.control
            self._emit_jsonl({
                "ch": "ctrl",
                "t": t,
                "r": round(c.roll, 3),
                "p": round(c.pitch, 3),
                "y": round(c.yaw, 3),
                "th": round(c.throttle, 3),
                "ex": round(c.error_x_norm, 3),
                "ey": round(c.error_y_norm, 3),
                "sr": round(c.size_ratio, 4),
            })

        # phase transition only
        if "lock" in self.cfg.channels:
            phase = row["phase"]
            if phase != self._prev_phase:
                self._emit_jsonl({
                    "ch": "lock",
                    "t": t,
                    "from": self._prev_phase,
                    "to": phase,
                    "elapsed": round(row["elapsed_s"], 3),
                    "target_id": row["target_id"],
                })
                self._prev_phase = phase

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _emit_jsonl(self, record: dict[str, Any]) -> None:
        if self._jsonl_file is None:
            return
        self._jsonl_file.write(json.dumps(record, separators=(",", ":")))
        self._jsonl_file.write("\n")

    def _row_from_state(self, state: HUDState) -> dict[str, Any]:
        ls = state.lock_state
        c = state.control

        best = _best_lockable(state.detections)
        n_lockable = sum(1 for d in state.detections if d.get("lockable"))

        target_bbox = best["bbox"] if best is not None else (None, None, None, None)
        kbbox = state.kalman_bbox if state.kalman_bbox is not None else (None, None, None, None)
        spec_err = state.spec_error_px if state.spec_error_px is not None else (None, None)
        spec_env = state.spec_envelope_half_px if state.spec_envelope_half_px is not None else (None, None)

        return {
            "frame_idx": state.frame_idx,
            "t_s": round(state.timestamp_s, 4),
            "fps": round(state.fps, 2),
            "inference_ms": round(state.inference_ms, 2),
            "n_det": len(state.detections),
            "n_lockable": n_lockable,
            "target_id": ls.get("target_id"),
            "target_x1": _r(target_bbox[0]),
            "target_y1": _r(target_bbox[1]),
            "target_x2": _r(target_bbox[2]),
            "target_y2": _r(target_bbox[3]),
            "target_conf": round(best["confidence"], 4) if best is not None else None,
            "kalman_x1": _r(kbbox[0]),
            "kalman_y1": _r(kbbox[1]),
            "kalman_x2": _r(kbbox[2]),
            "kalman_y2": _r(kbbox[3]),
            "locked": bool(ls.get("locked", False)),
            "progress": round(float(ls.get("progress", 0.0)), 4),
            "elapsed_s": round(float(ls.get("elapsed_s", 0.0)), 4),
            "dropout_s": round(float(ls.get("dropout_s", 0.0)), 4),
            "in_zone": bool(ls.get("in_zone", False)),
            "phase": phase_from_lock_state(ls),
            "err_x_px": _r(spec_err[0]),
            "err_y_px": _r(spec_err[1]),
            "env_half_w_px": _r(spec_env[0]),
            "env_half_h_px": _r(spec_env[1]),
            "in_envelope": bool(state.in_envelope),
            "roll": round(c.roll, 4) if c is not None else None,
            "pitch": round(c.pitch, 4) if c is not None else None,
            "yaw": round(c.yaw, 4) if c is not None else None,
            "throttle": round(c.throttle, 4) if c is not None else None,
            "error_x_norm": round(c.error_x_norm, 4) if c is not None else None,
            "error_y_norm": round(c.error_y_norm, 4) if c is not None else None,
            "size_ratio": round(c.size_ratio, 4) if c is not None else None,
        }


def _r(v: float | None, ndigits: int = 2) -> float | None:
    """Round optional float to ndigits, preserving None."""
    return None if v is None else round(float(v), ndigits)
