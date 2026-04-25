"""End-to-end simulation pipeline.

Glues InferencePipeline + LockOnStateMachine + Kalman + Controller +
HUDRenderer + FrameLogger into one frame-by-frame loop.

Per frame:
  1. Read frame (optionally simulate Pi-camera resolution).
  2. Run YOLO with ByteTrack (model.track persist=True).
  3. Pick highest-conf lockable detection as the target.
  4. Update Kalman (measurement update or dropout-predict).
  5. Update lock-on state machine (handles 4s counter + 200ms tolerance).
  6. Compute spec validity envelope (AH center vs target center).
  7. Update visual-servo controller; reset on track loss.
  8. Compose HUDState, render, log, write to output video.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import yaml

from src.inference import InferencePipeline, LockOnStateMachine
from src.sim.controller import ControllerConfig, VisualServoController
from src.sim.kalman import ConstantVelocityKalman, KalmanConfig
from src.sim.logger import FrameLogger, LoggerConfig
from src.sim.state import HUDState, phase_from_lock_state
from src.sim.visualizer import HUDRenderer

ROOT = Path(__file__).resolve().parent.parent.parent


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class RunConfig:
    """Per-run knobs that aren't in deployment.yaml or simulation.yaml."""
    video_path: Path
    out_video_path: Path | None = None
    csv_path: Path | None = None
    jsonl_path: Path | None = None
    display: bool = False
    simulate_camera: bool = False
    max_frames: int | None = None


def _best_lockable(detections: list[dict]) -> dict | None:
    lockable = [d for d in detections if d.get("lockable")]
    if not lockable:
        return None
    return max(lockable, key=lambda d: d.get("confidence", 0.0))


def _compute_spec_envelope(
    kalman_bbox: tuple[float, float, float, float] | None,
    target: dict | None,
) -> tuple[tuple[float, float], tuple[float, float], bool] | None:
    """Spec p.14: |err_x| <= ½ target_w AND |err_y| <= ½ target_h.

    Returns (error_px, half_envelope_px, in_envelope) or None if either
    bbox is missing.
    """
    if kalman_bbox is None or target is None:
        return None
    kx1, ky1, kx2, ky2 = kalman_bbox
    tx1, ty1, tx2, ty2 = target["bbox"]
    kcx, kcy = (kx1 + kx2) / 2, (ky1 + ky2) / 2
    tcx, tcy = (tx1 + tx2) / 2, (ty1 + ty2) / 2
    half_w = (tx2 - tx1) / 2
    half_h = (ty2 - ty1) / 2
    err_x = kcx - tcx
    err_y = kcy - tcy
    ok = abs(err_x) <= half_w and abs(err_y) <= half_h
    return (err_x, err_y), (half_w, half_h), ok


class SimulationPipeline:
    """Top-level orchestrator. Stateless across runs but stateful within one."""

    def __init__(
        self,
        model_path: str | Path,
        deployment_cfg: dict,
        sim_cfg: dict,
    ):
        self.deployment_cfg = deployment_cfg
        self.sim_cfg = sim_cfg
        self.model_path = Path(model_path)

        # Inference + lock-on (existing modules — single source of truth for
        # competition rules). We pass simulate_camera=False because we apply
        # the camera-resolution resize ourselves in pipeline.py (see _iter_frames).
        self.inference = InferencePipeline(str(model_path), deployment_cfg)
        self.lock_sm = LockOnStateMachine(deployment_cfg)

        # Sim-specific modules — config-driven so tuning lives in YAML
        self.kalman = ConstantVelocityKalman(KalmanConfig(**sim_cfg["kalman"]))
        self.controller = VisualServoController(
            ControllerConfig.from_dict(sim_cfg["controller"])
        )

        # Server time anchor (HUD display only)
        st = sim_cfg.get("server_time", {})
        self._server_time_offset_s = (
            st.get("start_h", 0) * 3600
            + st.get("start_m", 0) * 60
            + st.get("start_s", 0)
        )

        # Camera resolution for --simulate-camera (cached so the per-frame
        # resize doesn't re-read the dict).
        cam_res = deployment_cfg["camera"]["resolution"]
        self._camera_res = (int(cam_res[0]), int(cam_res[1]))

        # Display name for HUD/meta. Convention is runs/<name>/weights/best.pt;
        # fall back to the file stem if the path doesn't follow that layout.
        if self.model_path.parent.name == "weights":
            self.model_name = self.model_path.parent.parent.name
        else:
            self.model_name = self.model_path.stem

        # Track previous lock-target so we know when to reset Kalman/controller.
        # Reset triggers: target_id changed, or lock returned to SEARCHING.
        self._prev_target_id: int | None = None
        self._prev_phase: str | None = None

        self._renderer: HUDRenderer | None = None  # built lazily on first frame

    def run(self, run_cfg: RunConfig) -> dict:
        """Run the pipeline end-to-end. Returns summary statistics."""
        cap = cv2.VideoCapture(str(run_cfg.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {run_cfg.video_path}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n_frames_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Resolved processing resolution: source dims, or simulated Pi-cam res.
        if run_cfg.simulate_camera:
            fw, fh = self._camera_res
        else:
            fw, fh = src_w, src_h

        self._renderer = HUDRenderer(fw, fh)
        out_w, out_h = self._renderer.output_size

        writer = self._open_writer(run_cfg.out_video_path, src_fps, out_w, out_h)
        channels = frozenset(self.sim_cfg["logger"]["channels"])
        logger_cfg = LoggerConfig(
            csv_path=run_cfg.csv_path,
            jsonl_path=run_cfg.jsonl_path,
            channels=channels,
        )

        n_frames = 0
        n_locked = 0
        n_dropouts = 0
        wall_t0 = time.perf_counter()

        with FrameLogger(logger_cfg) as logger:
            logger.write_meta(
                model_name=self.model_name,
                model_path=str(self.model_path),
                source=str(run_cfg.video_path),
                src_fps=src_fps,
                src_frames=n_frames_in,
                output=str(run_cfg.out_video_path) if run_cfg.out_video_path else None,
                simulate_camera=run_cfg.simulate_camera,
                deployment_cfg=self.deployment_cfg,
                sim_cfg=self.sim_cfg,
            )

            for state, composite in self._iter_frames(cap, src_fps, run_cfg):
                logger.write_frame(state)
                if writer is not None:
                    writer.write(composite)
                if run_cfg.display:
                    cv2.imshow("UAV Simulation", composite)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break

                n_frames += 1
                if state.lock_state.get("locked"):
                    n_locked += 1
                if state.lock_state.get("dropout_s", 0) > 0:
                    n_dropouts += 1

                if run_cfg.max_frames is not None and n_frames >= run_cfg.max_frames:
                    break

        cap.release()
        if writer is not None:
            writer.release()
        if run_cfg.display:
            cv2.destroyAllWindows()

        wall_dur_s = time.perf_counter() - wall_t0
        return {
            "frames": n_frames,
            "locked_frames": n_locked,
            "dropout_frames": n_dropouts,
            "lock_pct": round(100 * n_locked / n_frames, 1) if n_frames else 0.0,
            "wall_s": round(wall_dur_s, 2),
            "pipeline_fps": round(n_frames / wall_dur_s, 1) if wall_dur_s > 0 else 0.0,
        }

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _iter_frames(
        self, cap: cv2.VideoCapture, src_fps: float, run_cfg: RunConfig,
    ) -> Iterator[tuple[HUDState, np.ndarray]]:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                return

            if run_cfg.simulate_camera:
                # Resize source frame to Pi Camera Module 3 resolution. Done
                # here (not in InferencePipeline) so all downstream consumers
                # see consistently-sized frames.
                frame = cv2.resize(frame, self._camera_res, interpolation=cv2.INTER_LINEAR)

            t_video_s = idx / src_fps  # simulation time

            # Inference timing measured separately so HUD shows model-only ms
            t0 = time.perf_counter()
            detections = self.inference.predict(frame, use_tracking=True)
            inference_ms = (time.perf_counter() - t0) * 1000.0

            # Pick the target before lock_sm.update so we can drive Kalman first
            target = _best_lockable(detections)
            new_target_id = target.get("track_id") if target is not None else None

            # Track-change reset: if the locked target changed identity, drop
            # accumulated state. (Track loss is handled separately via the
            # phase transition below.)
            if (
                new_target_id is not None
                and self._prev_target_id is not None
                and new_target_id != self._prev_target_id
            ):
                self.kalman.reset()
                self.controller.reset()

            # Kalman: measurement update if we have a target, else dropout-predict
            if target is not None:
                self.kalman.update(tuple(target["bbox"]), t_video_s)
            else:
                self.kalman.predict_only(t_video_s)

            # Lock-on state machine
            lock_state = self.lock_sm.update(detections, t_video_s, frame.shape)
            phase = phase_from_lock_state(lock_state)

            # Reset Kalman + controller on transition into SEARCHING (lock lost)
            if phase == "SEARCHING" and self._prev_phase != "SEARCHING":
                self.kalman.reset()
                self.controller.reset()

            # Spec envelope (AH center vs target center)
            envelope = _compute_spec_envelope(self.kalman.bbox, target)
            if envelope is not None:
                spec_err, spec_env_half, in_env = envelope
            else:
                spec_err = spec_env_half = None
                in_env = False

            # Controller — only when we have a Kalman bbox to follow
            control = None
            if self.kalman.bbox is not None:
                dt = 1.0 / src_fps
                control = self.controller.step(
                    self.kalman.bbox, frame.shape[1], frame.shape[0], dt,
                )
            else:
                self.controller.reset()

            # HUD state
            comp = self.deployment_cfg["competition"]
            state = HUDState(
                timestamp_s=self._server_time_offset_s + t_video_s,
                frame_idx=idx,
                fps=1000.0 / inference_ms if inference_ms > 0 else 0.0,
                inference_ms=inference_ms,
                detections=detections,
                kalman_bbox=self.kalman.bbox,
                lock_state=lock_state,
                control=control,
                spec_error_px=spec_err,
                spec_envelope_half_px=spec_env_half,
                in_envelope=in_env,
                ah_w_frac=comp.get("ah_width", 0.35),
                ah_h_frac=comp.get("ah_height", 0.50),
                av_w_frac=1.0 - 2 * comp.get("av_margin_x", 0.25),
                av_h_frac=1.0 - 2 * comp.get("av_margin_y", 0.10),
                model_name=self.model_name,
            )

            composite = self._renderer.render(frame, state)

            # Update previous-state trackers
            self._prev_target_id = new_target_id
            self._prev_phase = phase

            yield state, composite
            idx += 1

    @staticmethod
    def _open_writer(
        path: Path | None, fps: float, w: int, h: int,
    ) -> cv2.VideoWriter | None:
        if path is None:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        # mp4v is universally compatible with OpenCV (spec mentions OpenCV 4.5)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {path}")
        return writer
