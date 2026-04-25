"""Reference inference pipeline for Pi 5 + Hailo-8L deployment.

Implements the exact preprocessing, detection, tracking, and lock-on
logic that will run during competition. Test on your PC before deploying.

The pipeline:
  Camera frame -> [simulate resolution] -> preprocess -> detect -> track -> lock decision

Features:
  - Pi Camera Module 3 resolution simulation (--simulate-camera)
  - Object tracking with persistent IDs (ByteTrack)
  - Lock-on state machine (4s continuous detection)
  - 200ms dropout tolerance
  - 2Hz telemetry rate limiting
  - Dual confidence thresholds (det_conf=0.35 / lock_conf=0.75)
  - 6% minimum bbox size filtering (spec requires 5%, recommends 6%+)
  - AH zone check (target center must be inside lockdown quadrilateral)

Usage:
    # Webcam test (point at UAV images on screen)
    python src/inference.py --model runs/yolo11s_p2/weights/best.pt --source 0

    # Webcam with Pi Camera resolution simulation
    python src/inference.py --model runs/yolo11s_p2/weights/best.pt --source 0 --simulate-camera

    # Test on images (no tracking)
    python src/inference.py --model runs/yolo11s_p2/weights/best.pt --source test

    # Benchmark inference speed
    python src/inference.py --model runs/yolo11s_p2/weights/best.pt --benchmark
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_deployment_config() -> dict:
    with open(ROOT / "configs" / "deployment.yaml") as f:
        return yaml.safe_load(f)


class LockOnStateMachine:
    """Competition lock-on logic with AH zone check and dropout tolerance.

    Rules (from SAVASAN IHA competition 6.1.1):
      - Lock-on requires 4 seconds of continuous detection
      - Target CENTER must be inside AH (Lockdown Quadrilateral)
      - Target bbox must cover >=5% of frame in at least one axis (we use 6%)
      - Gaps <= 200ms don't reset the timer (but NOT at start/end)
      - Only detections meeting size + confidence + zone checks count
      - False lock costs -30 points
      - Telemetry sent at max 2Hz

    Time-based (not frame-based) so it works correctly at any FPS —
    whether testing on a PC webcam at 15fps or on the Pi at 56fps.
    """

    def __init__(self, config: dict):
        comp = config["competition"]
        self.lock_on_seconds = comp["lock_on_seconds"]          # 4.0
        self.dropout_tolerance_s = comp["dropout_tolerance_ms"] / 1000.0  # 0.2
        self.telemetry_interval = 1.0 / comp["telemetry_hz"]    # 0.5

        # AH zone (Lockdown Quadrilateral) — target center must be inside
        self.ah_width = comp.get("ah_width", 0.35)
        self.ah_height = comp.get("ah_height", 0.50)

        # State
        self.tracking_start = None   # when continuous tracking began
        self.last_seen = None        # last time a lockable detection appeared
        self.locked = False
        self.lock_target_id = None
        self.last_telemetry_time = 0.0
        self.frame_count_in_lock = 0  # track frames since lock start (for boundary check)

    def _is_in_ah_zone(self, detection: dict, frame_w: int, frame_h: int) -> bool:
        """Check if the detection center is inside the AH lockdown zone."""
        x1, y1, x2, y2 = detection["bbox"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        ah_x_min = frame_w * (0.5 - self.ah_width / 2)
        ah_x_max = frame_w * (0.5 + self.ah_width / 2)
        ah_y_min = frame_h * (0.5 - self.ah_height / 2)
        ah_y_max = frame_h * (0.5 + self.ah_height / 2)

        return ah_x_min <= cx <= ah_x_max and ah_y_min <= cy <= ah_y_max

    def update(self, detections: list[dict], timestamp: float,
               frame_shape: tuple = None) -> dict:
        """Process one frame's detections, return lock state.

        Args:
            detections: list of detection dicts from InferencePipeline.predict()
            timestamp: current time in seconds
            frame_shape: (h, w, ...) of the frame, needed for AH zone check

        Returns dict with: locked, progress (0-1), elapsed_s,
        dropout_s, target_id, in_zone, send_telemetry.
        """
        # Filter: must be lockable AND inside AH zone
        lockable = []
        in_zone = False
        if frame_shape is not None:
            fh, fw = frame_shape[:2]
            for d in detections:
                if d.get("lockable") and self._is_in_ah_zone(d, fw, fh):
                    lockable.append(d)
                    in_zone = True
        else:
            # Fallback: no zone check if frame_shape not provided
            lockable = [d for d in detections if d.get("lockable")]
            in_zone = bool(lockable)

        if lockable:
            best = max(lockable, key=lambda d: d["confidence"])
            self.lock_target_id = best.get("track_id")
            self.last_seen = timestamp
            self.frame_count_in_lock += 1

            if self.tracking_start is None:
                self.tracking_start = timestamp
                self.frame_count_in_lock = 1  # reset frame counter on new lock attempt
        else:
            # No valid detection — check dropout tolerance
            if self.last_seen is not None:
                gap = timestamp - self.last_seen

                # Per spec: dropout tolerance does not apply at the start or end
                # of the LOCK-ON ATTEMPT (the 4s window). Once the lock is
                # achieved (elapsed >= lock_on_seconds), normal 200ms tolerance
                # resumes for maintaining the lock.
                elapsed_so_far = (
                    timestamp - self.tracking_start
                    if self.tracking_start is not None
                    else 0.0
                )
                in_lock_attempt = elapsed_so_far < self.lock_on_seconds
                near_end_of_attempt = (
                    elapsed_so_far >= self.lock_on_seconds - self.dropout_tolerance_s
                )
                at_boundary = (
                    self.frame_count_in_lock <= 2
                    or (in_lock_attempt and near_end_of_attempt)
                )

                if at_boundary or gap > self.dropout_tolerance_s:
                    # Lost target — reset lock progress
                    self.tracking_start = None
                    self.locked = False
                    self.lock_target_id = None
                    self.frame_count_in_lock = 0

        # Calculate progress
        if self.tracking_start is not None:
            elapsed = timestamp - self.tracking_start
            progress = min(elapsed / self.lock_on_seconds, 1.0)
            if elapsed >= self.lock_on_seconds:
                self.locked = True
        else:
            elapsed = 0.0
            progress = 0.0

        # Telemetry rate limiting (2Hz)
        send_telemetry = False
        if timestamp - self.last_telemetry_time >= self.telemetry_interval:
            send_telemetry = True
            self.last_telemetry_time = timestamp

        dropout_s = 0.0
        if self.last_seen is not None and not lockable:
            dropout_s = timestamp - self.last_seen

        return {
            "locked": self.locked,
            "progress": progress,
            "elapsed_s": elapsed,
            "dropout_s": dropout_s,
            "target_id": self.lock_target_id,
            "in_zone": in_zone,
            "send_telemetry": send_telemetry,
        }

    def reset(self):
        """Clear all lock state (bound to 'r' key during testing)."""
        self.tracking_start = None
        self.last_seen = None
        self.locked = False
        self.lock_target_id = None


class InferencePipeline:
    """End-to-end inference pipeline matching deployment conditions.

    Preprocessing mirrors what the model saw during training:
    resize -> letterbox -> normalize. Optional CLAHE can be toggled via
    deployment.yaml based on ablation results.
    """

    def __init__(self, model_path: str, config: dict | None = None,
                 simulate_camera: bool = False):
        from ultralytics import YOLO

        self.config = config or load_deployment_config()
        self.model = YOLO(model_path)
        self.simulate_camera = simulate_camera

        # Inference settings
        inf = self.config["inference"]
        self.imgsz = inf["imgsz"]
        self.det_conf = inf["det_conf"]
        self.lock_conf = inf["lock_conf"]
        self.iou_thresh = inf["iou_thresh"]
        self.use_clahe = inf["use_clahe"]

        # Competition rules
        comp = self.config["competition"]
        self.min_bbox_ratio = comp["min_bbox_ratio"]

        # Camera specs
        cam = self.config["camera"]
        self.camera_res = (cam["resolution"][0], cam["resolution"][1])

        # CLAHE instance (created once, reused)
        if self.use_clahe:
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        cam_sim = (f"ON ({self.camera_res[0]}x{self.camera_res[1]})"
                   if self.simulate_camera else "OFF")
        print(f"  Pipeline initialized:")
        print(f"    Model:       {model_path}")
        print(f"    Input size:  {self.imgsz}x{self.imgsz}")
        print(f"    Det conf:    {self.det_conf}")
        print(f"    Lock conf:   {self.lock_conf}")
        print(f"    CLAHE:       {'ON' if self.use_clahe else 'OFF'}")
        print(f"    Min bbox:    {self.min_bbox_ratio*100:.0f}% of image axis (spec recommends >=6%)")
        print(f"    AH zone:    {comp.get('ah_width', 0.35)*100:.0f}%W x {comp.get('ah_height', 0.50)*100:.0f}%H centered")
        print(f"    Camera sim:  {cam_sim}")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Apply inference-time preprocessing to a raw camera frame.

        CLAHE is the only optional step. Letterboxing and normalization
        are handled by Ultralytics during model.predict()/track().
        """
        if self.use_clahe:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return frame

    def _apply_camera_simulation(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to Pi Camera Module 3 resolution (2304x1296).

        Tests the full downscaling pipeline the Pi will run:
        2304x1296 capture -> 640x640 letterbox -> model -> map back.
        Ensures bbox size thresholds use the correct base dimensions.
        """
        return cv2.resize(frame, self.camera_res, interpolation=cv2.INTER_LINEAR)

    def predict(self, frame: np.ndarray, use_tracking: bool = False) -> list[dict]:
        """Full pipeline: preprocess -> detect/track -> postprocess -> filter.

        Args:
            frame: BGR image (raw camera frame).
            use_tracking: If True, use model.track() with persistent IDs.
                          Use for video/webcam. False for single images.

        Returns list of detections, each with:
          - bbox: [x1, y1, x2, y2] in original frame coordinates
          - confidence: float
          - lockable: bool (meets size + confidence thresholds)
          - track_id: int or None (only when use_tracking=True)
        """
        processed = self.preprocess(frame)

        if use_tracking:
            results = self.model.track(
                processed,
                imgsz=self.imgsz,
                conf=self.det_conf,
                iou=self.iou_thresh,
                persist=True,
                verbose=False,
            )
        else:
            results = self.model.predict(
                processed,
                imgsz=self.imgsz,
                conf=self.det_conf,
                iou=self.iou_thresh,
                verbose=False,
            )

        return self._postprocess(results[0], frame.shape)

    def _postprocess(self, result, original_shape: tuple) -> list[dict]:
        """Filter detections by competition rules.

        Applies:
          1. Confidence threshold (det_conf already applied by YOLO)
          2. Minimum bbox size filter (5% of image axis)
          3. Lock-on eligibility (confidence >= lock_conf AND meets size)
          4. Extract track ID when available
        """
        h, w = original_shape[:2]
        min_px_w = w * self.min_bbox_ratio
        min_px_h = h * self.min_bbox_ratio

        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu())
            bw = x2 - x1
            bh = y2 - y1

            # Competition rule: reject if bbox < 5% on BOTH axes
            meets_size = bw >= min_px_w or bh >= min_px_h

            # Extract track ID (only present when using model.track())
            track_id = None
            if box.id is not None:
                track_id = int(box.id[0].cpu())

            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": round(conf, 4),
                "bbox_w_ratio": round(bw / w, 4),
                "bbox_h_ratio": round(bh / h, 4),
                "meets_size": meets_size,
                "lockable": meets_size and conf >= self.lock_conf,
                "track_id": track_id,
            })

        return detections

    def run_on_images(self, image_paths: list[Path], display: bool = True) -> None:
        """Run inference on images (single-frame mode, no tracking)."""
        for img_path in image_paths:
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"  WARNING: Could not read {img_path}")
                continue

            detections = self.predict(frame)
            vis = self._draw_detections(frame, detections)

            lockable = [d for d in detections if d["lockable"]]
            det_only = [d for d in detections if not d["lockable"]]
            print(
                f"  {img_path.name}: "
                f"{len(lockable)} lockable, "
                f"{len(det_only)} detected (below lock threshold)"
            )

            if display:
                cv2.imshow("UAV Detection", vis)
                key = cv2.waitKey(0) & 0xFF
                if key == ord("q"):
                    break

        if display:
            cv2.destroyAllWindows()

    def run_on_video(self, source, display: bool = True) -> None:
        """Run full competition pipeline on video/webcam.

        Includes: camera simulation, ByteTrack tracking, lock-on state
        machine with 4s timer, 200ms dropout tolerance, 2Hz telemetry.

        Controls:
          q — quit
          r — reset lock state
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"  ERROR: Could not open video source: {source}")
            return

        lock_sm = LockOnStateMachine(self.config)
        fps_times = []
        frame_count = 0
        start_time = time.perf_counter()

        sim_label = (f" (simulated {self.camera_res[0]}x{self.camera_res[1]})"
                     if self.simulate_camera else "")
        print(f"  Running on video source: {source}{sim_label}")
        print(f"  Tracking: ON (ByteTrack)")
        print(f"  Lock-on:  {self.config['competition']['lock_on_seconds']}s continuous")
        print(f"  Dropout:  {self.config['competition']['dropout_tolerance_ms']}ms tolerance")
        print(f"  Telemetry: {self.config['competition']['telemetry_hz']}Hz max")
        print(f"  Press 'q' to quit, 'r' to reset lock state\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Camera resolution simulation
            if self.simulate_camera:
                frame = self._apply_camera_simulation(frame)

            t0 = time.perf_counter()
            detections = self.predict(frame, use_tracking=True)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            fps_times.append(elapsed_ms)

            # Update lock-on state machine (with frame shape for AH zone check)
            timestamp = time.perf_counter() - start_time
            lock_state = lock_sm.update(detections, timestamp, frame.shape)

            # Visualize
            vis = self._draw_detections(frame, detections)
            self._draw_lock_status(vis, lock_state, elapsed_ms, len(detections))

            # Telemetry output at 2Hz
            if lock_state["send_telemetry"]:
                lockable = [d for d in detections if d.get("lockable")]
                if lockable:
                    best = max(lockable, key=lambda d: d["confidence"])
                    cx = (best["bbox"][0] + best["bbox"][2]) / 2
                    cy = (best["bbox"][1] + best["bbox"][3]) / 2
                    status = "LOCKED" if lock_state["locked"] else "TRACKING"
                    print(
                        f"  [TELEMETRY] {status} | "
                        f"ID:{best['track_id']} | "
                        f"pos:({cx:.0f},{cy:.0f}) | "
                        f"conf:{best['confidence']:.2f} | "
                        f"progress:{lock_state['progress']:.0%}"
                    )

            frame_count += 1

            if display:
                # Scale down for display if simulating high resolution
                if self.simulate_camera:
                    disp_h = 720
                    disp_w = int(disp_h * self.camera_res[0] / self.camera_res[1])
                    vis = cv2.resize(vis, (disp_w, disp_h))

                cv2.imshow("UAV Detection", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("r"):
                    lock_sm.reset()
                    print("  [RESET] Lock state cleared")

        cap.release()
        if display:
            cv2.destroyAllWindows()

        if fps_times:
            avg_ms = sum(fps_times) / len(fps_times)
            print(f"\n  Session summary:")
            print(f"    Frames:    {frame_count}")
            print(f"    Avg:       {avg_ms:.1f} ms/frame")
            print(f"    FPS:       {1000/avg_ms:.1f}")

    def benchmark(self, n_frames: int = 100) -> dict:
        """Benchmark inference speed on test images.

        Loops through test images to simulate continuous inference.
        Reports timing statistics matching deployment conditions.
        """
        test_dir = ROOT / "data" / "images" / "test"
        test_imgs = sorted(test_dir.glob("*"))
        if not test_imgs:
            print("  ERROR: No test images found")
            return {}

        # Warmup (5 frames)
        warmup_img = cv2.imread(str(test_imgs[0]))
        for _ in range(5):
            self.predict(warmup_img)

        # Benchmark
        times = []
        img_idx = 0
        for i in range(n_frames):
            frame = cv2.imread(str(test_imgs[img_idx % len(test_imgs)]))
            img_idx += 1

            t0 = time.perf_counter()
            self.predict(frame)
            times.append((time.perf_counter() - t0) * 1000)

        times.sort()
        result = {
            "n_frames": n_frames,
            "mean_ms": round(sum(times) / len(times), 1),
            "median_ms": round(times[len(times) // 2], 1),
            "min_ms": round(times[0], 1),
            "max_ms": round(times[-1], 1),
            "p95_ms": round(times[int(len(times) * 0.95)], 1),
            "fps": round(1000 / (sum(times) / len(times)), 1),
        }

        print(f"\n  Inference benchmark ({n_frames} frames):")
        print(f"    Mean:    {result['mean_ms']} ms")
        print(f"    Median:  {result['median_ms']} ms")
        print(f"    P95:     {result['p95_ms']} ms")
        print(f"    FPS:     {result['fps']}")

        # Check against competition requirements
        dropout_ms = self.config["competition"]["dropout_tolerance_ms"]
        if result["p95_ms"] > dropout_ms:
            print(
                f"    WARNING: P95 latency ({result['p95_ms']}ms) exceeds "
                f"dropout tolerance ({dropout_ms}ms)"
            )
        else:
            print(f"    OK: P95 latency within {dropout_ms}ms dropout tolerance")

        return result

    def _draw_detections(
        self, frame: np.ndarray, detections: list[dict]
    ) -> np.ndarray:
        """Draw bounding boxes with lock status, track IDs, and color coding.

        Green  = lockable (meets size + confidence thresholds)
        Yellow = detected but below lock threshold
        Red    = detected but too small (fails 5% size check)
        """
        vis = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            conf = det["confidence"]
            tid = det.get("track_id")

            if det["lockable"]:
                color = (0, 255, 0)   # green
                label = f"LOCK {conf:.2f}"
            elif det["meets_size"]:
                color = (0, 255, 255) # yellow
                label = f"DET {conf:.2f}"
            else:
                color = (0, 0, 255)   # red
                label = f"SMALL {conf:.2f}"

            # Prepend track ID when available
            if tid is not None:
                label = f"#{tid} {label}"

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                vis, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )

        return vis

    def _draw_lock_status(self, frame: np.ndarray, lock_state: dict,
                          inference_ms: float, n_detections: int) -> None:
        """Draw lock-on progress bar, FPS, and status overlay on frame."""
        h, w = frame.shape[:2]

        # --- FPS + detection count (top left) ---
        fps = 1000 / inference_ms if inference_ms > 0 else 0
        cv2.putText(
            frame,
            f"FPS: {fps:.1f} | Det: {n_detections}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )

        # --- Target ID (top right) ---
        if lock_state["target_id"] is not None:
            tid_text = f"Target: #{lock_state['target_id']}"
            text_size = cv2.getTextSize(
                tid_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )[0]
            cv2.putText(
                frame, tid_text, (w - text_size[0] - 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
            )

        # --- Lock progress bar (bottom center) ---
        bar_h = 28
        bar_w = min(int(w * 0.5), 500)
        bar_x = (w - bar_w) // 2
        bar_y = h - bar_h - 15
        progress = lock_state["progress"]

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (bar_x - 2, bar_y - 2),
            (bar_x + bar_w + 2, bar_y + bar_h + 2),
            (0, 0, 0), -1,
        )
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Progress fill
        fill_w = int(bar_w * progress)
        if lock_state["locked"]:
            fill_color = (0, 255, 0)     # green — locked
        elif lock_state["dropout_s"] > 0:
            fill_color = (0, 100, 255)   # orange — in dropout
        else:
            fill_color = (0, 200, 255)   # yellow — tracking

        if fill_w > 0:
            cv2.rectangle(
                frame, (bar_x, bar_y),
                (bar_x + fill_w, bar_y + bar_h), fill_color, -1,
            )

        # Border
        cv2.rectangle(
            frame, (bar_x, bar_y),
            (bar_x + bar_w, bar_y + bar_h), (255, 255, 255), 1,
        )

        # Status text inside bar
        lock_s = self.config["competition"]["lock_on_seconds"]
        if lock_state["locked"]:
            text = "LOCKED ON"
            text_color = (0, 255, 0)
        elif progress > 0:
            text = f"LOCKING: {lock_state['elapsed_s']:.1f}s / {lock_s}s"
            text_color = (0, 200, 255)
        else:
            text = "SEARCHING..."
            text_color = (180, 180, 180)

        cv2.putText(
            frame, text, (bar_x + 5, bar_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2,
        )

        # Dropout warning (above bar)
        if lock_state["dropout_s"] > 0:
            dropout_ms = self.config["competition"]["dropout_tolerance_ms"]
            dropout_text = (f"DROPOUT: {lock_state['dropout_s']*1000:.0f}ms"
                            f" / {dropout_ms}ms")
            cv2.putText(
                frame, dropout_text, (bar_x, bar_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1,
            )

        # Zone status (below bar)
        in_zone = lock_state.get("in_zone", False)
        zone_color = (0, 255, 0) if in_zone else (0, 0, 255)
        zone_text = "IN AH ZONE" if in_zone else "OUTSIDE AH"
        cv2.putText(
            frame, zone_text, (bar_x, bar_y + bar_h + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, zone_color, 1,
        )


def _resolve_source(source_str: str):
    """Resolve source string to video capture input."""
    if source_str == "test":
        return "test"
    try:
        return int(source_str)
    except ValueError:
        return source_str


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run inference pipeline (reference Pi deployment)"
    )
    parser.add_argument(
        "--model", "-m", type=str, required=True,
        help="Path to model weights (.pt)",
    )
    parser.add_argument(
        "--source", "-s", type=str, default="test",
        help="Input source: 'test' (images), webcam index (0), or video path",
    )
    parser.add_argument(
        "--benchmark", "-b", action="store_true",
        help="Run speed benchmark",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable visualization windows",
    )
    parser.add_argument(
        "--simulate-camera", action="store_true",
        help="Resize frames to Pi Camera Module 3 resolution (2304x1296)",
    )
    parser.add_argument(
        "--n-frames", type=int, default=100,
        help="Number of frames for benchmark",
    )
    args = parser.parse_args()

    # Resolve model path
    model_path = args.model
    if not Path(model_path).exists():
        full_path = ROOT / model_path
        if full_path.exists():
            model_path = str(full_path)
        else:
            print(f"ERROR: Model not found: {model_path}", file=sys.stderr)
            return 1

    pipeline = InferencePipeline(
        model_path,
        simulate_camera=args.simulate_camera,
    )

    if args.benchmark:
        pipeline.benchmark(n_frames=args.n_frames)
        return 0

    source = _resolve_source(args.source)
    display = not args.no_display

    if source == "test":
        test_dir = ROOT / "data" / "images" / "test"
        images = sorted(test_dir.glob("*"))
        if not images:
            print("ERROR: No test images found in data/images/test/", file=sys.stderr)
            return 1
        print(f"\n  Running on {len(images)} test images...")
        pipeline.run_on_images(images, display=display)
    else:
        pipeline.run_on_video(source, display=display)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
