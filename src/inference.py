"""Reference inference pipeline for Pi 5 + Hailo-8L deployment.

Implements the exact preprocessing and postprocessing that will run
during competition. Use this to test detection on your PC before
deploying to the Pi.

The pipeline:
  Camera frame → preprocess → detect → postprocess → track → lock decision

Usage:
    # Test on test images
    python src/inference.py --model runs/yolo11s_p2/weights/best.pt --source test

    # Test on a video file
    python src/inference.py --model runs/yolo11s_p2/weights/best.pt --source video.mp4

    # Test on webcam (show plane images to camera for testing)
    python src/inference.py --model runs/yolo11s_p2/weights/best.pt --source 0

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


class InferencePipeline:
    """End-to-end inference pipeline matching deployment conditions.

    Preprocessing exactly mirrors what the model saw during training:
    resize → letterbox → normalize. Optional CLAHE can be toggled via
    deployment.yaml based on Round 2 ablation results.
    """

    def __init__(self, model_path: str, config: dict | None = None):
        from ultralytics import YOLO

        self.config = config or load_deployment_config()
        self.model = YOLO(model_path)

        # Inference settings from deployment config
        inf = self.config["inference"]
        self.imgsz = inf["imgsz"]
        self.det_conf = inf["det_conf"]
        self.lock_conf = inf["lock_conf"]
        self.iou_thresh = inf["iou_thresh"]
        self.use_clahe = inf["use_clahe"]

        # Competition rules
        comp = self.config["competition"]
        self.min_bbox_ratio = comp["min_bbox_ratio"]

        # Camera specs (for logging)
        cam = self.config["camera"]
        self.camera_res = tuple(cam["resolution"])

        # CLAHE instance (created once, reused)
        if self.use_clahe:
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        print(f"  Pipeline initialized:")
        print(f"    Model:       {model_path}")
        print(f"    Input size:  {self.imgsz}x{self.imgsz}")
        print(f"    Det conf:    {self.det_conf}")
        print(f"    Lock conf:   {self.lock_conf}")
        print(f"    CLAHE:       {'ON' if self.use_clahe else 'OFF'}")
        print(f"    Min bbox:    {self.min_bbox_ratio*100:.0f}% of image axis")

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Apply inference-time preprocessing to a raw camera frame.

        Steps:
          1. Optional CLAHE contrast enhancement
          2. Resize + letterbox handled by YOLO internally

        CLAHE is the only optional preprocessing step. Letterboxing and
        normalization are handled by Ultralytics during model.predict().
        We apply CLAHE here BEFORE the model sees the frame.
        """
        if self.use_clahe:
            # CLAHE works on single-channel — convert to LAB, enhance L, convert back
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
            frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

        return frame

    def predict(self, frame: np.ndarray) -> list[dict]:
        """Full pipeline: preprocess → detect → postprocess → filter.

        Returns list of detections, each with:
          - bbox: [x1, y1, x2, y2] in original frame coordinates
          - confidence: float
          - lockable: bool (meets competition size + confidence thresholds)
        """
        # Preprocess
        processed = self.preprocess(frame)

        # Detect (YOLO handles resize, letterbox, normalize internally)
        results = self.model.predict(
            processed,
            imgsz=self.imgsz,
            conf=self.det_conf,
            iou=self.iou_thresh,
            verbose=False,
        )

        # Postprocess
        return self._postprocess(results[0], frame.shape)

    def _postprocess(self, result, original_shape: tuple) -> list[dict]:
        """Filter detections by competition rules.

        Applies:
          1. Confidence threshold (det_conf already applied by YOLO)
          2. Minimum bbox size filter (5% of image axis)
          3. Lock-on eligibility check (confidence >= lock_conf)
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

            detections.append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": round(conf, 4),
                "bbox_w_ratio": round(bw / w, 4),
                "bbox_h_ratio": round(bh / h, 4),
                "meets_size": meets_size,
                "lockable": meets_size and conf >= self.lock_conf,
            })

        return detections

    def run_on_images(self, image_paths: list[Path], display: bool = True) -> None:
        """Run inference on a list of images with optional visualization."""
        for img_path in image_paths:
            frame = cv2.imread(str(img_path))
            if frame is None:
                print(f"  WARNING: Could not read {img_path}")
                continue

            detections = self.predict(frame)
            vis = self._draw_detections(frame, detections)

            # Print summary
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
        """Run inference on video source (file path, webcam index, or 'test').

        For testing: point your webcam at plane/UAV images on a screen.
        Press 'q' to quit.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"  ERROR: Could not open video source: {source}")
            return

        fps_counter = []
        frame_count = 0

        print(f"  Running on video source: {source}")
        print(f"  Press 'q' to quit\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            detections = self.predict(frame)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            fps_counter.append(elapsed_ms)

            vis = self._draw_detections(frame, detections)

            # Overlay FPS
            current_fps = 1000 / elapsed_ms if elapsed_ms > 0 else 0
            cv2.putText(
                vis,
                f"FPS: {current_fps:.1f} | Det: {len(detections)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            frame_count += 1

            if display:
                cv2.imshow("UAV Detection", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if display:
            cv2.destroyAllWindows()

        if fps_counter:
            avg_ms = sum(fps_counter) / len(fps_counter)
            print(f"\n  Video inference summary:")
            print(f"    Frames:   {frame_count}")
            print(f"    Avg:      {avg_ms:.1f} ms/frame")
            print(f"    FPS:      {1000/avg_ms:.1f}")

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
        """Draw bounding boxes on frame with lock-on status color coding.

        Green = lockable (meets size + confidence thresholds)
        Yellow = detected but below lock threshold
        Red = detected but too small (fails size check)
        """
        vis = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
            conf = det["confidence"]

            if det["lockable"]:
                color = (0, 255, 0)  # green — ready to lock
                label = f"LOCK {conf:.2f}"
            elif det["meets_size"]:
                color = (0, 255, 255)  # yellow — detected, low confidence
                label = f"DET {conf:.2f}"
            else:
                color = (0, 0, 255)  # red — too small
                label = f"SMALL {conf:.2f}"

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                vis, label, (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )

        return vis


def _resolve_source(source_str: str):
    """Resolve source string to video capture input.

    Handles: 'test' (test images), integer (webcam), or file path.
    """
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
        "--model", "-m", type=str, required=True, help="Path to model weights (.pt)"
    )
    parser.add_argument(
        "--source", "-s", type=str, default="test",
        help="Input source: 'test' (test images), webcam index (0), or video path",
    )
    parser.add_argument(
        "--benchmark", "-b", action="store_true", help="Run speed benchmark"
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Disable visualization windows"
    )
    parser.add_argument(
        "--n-frames", type=int, default=100, help="Number of frames for benchmark"
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

    pipeline = InferencePipeline(model_path)

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
