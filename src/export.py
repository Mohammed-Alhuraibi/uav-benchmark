"""Export a trained model for Hailo-8L deployment.

Handles the first half of the deployment pipeline:
  PyTorch (.pt) → ONNX (.onnx) → [Hailo DFC] → HEF

This script covers PyTorch → ONNX export and validation.
The ONNX → HEF step requires the Hailo Dataflow Compiler (DFC)
which runs on an x86 machine, not on the Pi. Instructions are
printed via --hailo-guide.

Usage:
    python src/export.py --experiment yolo11s_p2
    python src/export.py --experiment yolo11s_p2 --validate
    python src/export.py --calibration-data
    python src/export.py --hailo-guide
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_deployment_config() -> dict:
    with open(ROOT / "configs" / "deployment.yaml") as f:
        return yaml.safe_load(f)


def find_best_weights(experiment: str) -> Path:
    weights = ROOT / "runs" / experiment / "weights" / "best.pt"
    if not weights.exists():
        weights = ROOT / "runs" / experiment / "weights" / "last.pt"
    if not weights.exists():
        raise FileNotFoundError(
            f"No weights found for '{experiment}' in {weights.parent}"
        )
    return weights


def export_onnx(experiment: str) -> Path:
    """Export trained model to ONNX format.

    Uses deployment.yaml for input size and inference settings.
    Returns path to the exported .onnx file.
    """
    from ultralytics import YOLO

    config = load_deployment_config()
    imgsz = config["inference"]["imgsz"]

    weights = find_best_weights(experiment)
    model = YOLO(str(weights))

    print(f"\n  Exporting: {experiment}")
    print(f"  Weights:   {weights}")
    print(f"  Format:    ONNX (opset 12)")
    print(f"  Input:     {imgsz}x{imgsz}")

    export_path = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=True,
        opset=12,
        dynamic=False,
    )

    export_path = Path(export_path)
    print(f"  Exported:  {export_path}")
    print(f"  Size:      {export_path.stat().st_size / 1e6:.1f} MB")

    return export_path


def validate_export(experiment: str) -> dict:
    """Compare PyTorch vs ONNX predictions on test images.

    Ensures the ONNX export didn't degrade accuracy. Runs both models
    on the same test images and compares detection counts and confidences.
    """
    from ultralytics import YOLO

    config = load_deployment_config()
    imgsz = config["inference"]["imgsz"]
    det_conf = config["inference"]["det_conf"]

    weights = find_best_weights(experiment)
    onnx_path = weights.with_suffix(".onnx")
    if not onnx_path.exists():
        print(f"  ONNX not found at {onnx_path}, exporting first...")
        onnx_path = export_onnx(experiment)

    # Load both models
    pt_model = YOLO(str(weights))
    onnx_model = YOLO(str(onnx_path))

    # Get test images
    test_dir = ROOT / "data" / "images" / "test"
    test_imgs = sorted(test_dir.glob("*"))
    if not test_imgs:
        print("  ERROR: No test images found")
        return {"error": "no test images"}

    # Use a consistent subset
    rng = random.Random(42)
    samples = rng.sample(test_imgs, min(30, len(test_imgs)))

    print(f"\n  Validating ONNX export ({len(samples)} test images)...")
    print(f"  PyTorch:  {weights.name}")
    print(f"  ONNX:     {onnx_path.name}")

    pt_total = 0
    onnx_total = 0
    conf_diffs = []

    for img_path in samples:
        pt_results = pt_model.predict(
            str(img_path), imgsz=imgsz, conf=det_conf, verbose=False
        )
        onnx_results = onnx_model.predict(
            str(img_path), imgsz=imgsz, conf=det_conf, verbose=False
        )

        pt_count = len(pt_results[0].boxes)
        onnx_count = len(onnx_results[0].boxes)
        pt_total += pt_count
        onnx_total += onnx_count

        # Compare confidences if both have detections
        if pt_count > 0 and onnx_count > 0:
            pt_conf = pt_results[0].boxes.conf.mean().item()
            onnx_conf = onnx_results[0].boxes.conf.mean().item()
            conf_diffs.append(abs(pt_conf - onnx_conf))

    avg_conf_diff = sum(conf_diffs) / len(conf_diffs) if conf_diffs else 0

    result = {
        "pt_detections": pt_total,
        "onnx_detections": onnx_total,
        "detection_match": pt_total == onnx_total,
        "avg_confidence_diff": round(avg_conf_diff, 4),
        "images_tested": len(samples),
    }

    # Report
    status = "PASS" if avg_conf_diff < 0.01 else "WARN"
    print(f"\n  Results:")
    print(f"    PyTorch detections:  {pt_total}")
    print(f"    ONNX detections:     {onnx_total}")
    print(f"    Avg confidence diff: {avg_conf_diff:.4f}")
    print(f"    Status:              [{status}]")

    if avg_conf_diff >= 0.01:
        print(f"    WARNING: Confidence drift > 0.01 — check ONNX export quality")

    # Save validation results
    eval_dir = ROOT / "runs" / experiment / "export"
    eval_dir.mkdir(parents=True, exist_ok=True)
    with open(eval_dir / "onnx_validation.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"    Saved:  {eval_dir / 'onnx_validation.json'}")

    return result


def generate_calibration_data() -> Path:
    """Copy a random subset of training images for Hailo INT8 calibration.

    Hailo DFC requires representative images to determine activation ranges
    for INT8 quantization. We sample from TRAINING data only (never val/test)
    to avoid data leakage.
    """
    config = load_deployment_config()
    n_images = config["hailo"]["calibration_images"]
    source = config["hailo"]["calibration_source"]

    src_dir = ROOT / "data" / "images" / source
    if not src_dir.exists():
        print(f"  ERROR: {src_dir} not found")
        sys.exit(1)

    all_images = sorted(src_dir.glob("*"))
    if len(all_images) < n_images:
        print(f"  WARNING: Only {len(all_images)} images available, using all")
        n_images = len(all_images)

    rng = random.Random(42)
    selected = rng.sample(all_images, n_images)

    out_dir = ROOT / "runs" / "calibration_data"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    for img in selected:
        shutil.copy2(img, out_dir / img.name)

    print(f"\n  Calibration dataset generated:")
    print(f"    Source:   {src_dir}")
    print(f"    Images:   {n_images}")
    print(f"    Output:   {out_dir}")
    print(f"    Purpose:  Hailo DFC INT8 quantization calibration")

    return out_dir


def print_hailo_guide() -> None:
    """Print step-by-step instructions for ONNX → HEF conversion via Hailo DFC."""
    config = load_deployment_config()

    print(f"""
{'='*60}
 Hailo-8L Deployment Guide
{'='*60}

 Prerequisites:
   - Hailo Dataflow Compiler (DFC) installed on an x86 machine
   - ONNX model exported via: python src/export.py --experiment <name>
   - Calibration data generated via: python src/export.py --calibration-data

 Pipeline: ONNX → HAR → HEF

 Step 1: Parse ONNX to HAR (Hailo Archive)
 ──────────────────────────────────────────
   from hailo_sdk_client import ClientRunner

   runner = ClientRunner(hw_arch="hailo8l")
   hn, npz = runner.translate_onnx_model(
       "runs/<experiment>/weights/best.onnx",
       net_name="uav_detector",
       start_node_names=["images"],
       end_node_names=["output0"],
   )

 Step 2: Quantize to INT8
 ────────────────────────
   # Point to calibration images
   runner.load_model_script(\"\"\"
       quantization_param(conv*, precision_mode=a{quant})
   \"\"\".format(quant="{config['hailo']['quantization']}"))

   runner.optimize(
       calib_dataset="runs/calibration_data/",
       data_type="{config['hailo']['quantization']}",
   )

 Step 3: Compile to HEF
 ───────────────────────
   hef = runner.compile()
   with open("runs/<experiment>/weights/best.hef", "wb") as f:
       f.write(hef)

 Step 4: Deploy on Pi 5
 ──────────────────────
   Copy the .hef file to the Raspberry Pi 5.
   Use hailort Python API or the community pipeline:
   https://github.com/DanielDubworsky/yolo26_hailo

 Hardware Specs:
   Target:      {config['hailo']['target']} (13 TOPS)
   Quantization: {config['hailo']['quantization']}
   Input size:   {config['inference']['imgsz']}x{config['inference']['imgsz']}
   Camera:       {config['camera']['sensor']} @ {config['camera']['resolution']}

 Expected Performance (YOLO26s on Hailo-8L):
   ~37.5 FPS @ 640x640, ~88.9% INT8 accuracy retention
{'='*60}
""")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export model for Hailo deployment")
    parser.add_argument("--experiment", "-e", type=str, help="Experiment to export")
    parser.add_argument(
        "--validate", action="store_true", help="Validate ONNX vs PyTorch predictions"
    )
    parser.add_argument(
        "--calibration-data",
        action="store_true",
        help="Generate calibration dataset for Hailo INT8 quantization",
    )
    parser.add_argument(
        "--hailo-guide",
        action="store_true",
        help="Print Hailo DFC conversion instructions",
    )
    args = parser.parse_args()

    if args.hailo_guide:
        print_hailo_guide()
        return 0

    if args.calibration_data:
        generate_calibration_data()
        return 0

    if not args.experiment:
        print(
            "ERROR: --experiment required (or use --calibration-data / --hailo-guide)",
            file=sys.stderr,
        )
        return 1

    export_onnx(args.experiment)

    if args.validate:
        validate_export(args.experiment)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
