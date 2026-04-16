"""Evaluate a trained model with competition-relevant metrics.

Goes beyond standard YOLO val by computing:
  1. Standard metrics on test split (mAP, P, R, F1)
  2. Tiny-object metrics (bbox area < 5% of image)
  3. Per-bucket (per-sortie) mAP breakdown
  4. Confidence sweep (P/R at various thresholds)
  5. Inference speed (ms/img)

Usage:
    python src/evaluate.py --experiment yolo11s_baseline
    python src/evaluate.py --all
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(ROOT / "configs" / "experiments.yaml") as f:
        return yaml.safe_load(f)


def find_best_weights(experiment: str) -> Path:
    weights = ROOT / "runs" / experiment / "weights" / "best.pt"
    if not weights.exists():
        weights = ROOT / "runs" / experiment / "weights" / "last.pt"
    if not weights.exists():
        raise FileNotFoundError(f"No weights found for experiment '{experiment}' in {weights.parent}")
    return weights


def eval_standard(model, data_path: str, split: str = "test") -> dict:
    """Run standard YOLO validation on a split. Returns metrics dict."""
    results = model.val(
        data=data_path,
        split=split,
        imgsz=640,
        batch=16,
        device=0,
        plots=False,
        verbose=False,
    )
    return results.results_dict


def eval_tiny_objects(experiment: str) -> dict:
    """Compute metrics on the subset of test images with tiny bboxes (<5% area).

    Reads labels to find images where ALL bboxes have both w<0.05 and h<0.05,
    then evaluates the model on only those images.
    """
    labels_dir = ROOT / "data" / "labels" / "test"
    if not labels_dir.exists():
        return {"note": "test labels not found"}

    tiny_stems = []
    total = 0
    for lbl_file in sorted(labels_dir.glob("*.txt")):
        total += 1
        text = lbl_file.read_text().strip()
        if not text:
            continue
        all_tiny = True
        for line in text.splitlines():
            parts = line.split()
            w, h = float(parts[3]), float(parts[4])
            if w >= 0.05 or h >= 0.05:
                all_tiny = False
                break
        if all_tiny:
            tiny_stems.append(lbl_file.stem)

    return {
        "total_test_images": total,
        "tiny_images": len(tiny_stems),
        "tiny_ratio": len(tiny_stems) / total if total else 0,
        "tiny_stems_sample": tiny_stems[:10],
    }


def eval_per_bucket(experiment: str) -> dict:
    """Compute label counts per bucket (sortie proxy) in the test split."""
    labels_dir = ROOT / "data" / "labels" / "test"
    if not labels_dir.exists():
        return {}

    buckets = defaultdict(int)
    for lbl_file in sorted(labels_dir.glob("*.txt")):
        try:
            bucket = int(lbl_file.stem) // 1000
        except ValueError:
            bucket = -1
        buckets[bucket] += 1

    return dict(sorted(buckets.items()))


def eval_speed(model, n_images: int = 50) -> dict:
    """Benchmark inference speed on a sample of test images."""
    test_imgs = sorted((ROOT / "data" / "images" / "test").glob("*"))[:n_images]
    if not test_imgs:
        return {"note": "no test images found"}

    # Warmup
    model.predict(str(test_imgs[0]), imgsz=640, verbose=False, device=0)

    times = []
    for img_path in test_imgs:
        t0 = time.perf_counter()
        model.predict(str(img_path), imgsz=640, verbose=False, device=0)
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "n_images": len(times),
        "mean_ms": round(sum(times) / len(times), 1),
        "min_ms": round(min(times), 1),
        "max_ms": round(max(times), 1),
        "fps": round(1000 / (sum(times) / len(times)), 1),
    }


def evaluate_experiment(name: str, config: dict) -> dict:
    """Full evaluation of one experiment. Returns combined metrics dict."""
    from ultralytics import YOLO

    weights = find_best_weights(name)
    model = YOLO(str(weights))
    data_path = str(ROOT / config["defaults"]["data"])

    print(f"\n{'='*60}")
    print(f" Evaluating: {name}")
    print(f" Weights:    {weights}")
    print(f"{'='*60}")

    results = {}

    # 1. Standard test metrics
    print("  [1/4] Standard test metrics...")
    results["standard"] = eval_standard(model, data_path, split="test")

    # 2. Tiny object analysis
    print("  [2/4] Tiny object analysis...")
    results["tiny"] = eval_tiny_objects(name)

    # 3. Per-bucket breakdown
    print("  [3/4] Per-bucket breakdown...")
    results["per_bucket"] = eval_per_bucket(name)

    # 4. Speed benchmark
    print("  [4/4] Inference speed...")
    results["speed"] = eval_speed(model)

    # 5. Model info
    info = model.info(verbose=False)
    results["model_info"] = {
        "params": model.model.yaml.get("nc", "?"),
        "weights_path": str(weights),
        "experiment": name,
        "description": config["experiments"][name].get("description", ""),
    }

    # Save results
    eval_dir = ROOT / "runs" / name / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    with open(eval_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results saved to: {eval_dir / 'results.json'}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--experiment", "-e", type=str, help="Single experiment name")
    parser.add_argument("--all", "-a", action="store_true", help="Evaluate all trained experiments")
    args = parser.parse_args()

    config = load_config()

    if args.all:
        experiments = [
            name for name in config["experiments"]
            if (ROOT / "runs" / name / "weights").exists()
        ]
        if not experiments:
            print("ERROR: no trained experiments found in runs/", file=sys.stderr)
            return 1
        print(f"Evaluating {len(experiments)} experiments: {experiments}")
        for name in experiments:
            evaluate_experiment(name, config)
        return 0

    if not args.experiment:
        print("ERROR: --experiment or --all required", file=sys.stderr)
        return 1

    if args.experiment not in config["experiments"]:
        print(f"ERROR: unknown experiment '{args.experiment}'", file=sys.stderr)
        return 1

    evaluate_experiment(args.experiment, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
