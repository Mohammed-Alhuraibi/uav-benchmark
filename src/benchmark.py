"""Main benchmark orchestrator.

Runs all experiments end-to-end: verify → train → evaluate → report.
Single command: python src/benchmark.py

Supports resuming from a specific experiment if interrupted:
    python src/benchmark.py --start-from yolo26s

And verify-only mode to check GPU + dataset before committing to training:
    python src/benchmark.py --verify-only
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def verify_gpu() -> dict:
    """Check GPU availability and print summary."""
    info = {"cuda": torch.cuda.is_available()}
    if info["cuda"]:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["vram_gb"] = round(getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9, 1)
        info["cuda_version"] = torch.version.cuda
        print(f"  GPU:  {info['gpu_name']}")
        print(f"  VRAM: {info['vram_gb']} GB")
        print(f"  CUDA: {info['cuda_version']}")
    else:
        print("  WARNING: No CUDA GPU detected. Training will be SLOW.")
    return info


def verify_dataset() -> bool:
    """Check that data/ has the expected structure and counts."""
    data_dir = ROOT / "data"
    ok = True

    for split in ("train", "val", "test"):
        imgs = list((data_dir / "images" / split).glob("*"))
        lbls = list((data_dir / "labels" / split).glob("*.txt"))
        img_stems = {p.stem for p in imgs}
        lbl_stems = {p.stem for p in lbls}

        matched = len(img_stems & lbl_stems)
        orphan_imgs = len(img_stems - lbl_stems)
        orphan_lbls = len(lbl_stems - img_stems)

        status = "OK" if orphan_imgs == 0 and orphan_lbls == 0 else "WARN"
        print(f"  {split:<6s}: {matched:>5d} pairs, {orphan_imgs} orphan imgs, {orphan_lbls} orphan lbls [{status}]")

        if matched == 0:
            print(f"  ERROR: {split} has zero matched pairs!")
            ok = False

    return ok


def verify_configs() -> bool:
    """Check that all config files parse and reference valid paths."""
    for cfg_name in ("experiments.yaml", "augmentation.yaml", "albumentations.yaml", "dataset.yaml"):
        cfg_path = ROOT / "configs" / cfg_name
        if not cfg_path.exists():
            print(f"  ERROR: {cfg_path} not found")
            return False
        try:
            with open(cfg_path) as f:
                yaml.safe_load(f)
            print(f"  {cfg_name}: OK")
        except Exception as e:
            print(f"  {cfg_name}: PARSE ERROR — {e}")
            return False

    # Check model YAMLs referenced in experiments
    with open(ROOT / "configs" / "experiments.yaml") as f:
        config = yaml.safe_load(f)
    for name, exp in config["experiments"].items():
        model = exp["model"]
        if not model.endswith(".pt"):
            model_path = ROOT / model
            if not model_path.exists():
                print(f"  ERROR: model config '{model}' not found at {model_path}")
                return False
            print(f"  {model}: OK")

    return True


def run_benchmark(start_from: str | None = None) -> None:
    """Run the full benchmark pipeline."""
    from src.train import train_experiment, load_config
    from src.evaluate import evaluate_experiment
    from src.report import generate_report

    config = load_config()
    experiments = list(config["experiments"].keys())

    # Handle --start-from
    if start_from:
        if start_from not in experiments:
            print(f"ERROR: '{start_from}' not in experiments: {experiments}", file=sys.stderr)
            sys.exit(1)
        idx = experiments.index(start_from)
        experiments = experiments[idx:]
        print(f"Resuming from: {start_from} (skipping {idx} experiments)")

    total_start = time.time()

    # Phase 1: Train all experiments
    print(f"\n{'='*60}")
    print(f" PHASE 1: TRAINING ({len(experiments)} experiments)")
    print(f"{'='*60}")

    for i, name in enumerate(experiments, 1):
        print(f"\n--- Experiment {i}/{len(experiments)}: {name} ---")
        exp_start = time.time()

        # Skip if already trained
        weights = ROOT / "runs" / name / "weights" / "best.pt"
        if weights.exists():
            print(f"  Weights already exist at {weights}, skipping training.")
            print(f"  (Delete runs/{name}/ to retrain)")
            continue

        train_experiment(name, config)
        elapsed = (time.time() - exp_start) / 60
        print(f"  Completed in {elapsed:.1f} minutes")

    # Phase 2: Evaluate all experiments
    print(f"\n{'='*60}")
    print(f" PHASE 2: EVALUATION")
    print(f"{'='*60}")

    all_results = {}
    all_experiments = list(config["experiments"].keys())
    for name in all_experiments:
        if (ROOT / "runs" / name / "weights").exists():
            all_results[name] = evaluate_experiment(name, config)

    # Phase 3: Generate report
    print(f"\n{'='*60}")
    print(f" PHASE 3: REPORT GENERATION")
    print(f"{'='*60}")

    generate_report(all_results, config)

    total_elapsed = (time.time() - total_start) / 3600
    print(f"\n{'='*60}")
    print(f" BENCHMARK COMPLETE — {total_elapsed:.1f} hours total")
    print(f" Reports in: {ROOT / 'reports'}")
    print(f"{'='*60}")


def main() -> int:
    parser = argparse.ArgumentParser(description="UAV Detection Model Benchmark")
    parser.add_argument("--start-from", type=str, help="Resume from this experiment")
    parser.add_argument("--verify-only", action="store_true", help="Only verify setup, don't train")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f" UAV Detection Benchmark")
    print(f"{'='*60}")

    print("\n[1/3] Verifying GPU...")
    gpu_info = verify_gpu()

    print("\n[2/3] Verifying dataset...")
    data_ok = verify_dataset()

    print("\n[3/3] Verifying configs...")
    configs_ok = verify_configs()

    if not data_ok:
        print("\nERROR: Dataset verification failed. See data/README.md for setup.")
        return 1

    if not configs_ok:
        print("\nERROR: Config verification failed.")
        return 1

    if args.verify_only:
        print("\nVerification passed. Ready to benchmark.")
        return 0

    run_benchmark(start_from=args.start_from)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
