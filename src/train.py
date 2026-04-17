"""Train a single YOLO experiment.

Reads experiment definitions from configs/experiments.yaml and applies
augmentation config from configs/augmentation.yaml (or a per-experiment
override via the 'augmentation' field).

Priority order: defaults → augmentation config → experiment overrides.
Experiment-level values always win, enabling ablation studies.

Usage:
    python src/train.py --experiment yolo11s_baseline
    python src/train.py --experiment yolo26s
    python src/train.py --list  # show available experiments
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config() -> dict:
    with open(ROOT / "configs" / "experiments.yaml") as f:
        return yaml.safe_load(f)


def load_augmentation(aug_path: str | None = None) -> dict:
    path = ROOT / (aug_path or "configs/augmentation.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def list_experiments(config: dict) -> None:
    print("Available experiments:")
    for name, exp in config["experiments"].items():
        desc = exp.get("description", "")
        print(f"  {name:<25s} {desc}")


def resolve_model_path(model: str) -> str:
    """Resolve model path: stock names pass through, local paths get resolved."""
    if model.endswith(".pt"):
        return model  # stock pretrained — Ultralytics downloads automatically
    local = ROOT / model
    if local.exists():
        return str(local)
    return model


def train_experiment(name: str, config: dict) -> Path:
    """Train a single experiment. Returns path to the run directory."""
    from ultralytics import YOLO

    # Register custom albumentations before training
    from src.albumentations_config import register_albumentations
    register_albumentations()

    defaults = config["defaults"]
    exp = config["experiments"][name]

    # Build train args: defaults → augmentation → experiment overrides
    # This order ensures experiment-level values always win (for ablation)
    train_args = {**defaults}

    # Load augmentation config (experiment can specify a different file)
    aug_path = exp.get("augmentation")
    aug = load_augmentation(aug_path)
    train_args.update(aug)

    # Experiment overrides go LAST so they win over augmentation defaults
    skip_keys = ("description", "strategy", "pretrained", "augmentation")
    for k, v in exp.items():
        if k not in skip_keys:
            train_args[k] = v

    # Resolve paths
    model_path = resolve_model_path(train_args.pop("model"))

    # Resolve dataset YAML with absolute path so YOLO doesn't depend on CWD.
    # dataset.yaml uses 'path: ../data' (relative to yaml location), but YOLO
    # resolves it relative to CWD which varies across environments (Colab, SSH, etc).
    dataset_yaml = ROOT / train_args.pop("data")
    with open(dataset_yaml) as f:
        ds_config = yaml.safe_load(f)
    ds_config["path"] = str((dataset_yaml.parent / ds_config["path"]).resolve())
    resolved_yaml = ROOT / "configs" / ".dataset_resolved.yaml"
    with open(resolved_yaml, "w") as f:
        yaml.dump(ds_config, f)
    train_args["data"] = str(resolved_yaml)

    # Output config
    train_args["project"] = str(ROOT / "runs")
    train_args["name"] = name
    train_args["exist_ok"] = True

    # Load model
    if "pretrained" in exp:
        # Custom architecture with pretrained weight transfer
        model = YOLO(model_path)
        train_args["pretrained"] = exp["pretrained"]
    else:
        model = YOLO(model_path)

    print(f"\n{'='*60}")
    print(f" Training: {name}")
    print(f" Model:    {model_path}")
    print(f" {'='*60}\n")

    model.train(**train_args)

    run_dir = ROOT / "runs" / name
    print(f"\nTraining complete. Results in: {run_dir}")
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a YOLO experiment")
    parser.add_argument("--experiment", "-e", type=str, help="Experiment name")
    parser.add_argument("--list", "-l", action="store_true", help="List experiments")
    args = parser.parse_args()

    config = load_config()

    if args.list:
        list_experiments(config)
        return 0

    if not args.experiment:
        print("ERROR: --experiment required. Use --list to see options.", file=sys.stderr)
        return 1

    if args.experiment not in config["experiments"]:
        print(f"ERROR: unknown experiment '{args.experiment}'", file=sys.stderr)
        list_experiments(config)
        return 1

    train_experiment(args.experiment, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
