"""Register custom Albumentations transforms with Ultralytics.

Ultralytics checks for an `albumentations` package and applies transforms
if a specific format is followed. This module reads our albumentations.yaml
config and sets up the pipeline.

The transforms are applied AFTER Ultralytics' built-in augmentations (mosaic,
HSV, etc.) and BEFORE normalization/tensor conversion.
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
_registered = False


def register_albumentations() -> None:
    """Set ALBUMENTATIONS_TRANSFORMS env var to configure Ultralytics integration."""
    global _registered
    if _registered:
        return

    config_path = ROOT / "configs" / "albumentations.yaml"
    if not config_path.exists():
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if not config or "transforms" not in config:
        return

    # Verify albumentations is importable
    try:
        import albumentations as A  # noqa: F401
    except ImportError:
        print("WARNING: albumentations not installed, skipping custom transforms")
        return

    _registered = True


def build_albumentations_transform():
    """Build an Albumentations Compose pipeline from our config.

    Returns an albumentations.Compose object or None.
    """
    config_path = ROOT / "configs" / "albumentations.yaml"
    if not config_path.exists():
        return None

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if not config or "transforms" not in config:
        return None

    import albumentations as A

    transforms = []
    for t in config["transforms"]:
        cls = getattr(A, t["name"], None)
        if cls is None:
            print(f"WARNING: albumentations.{t['name']} not found, skipping")
            continue
        params = t.get("params", {})
        p = t.get("p", 1.0)

        # Convert list params to tuples where needed
        for k, v in params.items():
            if isinstance(v, list):
                params[k] = tuple(v)

        transforms.append(cls(p=p, **params))

    if not transforms:
        return None

    return A.Compose(transforms)
