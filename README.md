# UAV Detection Model Benchmark

Comparative benchmark of YOLO architectures for the TeknoFest Fighter UAV competition.
Evaluates small-object detection performance on the MMFW-UAV dataset targeting
deployment on Raspberry Pi 5 + Hailo-8L (13 TOPS, INT8).

## Models Under Test

| ID | Model | Small-Object Strategy |
|----|-------|-----------------------|
| 1  | YOLOv11s | Baseline (stride 8/16/32) |
| 2  | YOLOv11s + P2 | Architectural — stride-4 detection head |
| 3  | YOLO26s | Training — STAL + ProgLoss |
| 4  | YOLO26n | Same as YOLO26s, speed trade-off |

## Quick Start

```bash
# 1. Clone
git clone <repo-url>
cd uav-benchmark

# 2. Setup environment (auto-detects GPU, installs deps)
bash setup.sh

# 3. Add dataset
#    Download from cloud and place in data/
#    See data/README.md for expected structure

# 4. Run full benchmark
python src/benchmark.py

# Or run a single experiment
python src/train.py --experiment yolo11s_baseline
python src/evaluate.py --experiment yolo11s_baseline
```

## Project Structure

```
uav-benchmark/
├── configs/
│   ├── dataset.yaml            # Data paths for Ultralytics
│   ├── augmentation.yaml       # Tuned augmentation (shared across all experiments)
│   ├── albumentations.yaml     # Custom transforms: CLAHE, blur, noise
│   └── experiments.yaml        # All experiment definitions
├── models/
│   └── yolo11s-p2.yaml         # Custom YOLOv11s + P2 head architecture
├── data/                       # Dataset (gitignored, populate on target machine)
├── src/
│   ├── benchmark.py            # Main orchestrator
│   ├── train.py                # Single experiment trainer
│   ├── evaluate.py             # Test evaluation with competition-relevant metrics
│   ├── report.py               # Comparison tables + visualizations
│   └── albumentations_config.py
├── docs/                       # Competition specification documents
├── runs/                       # Training outputs (gitignored)
└── reports/                    # Generated benchmark reports
```

## Hardware Requirements

- NVIDIA GPU with 4+ GB VRAM (RTX 3050 minimum, RTX 4060+ recommended)
- 16+ GB RAM
- Ubuntu 22.04 / 24.04 LTS
- CUDA 12.x compatible driver

## Dataset

MMFW-UAV derived — 3,558 matched image/label pairs, single class (Fixed_Wing_UAV).
Split: 80/10/10 (2846 train / 358 val / 354 test), stratified by sortie bucket.
See `data/README.md` for setup instructions.

## Outputs

After benchmark completes, find results in `reports/`:
- `comparison_table.csv` — all models x all metrics
- `loss_curves.png` — overlaid training loss curves
- `visual_grid.png` — same test images, predictions from all models
- `precision_recall.png` — overlaid P/R curves
- `per_bucket_heatmap.png` — per-sortie mAP breakdown
- `benchmark_summary.md` — formatted summary with decision recommendation
