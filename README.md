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

# 4. Verify setup (GPU + dataset + configs)
python src/benchmark.py --verify-only

# 5. Run full benchmark (Round 1: model comparison)
python src/benchmark.py

# Or run a single experiment
python src/train.py --experiment yolo11s_baseline
python src/evaluate.py --experiment yolo11s_baseline
```

## Workflow

The benchmark follows a two-round approach:

### Round 1: Find the best model
All 4 models train with identical augmentation. Compares architectures only.
```bash
python src/benchmark.py
```

### Round 2: Find the best preprocessing
After Round 1, uncomment ablation experiments in `configs/experiments.yaml`,
replacing `WINNER.pt` with the Round 1 winner. Compares augmentation strategies.
```bash
python src/benchmark.py --start-from winner_no_aug
```

### Export for Hailo deployment
Export the winning model to ONNX and prepare for Hailo INT8 quantization.
```bash
python src/export.py --experiment <winner> --validate
python src/export.py --calibration-data
python src/export.py --hailo-guide
```

### Test inference on your PC
Test the model with webcam, video, or test images before deploying to the Pi.
```bash
# Test on test images (with visualization)
python src/inference.py --model runs/<winner>/weights/best.pt --source test

# Test with webcam (show plane images to camera)
python src/inference.py --model runs/<winner>/weights/best.pt --source 0

# Benchmark inference speed
python src/inference.py --model runs/<winner>/weights/best.pt --benchmark
```

## Project Structure

```
uav-benchmark/
├── configs/
│   ├── dataset.yaml              # Data paths for Ultralytics
│   ├── augmentation.yaml         # Training augmentation (tuned for small objects)
│   ├── augmentation_none.yaml    # Zero augmentation (Round 2 ablation baseline)
│   ├── albumentations.yaml       # Custom transforms: CLAHE, blur, noise
│   ├── experiments.yaml          # All experiment definitions + Round 2 templates
│   └── deployment.yaml           # Camera, Hailo, and competition rule specs
├── models/
│   └── yolo11s-p2.yaml           # Custom YOLOv11s + P2 head architecture
├── data/                         # Dataset (gitignored, populate on target machine)
├── src/
│   ├── benchmark.py              # Main orchestrator (verify → train → eval → report)
│   ├── train.py                  # Single experiment trainer
│   ├── evaluate.py               # Test evaluation with competition-relevant metrics
│   ├── report.py                 # Comparison tables + visualizations
│   ├── export.py                 # ONNX export + Hailo calibration + deployment guide
│   ├── inference.py              # Reference Pi inference pipeline + webcam testing
│   └── albumentations_config.py  # Custom Albumentations transform loader
├── docs/                         # Competition specification documents
├── runs/                         # Training outputs (gitignored)
└── reports/                      # Generated benchmark reports
```

## Hardware

### Training / Benchmarking
- NVIDIA GPU with 4+ GB VRAM (recommended) or CPU-only (slower)
- 16+ GB RAM
- Ubuntu 22.04 / 24.04 LTS

### Deployment Target
- Raspberry Pi 5 + Hailo-8L (13 TOPS, INT8)
- Pi Camera Module 3 (IMX708, 2304x1296 @ 56fps)

## Dataset

MMFW-UAV derived — 3,558 matched image/label pairs, single class (Fixed_Wing_UAV).
Split: 80/10/10 (2846 train / 358 val / 354 test), stratified by sortie bucket.
See `data/README.md` for setup instructions.

## Outputs

After benchmark completes, find results in `reports/`:
- `comparison_table.csv` — all models x all metrics
- `loss_curves.png` — overlaid training loss curves
- `metric_curves.png` — mAP progression across epochs
- `visual_grid.png` — same test images, predictions from all models
- `benchmark_summary.md` — formatted summary with key findings
