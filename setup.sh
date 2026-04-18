#!/usr/bin/env bash
# Environment setup for UAV Detection Benchmark
# Auto-detects GPU, installs correct PyTorch variant, verifies everything works.
#
# Usage: bash setup.sh

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$REPO_DIR/.venv"
PYTHON="${VENV_DIR}/bin/python"
PIP="${VENV_DIR}/bin/pip"

echo "========================================"
echo " UAV Benchmark — Environment Setup"
echo "========================================"
echo ""

# --- Step 1: Create virtual environment ---
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/5] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
else
    echo "[1/5] Virtual environment already exists."
fi

echo "[2/5] Upgrading pip..."
"$PIP" install --quiet --upgrade pip

# --- Step 2: Detect GPU and CUDA ---
echo "[3/5] Detecting GPU..."

if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
    GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    CUDA_VERSION=$(nvidia-smi | grep -oP "CUDA Version: \K[0-9.]+" || echo "unknown")

    echo "  GPU:    $GPU_NAME"
    echo "  VRAM:   ${GPU_VRAM} MiB"
    echo "  Driver: $DRIVER_VERSION"
    echo "  CUDA:   $CUDA_VERSION"

    # Select PyTorch CUDA index based on detected CUDA version
    CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
    if [ "$CUDA_MAJOR" -ge 12 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu124"
        echo "  Using: PyTorch with CUDA 12.4"
    elif [ "$CUDA_MAJOR" -ge 11 ]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        echo "  Using: PyTorch with CUDA 11.8"
    else
        echo "  WARNING: CUDA $CUDA_VERSION is old. Falling back to CPU."
        TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    fi
else
    echo "  WARNING: nvidia-smi not found. Installing CPU-only PyTorch."
    GPU_NAME="none"
    GPU_VRAM="0"
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
fi

# --- Step 3: Install dependencies ---
echo "[4/5] Installing PyTorch..."
"$PIP" install --quiet torch torchvision --index-url "$TORCH_INDEX"

echo "       Installing remaining dependencies..."
"$PIP" install --quiet -r "$REPO_DIR/requirements.txt"

# --- Step 4: Verify installation ---
echo "[5/5] Verifying installation..."
"$PYTHON" -c "
import torch
import ultralytics
import albumentations

print(f'  torch:          {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU device:     {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    vram_gb = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / 1e9
    print(f'  VRAM:           {vram_gb:.1f} GB')
    # Auto-suggest batch size
    if vram_gb >= 8:
        batch = 32
    elif vram_gb >= 6:
        batch = 16
    else:
        batch = 8
    print(f'  Suggested batch: {batch}')
print(f'  ultralytics:    {ultralytics.__version__}')
print(f'  albumentations: {albumentations.__version__}')
print()
print('All checks passed.')
"

echo ""
echo "========================================"
echo " Setup complete!"
echo ""
echo " Next steps:"
echo "   1. Place dataset in data/ (see data/README.md)"
echo "   2. Run: $PYTHON src/benchmark.py"
echo "========================================"
