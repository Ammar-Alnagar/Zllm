#!/bin/bash
# =============================================================================
# setup_env.sh - Environment Setup Script
# =============================================================================
# Run this script to set up the development environment.
# Usage: source scripts/setup_env.sh
# =============================================================================

set -e  # Exit on error

echo "=== Mini-vLLM Environment Setup ==="

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Check Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" < "3.10" ]]; then
    echo "Error: Python 3.10+ required"
    exit 1
fi

# Create virtual environment if it doesn't exist
VENV_DIR="$PROJECT_DIR/.venv"
if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
echo "Virtual environment activated: $VIRTUAL_ENV"

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
echo "Installing dependencies..."
pip install tiktoken>=0.5.0
pip install fastapi>=0.100.0
pip install uvicorn>=0.23.0
pip install pybind11>=2.10.0
pip install numpy>=1.24.0
pip install requests>=2.31.0
pip install ninja  # For faster builds

# Verify CUDA
echo ""
echo "=== Verification ==="
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

# Set environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo ""
echo "=== Environment Ready ==="
echo "To activate: source $VENV_DIR/bin/activate"
echo "To build: cd $PROJECT_DIR && pip install -e ."
