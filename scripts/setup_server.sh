#!/usr/bin/env bash
# Server one-time setup script for BalaRL training.
# Run this ONCE after cloning the repo on your server.
#
# Usage:
#   chmod +x scripts/setup_server.sh
#   ./scripts/setup_server.sh

set -e

echo "=== BalaRL Server Setup ==="

# 1. Install uv (fast Python package manager)
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# 2. Create virtual environment
echo "Creating virtual environment..."
uv venv --python 3.11

# 3. Install dependencies
echo "Installing dependencies..."
uv pip install -e ".[training]"

# 4. Detect GPU and install appropriate torch
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected! Installing CUDA-enabled PyTorch..."
    uv pip install torch --index https://download.pytorch.org/whl/cu121
else
    echo "No GPU detected, using CPU PyTorch..."
    uv pip install torch --index https://download.pytorch.org/whl/cpu
fi

echo ""
echo "=== Setup complete! ==="
echo "Run training with:  ./scripts/run_train.sh"
