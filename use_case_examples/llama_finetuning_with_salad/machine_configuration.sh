#!/usr/bin/env bash
set -e

# --- 1. Install Some Dependancies
apt-get update
apt-get install -y curl
sudo apt install vim

# --- 2. Install Docker
apt-get install -y docker.io

# --- 3. Install Concrete GPU ---
CONCRETE_WITH_VERSION="concrete-python==2.10.0"
echo "Installing $CONCRETE_WITH_VERSION from Zama GPU index..."
pip install --upgrade pip
pip install --extra-index-url https://pypi.zama.ai/gpu "$CONCRETE_WITH_VERSION"

# --- 4. Verify GPU with nvidia-smi ---
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Ensure NVIDIA drivers are installed!"
    exit 1
fi

echo "nvidia-smi output:"
nvidia-smi

# --- 6. Validate CUDA & Concrete GPU from Python ---
echo "Running GPU checks..."
python <<EOF
import torch
from concrete.compiler import check_gpu_available, check_gpu_enabled

assert torch.cuda.is_available(), "Torch CUDA is NOT available!"
print("CUDA device:", torch.cuda.get_device_name(0))

assert check_gpu_available(), "Concrete GPU is NOT available!"
assert check_gpu_enabled(), "Concrete GPU is NOT enabled!"
print("Concrete GPU checks passed successfully.")
EOF

echo "✅ Configuration completed successfully."
