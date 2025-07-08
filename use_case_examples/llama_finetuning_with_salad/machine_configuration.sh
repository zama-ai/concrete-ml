#!/usr/bin/env bash
set -e

# --- Create environment
echo "🐍 Creating virtual environment..."
apt install -y python3.10-venv
python3 -m venv .venv
source .venv/bin/activate

# --- Install some dependancies
echo "🔧 Installing dependencies..."
apt-get update
apt-get install -y curl
apt-get install vim

if ! command -v docker &> /dev/null; then
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
fi

cd ../..

pip install -e .

# --- Install Concrete GPU ---
CONCRETE_WITH_VERSION="concrete-python==2.10.0"
pip uninstall -y concrete-python
echo "Installing $CONCRETE_WITH_VERSION from Zama GPU index..."
pip install --upgrade pip
pip install --extra-index-url https://pypi.zama.ai/gpu "$CONCRETE_WITH_VERSION"
pip install typing-extensions

cd use_case_examples/llama_finetuning_with_salad
pip install -r requirements_server.txt

# --- Verify GPU with nvidia-smi ---
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Ensure NVIDIA drivers are installed!"
    exit 1
fi

echo "nvidia-smi output:"
nvidia-smi

# --- Validate CUDA & Concrete GPU from Python ---
echo "Running GPU checks..."
python3 <<EOF
import torch
from concrete.compiler import check_gpu_available, check_gpu_enabled

assert torch.cuda.is_available(), "❌ Torch CUDA is NOT available!"
print("✅ CUDA device:", torch.cuda.get_device_name(0))

assert check_gpu_available(), "❌ Concrete GPU is NOT available!"
assert check_gpu_enabled(), "❌ Concrete GPU is NOT enabled!"
print("✅ Concrete GPU checks passed successfully.")
EOF

echo "🎉 All GPU checks passed. Ready to go!"
