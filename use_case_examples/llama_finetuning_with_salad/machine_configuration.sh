#!/usr/bin/env bash
set -e

# --- Create environment
echo "🐍 Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# --- Install Some Dependancies
echo "🔧 Installing dependencies..."
apt-get update
apt-get install -y curl
apt-get install vim
apt-get install -y docker.io

# --- Clone CML
echo "📥 Cloning Concrete-ML repository..."
GH_TOKEN=$(python3 -c "from my_secrets import GH_TOKEN; print(GH_TOKEN)")
git clone https://$GH_TOKEN@github.com/zama-ai/concrete-ml.git

cd concrete-ml
git checkout llm_finetuning_on_salad
pip install -e .

# --- Install Concrete GPU ---
CONCRETE_WITH_VERSION="concrete-python==2.10.0"
echo "Installing $CONCRETE_WITH_VERSION from Zama GPU index..."
pip install --upgrade pip
pip install --extra-index-url https://pypi.zama.ai/gpu "$CONCRETE_WITH_VERSION"

# --- Verify GPU with nvidia-smi ---
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Ensure NVIDIA drivers are installed!"
    exit 1
fi

echo "nvidia-smi output:"
nvidia-smi

# --- Validate CUDA & Concrete GPU from Python ---
echo "Running GPU checks..."
python <<EOF
import torch
from concrete.compiler import check_gpu_available, check_gpu_enabled

assert torch.cuda.is_available(), "❌ Torch CUDA is NOT available!"
print("✅ CUDA device:", torch.cuda.get_device_name(0))

assert check_gpu_available(), "❌ Concrete GPU is NOT available!"
assert check_gpu_enabled(), "❌ Concrete GPU is NOT enabled!"
print("✅ Concrete GPU checks passed successfully.")
EOF

echo "🎉 All GPU checks passed. Ready to go!"
