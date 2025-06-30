from typing import Union
from pathlib import Path

import numpy
import torch

from fastapi import HTTPException

torch.set_printoptions(precision=10, sci_mode=False)

# Path configuration

COMPILED_MODELS_PATH = Path("compiled_models")

SERVER_DIR = Path(COMPILED_MODELS_PATH) / "server"
KEY_PATH   = SERVER_DIR / "serialized_key.bin"

ENCRYPTED_FILENAME_INPUT = "encrypted_input.bin"
CLEAR_FILENAME_INPUT = "clear_input.npy"

ENCRYPTED_FILENAME_OUTPUT  = "encrypted_output.bin"
CLEAR_FILENAME_OUTPUT  = "clear_output.bin"

FILENAME_WEIGHTS_FORMAT = "remote_weights"
FILENAME_WEIGHTS_EXTENSION = "npy"

FILENAME_BIAS    = "remote_bias.pth"
FILENAME_INFO    = "information.json"

DEVICE = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

print(f'Device: {DEVICE}')

def fetch_remote_weights(layer_dir: Union[str, Path]) -> Path:
    """Fetch remote weights given a layer_dir."""

    layer_dir = Path(layer_dir)

    pattern = f"{FILENAME_WEIGHTS_FORMAT}*.{FILENAME_WEIGHTS_EXTENSION}"
    candidates = list(layer_dir.glob(pattern))

    if not candidates:
        raise HTTPException(
            status_code=404,
            detail=f"No weight file matching pattern '{pattern}' in `{layer_dir}`"
        )

    if len(candidates) > 1:
        raise HTTPException(
            status_code=400,
            detail=f"Multiple weight files matching pattern '{pattern}' in` {layer_dir}`: `{[str(p) for p in candidates]}`"
        )

    return candidates[0]


def per_channel_weight_quantization(weight: numpy.ndarray, n_bits: int = 7):
    """Quantize the weights, per-channel using symmetric (signed) quantization.

    Args:
        weight: Weight tensor to quantize
        q_module: Quantized module containing quantization parameters

    Returns:
        tuple: Quantized weights, scale, zero point and weight sum
    """

    weight_float = torch.from_numpy(weight).to(DEVICE)

    q_min, q_max =  -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1

    w_min_vals, _ = weight_float.min(dim=0, keepdim=True)
    w_max_vals, _ = weight_float.max(dim=0, keepdim=True)

    weight_scale = (w_max_vals - w_min_vals) / (q_max - q_min)
    # Avoid division by zero.
    weight_scale = torch.where(
        (w_max_vals > w_min_vals),
        weight_scale,
        torch.ones_like(weight_scale, device=DEVICE)
    )
    weight_scale = weight_scale.squeeze(-1)  # shape: (out_dim,)
    # Quantization
    weight_zp = torch.round(
        q_min - w_min_vals / weight_scale
    ).to(dtype=torch.float32, device=DEVICE)


    # Apply quantization with proper broadcasting
    weight_q = torch.round(weight_float / weight_scale) + weight_zp
    weight_q = torch.clamp(weight_q, q_min, q_max).to(dtype=torch.float32, device=DEVICE)
    sum_w = weight_q.sum(dim=0)  # sum over the input dimension

    return weight_q, weight_scale, weight_zp, sum_w

