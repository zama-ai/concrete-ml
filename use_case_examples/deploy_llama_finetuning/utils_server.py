import csv
import re
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy
import pandas as pd
import torch
from fastapi import HTTPException

torch.set_printoptions(precision=10, sci_mode=False)
pd.set_option("display.max_columns", None)

# Path configuration
ROOT_SERVER_DIR = Path("./deployment_float")
COMPILED_MODELS_DIR = Path("compiled_models")
MODEL_NAME = "meta_llama"

ARCHIVE_PATH = ROOT_SERVER_DIR / "compiled_models.tar.gz"
TARGET_DIR = ROOT_SERVER_DIR / COMPILED_MODELS_DIR

SERVER_DIR = ROOT_SERVER_DIR / COMPILED_MODELS_DIR / "server"
KEY_PATH = ROOT_SERVER_DIR / COMPILED_MODELS_DIR / "Keys" / "serialized_key.bin"

BENCHMARK_DIR = ROOT_SERVER_DIR / Path("Logs")
BENCHMARK_FILE_PATH = BENCHMARK_DIR / "server_benchmark.csv"

ENCRYPTED_FILENAME_INPUT = "encrypted_input.bin"
CLEAR_FILENAME_INPUT = "clear_input.npy"

ENCRYPTED_FILENAME_OUTPUT = "encrypted_output.bin"
CLEAR_FILENAME_OUTPUT = "clear_output.bin"

FILENAME_WEIGHTS_FORMAT = "remote_weights"
FILENAME_WEIGHTS_Q_FORMAT = "remote_quantized_weights"
FILENAME_WEIGHTS_EXTENSION = "npy"

FILENAME_BIAS = "remote_bias.pth"
FILENAME_INFO = "information.json"

MACHINE = "g4dn.16xlarge"
DEVICE = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

print(f"Server: `device={DEVICE}`")

BENCHMARK_COLUMNS = [
    "endpoint",
    "date",
    "device",
    "machine",
    "uid",
    "layer_name",
    "index",
    "input_shape",
    "remote_weight_shape",
    "time_read_key",
    "time_serialization_key",
    "time_deserialization_key",
    "time_storage_key",
    "time_load_key",
    "time_read_input",
    "time_serialization_input",
    "time_deserialization_input",
    "time_storage_input",
    "time_load_input",
    "time_weight_quantization",
    "time_serialization_output",
    "time_storage_output",
    "time_matmul",
    "time_packing_output_response",
    "total_add_key_func",
    "total_send_input_func",
    "total_compute_func",
]


def init_benchmark_file(reset=False):
    BENCHMARK_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if reset and BENCHMARK_FILE_PATH.exists():
        BENCHMARK_FILE_PATH.unlink()
        print(f"🗑️ Existing benchmark file deleted: `{BENCHMARK_FILE_PATH.resolve()}`")

    if not BENCHMARK_FILE_PATH.exists():
        with BENCHMARK_FILE_PATH.open("w", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            writer.writerow(BENCHMARK_COLUMNS)


def save_benchmark_row(
    data: dict,
):
    invalid_keys = set(data.keys()) - set(BENCHMARK_COLUMNS)
    if invalid_keys:
        raise ValueError(
            f"❌ Invalid column(s) in data: {invalid_keys}\n"
            f"✅ Allowed columns: {BENCHMARK_COLUMNS}"
        )

    row = [
        data.get("endpoint"),
        str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        str(DEVICE),
        str(MACHINE),
        data.get("uid"),
        data.get("layer_name"),
        data.get("index"),
        data.get("input_shape"),
        data.get("remote_weight_shape"),
        data.get("time_read_key"),
        data.get("time_serialization_key"),
        data.get("time_deserialization_key"),
        data.get("time_storage_key"),
        data.get("time_load_key"),
        data.get("time_read_input"),
        data.get("time_serialization_input"),
        data.get("time_deserialization_input"),
        data.get("time_storage_input"),
        data.get("time_load_input"),
        data.get("time_weight_quantization"),
        data.get("time_serialization_output"),
        data.get("time_storage_output"),
        data.get("time_matmul"),
        data.get("time_packing_output_response"),
        data.get("total_add_key_func"),
        data.get("total_send_input_func"),
        data.get("total_compute_func"),
    ]

    with BENCHMARK_FILE_PATH.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=";")
        writer.writerow(row)

    print(f"✅ Benchmark saved at `{BENCHMARK_FILE_PATH.resolve()}`")


def read_csv(path):
    return pd.read_csv(path, sep=";")


def fetch_remote_weights(
    layer_dir: Union[str, Path], filename_weight_format=FILENAME_WEIGHTS_FORMAT
) -> Path:
    """Fetch remote weights given a layer_dir."""

    layer_dir = Path(layer_dir).resolve()
    root_dir = ROOT_SERVER_DIR.resolve()

    # Validate that layer_dir is within the safe root directory
    if not str(layer_dir).startswith(str(root_dir)):
        raise HTTPException(
            status_code=403, detail=f"Access to the directory `{layer_dir}` is not allowed."
        )

    # Validate that layer_dir is within the safe root directory
    if not str(layer_dir).startswith(str(ROOT_SERVER_DIR.resolve())):
        raise HTTPException(
            status_code=403, detail=f"Access to the directory `{layer_dir}` is not allowed."
        )

    pattern = f"{filename_weight_format}*.{FILENAME_WEIGHTS_EXTENSION}"
    candidates = list(layer_dir.glob(pattern))

    if not candidates:
        raise HTTPException(
            status_code=404, detail=f"No weight file matching pattern '{pattern}' in `{layer_dir}`"
        )

    if len(candidates) > 1:
        raise HTTPException(
            status_code=400,
            detail=f"Multiple weight files matching pattern '{pattern}' in` {layer_dir}`: `{[str(p) for p in candidates]}`",
        )

    return candidates[0]


def extract_layer_index(path) -> int:
    path_str = str(path)
    match = re.search(r"remote_weights_layer(\d+)\.npy$", path_str)
    if not match:
        raise ValueError(f"❌ No layer index found in path: `{path_str}`")
    return int(match.group(1))


def extract_archive():
    if not TARGET_DIR.exists():
        print("📦 Extracting compiled_models.tar.gz...")
        with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
            tar.extractall(path=TARGET_DIR)
        print("✅ Extraction complete.")
    else:
        print(f"✅ `{TARGET_DIR=}` directory already exists.")


def per_channel_weight_quantization(weight: numpy.ndarray, n_bits: int = 7):
    """Quantize the weights, per-channel using symmetric (signed) quantization.

    Args:
        weight: Weight tensor to quantize
        q_module: Quantized module containing quantization parameters

    Returns:
        tuple: Quantized weights, scale, zero point and weight sum
    """
    weight_float = torch.from_numpy(weight).to(DEVICE)

    q_min, q_max = -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1

    w_min_vals, _ = weight_float.min(dim=0, keepdim=True)
    w_max_vals, _ = weight_float.max(dim=0, keepdim=True)

    weight_scale = (w_max_vals - w_min_vals) / (q_max - q_min)
    # Avoid division by zero.
    weight_scale = torch.where(
        (w_max_vals > w_min_vals), weight_scale, torch.ones_like(weight_scale, device=DEVICE)
    )
    weight_scale = weight_scale.squeeze(-1)  # shape: (out_dim,)
    # Quantization
    weight_zp = torch.round(q_min - w_min_vals / weight_scale).to(
        dtype=torch.float32, device=DEVICE
    )

    # Apply quantization with proper broadcasting
    weight_q = torch.round(weight_float / weight_scale) + weight_zp
    weight_q = torch.clamp(weight_q, q_min, q_max).to(dtype=torch.float32, device=DEVICE)
    sum_w = weight_q.sum(dim=0)  # sum over the input dimension

    return weight_q, weight_scale, weight_zp, sum_w
