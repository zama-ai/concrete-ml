import numpy as np
import io
import json
import uuid
import torch
import numpy
from pathlib import Path
from typing import Optional, Tuple
from glob import glob

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
import uvicorn

import concrete_ml_extensions as fhext
from common_variables import COMPILED_MODELS_PATH

# Path configuration

torch.set_printoptions(precision=10, sci_mode=False)

SERVER_DIR = Path(COMPILED_MODELS_PATH) / "server"
KEY_PATH   = SERVER_DIR / "serialized_key.bin"

ENCRYPTED_FILENAME_INPUT = "encrypted_input.bin"
CLEAR_FILENAME_INPUT = "input_plain.npy"

ENCRYPTED_FILENAME_OUTPUT  = "encrypted_output.bin"
CLEAR_FILENAME_OUTPUT  = "clear_output.bin"

FILENAME_WEIGHTS_FORMAT = "remote_weights"
FILENAME_BIAS    = "remote_bias.pth"
FILENAME_INFO    = "information.json"


DEVICE = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

app = FastAPI()

@app.post("/add_key")
async def add_key(key: UploadFile):
    """Upload and store the public evaluation key on the server side."""

    uid = str(uuid.uuid4())
    SERVER_DIR.mkdir(parents=True, exist_ok=True)

    print(f"📡 [Endpoint `add_key`], {uid=}")

    serialized_ckey = await key.read()
    compression_key = fhext.deserialize_compression_key(serialized_ckey)

    with KEY_PATH.open("wb") as f:
        f.write(compression_key.serialize())
        print(f"🔐 {uid=}, Server key saved at {KEY_PATH}. ")

    return {"uid": uid}


@app.post("/send_encrypted_input")
async def send_data(
    encrypted_input: UploadFile = File(...),
    clear_input: Optional[UploadFile] = File(None),
    linear_layer_name_path: str = Form(...),
    uid: Optional[str] = Form(None)
):
    """Save an encrypted input tensor to disk."""
    print(f"📡 [Endpoint `send_encrypted_input`] {uid=}")

    dest_path = Path(linear_layer_name_path)
    dest_path.mkdir(parents=True, exist_ok=True)

    if clear_input is not None:
        clear_content = await clear_input.read()
        print(f"📥 Received clear input ({len(clear_content)} bytes)")
        with (dest_path / CLEAR_FILENAME_INPUT).open("wb") as f:
            f.write(clear_content)

    encrypted_content = await encrypted_input.read()
    print(f"📥 Received ciphertext ({len(encrypted_content)} bytes)")

    with (dest_path / ENCRYPTED_FILENAME_INPUT).open("wb") as f:
        f.write(encrypted_content)

    return {"message": "Data received successfully", "uid": uid}


def _per_channel_weight_quantization(weight: numpy.ndarray, device: torch.device, n_bits: int = 7):
    """Quantize the weights, per-channel using symmetric (signed) quantization.

    Args:
        weight: Weight tensor to quantize
        q_module: Quantized module containing quantization parameters
        device: Device to place tensors on

    Returns:
        tuple: Quantized weights, scale, zero point and weight sum
    """

    print('in _per_channel_weight_quantization: weight', weight.flatten()[:5])

    weight_float = torch.from_numpy(weight).to(device)
    print('in _per_channel_weight_quantization: weight', weight.flatten()[:5])

    # weight_float = weight.to(device)
    # Get the signed integer range

    q_min, q_max = -64, 63 #0, 2 ** n_bits - 1

    w_min_vals, _ = weight_float.min(dim=0, keepdim=True)
    w_max_vals, _ = weight_float.max(dim=0, keepdim=True)
    print('--------======------')
    print(f'{w_min_vals.shape=}, {w_max_vals.shape=}')

    weight_scale = (w_max_vals - w_min_vals) / (q_max - q_min)
    # Avoid division by zero.
    weight_scale = torch.where(
        (w_max_vals > w_min_vals), weight_scale, torch.ones_like(weight_scale)
    )
    print(f'{weight_scale.shape=}')
    weight_scale = weight_scale.squeeze(-1)  # shape: (out_dim,)
    print(f'{weight_scale.shape=}')
    # Quantization
    weight_zp = torch.round(q_min - w_min_vals / weight_scale).to(torch.float32)

    # Apply quantization with proper broadcasting
    weight_q = torch.round(weight_float / weight_scale) + weight_zp
    weight_q = torch.clamp(weight_q, q_min, q_max).to(torch.float32)
    sum_w = weight_q.sum(dim=0)  # sum over the input dimension

    return weight_q, weight_scale, weight_zp, sum_w


@app.post("/compute")
async def compute(
    uid: str = Form(...),
    shape: Tuple[int, int] = Form(...),
    linear_layer_name_path: str = Form(...),
):
    """Computes the FHE matmul over encrypted input."""

    print(f"📡 [Endpoint `compute`] for `{uid=}`")

    model_dir = Path(linear_layer_name_path)

    path_weights = Path(glob(f'{model_dir/FILENAME_WEIGHTS_FORMAT}*.npy')[0])

    path_input = model_dir / ENCRYPTED_FILENAME_INPUT
    path_disable_result = model_dir / CLEAR_FILENAME_INPUT

    path_bias = model_dir / FILENAME_BIAS

    path_info = model_dir / FILENAME_INFO

    path_output = model_dir / ENCRYPTED_FILENAME_OUTPUT

    for p in [path_weights, path_info]:
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Missing file: {p}")

    clear_input = np.load(path_disable_result)

    with path_input.open("rb") as f:
        encrypted_deserialized_input = fhext.EncryptedMatrix.deserialize(f.read())

    with KEY_PATH.open("rb") as f:
        compression_key = fhext.deserialize_compression_key(f.read())
        print("🔐 Compression key loaded.")

    with path_info.open("r") as f:
        info = json.load(f)
        transpose_inputs1 = info.get("transpose_inputs1", False)
        transpose_inputs2 = info.get("transpose_inputs2", False)
        has_bias = info.get("bias", False)
        input_n_bits = info.get("input_n_bits", 7)

    weights = numpy.load(path_weights)

    bias = torch.load(path_bias) if has_bias and path_bias.exists() else None
    if bias is not None:
        print(f"📥📥📥📥📥📥📥 Bias loaded: {bias.shape=}")

    if shape[1] == weights.shape[1]:
        weights = weights.T
    else:
        print(f"📥📥📥📥📥📥📥 Not transposed")

    weight_q, weight_scale, weight_zp, sum_w = _per_channel_weight_quantization(weights, device=DEVICE)
    weight_q_int = weight_q.long().cpu().numpy().astype(numpy.int64).astype(numpy.uint64)
    print(f"📥 Weights loaded: {path_weights=} - {clear_input.shape=}, {weight_q_int.shape=},")

    clear_output = clear_input @ weight_q_int

    encrypted_output = fhext.matrix_multiplication(
        encrypted_matrix=encrypted_deserialized_input,
        data=weight_q_int,
        compression_key=compression_key,
    )

    encrypted_deserialized_output = encrypted_output.serialize()

    # encrypted_output = encrypted_input

    print(f"📡 -- `input.shape={shape}` vs `{weights.T.shape=}`")
    print(f"📡 -- {weights.shape=}")
    print(f"📡 -- {clear_input.shape=}")
    print(f"📡 -- {weight_q.shape=}")
    print(f"📡 -- {weight_q_int.shape=}")
    print(f"📡 -- {weight_scale.shape =}")
    print(f"📡 -- {weight_zp.shape    =}")
    print(f"📡 -- {sum_w.shape        =}")
    print(f"📡 -- {weights.T.flatten()[:5]=}")
    print(f"📡 -- {weight_q.flatten()[:5]=}")
    print(f"📡 -- {weight_q_int.flatten()[:5]=}")

    with (model_dir / ENCRYPTED_FILENAME_OUTPUT).open("wb") as f:
        f.write(clear_output)
        print(f"📤 Clear result saved ({len(clear_output)} bytes)")

    with path_output.open("wb") as f:
        f.write(encrypted_deserialized_output)
        print(f"📤 Encrypted result saved ({len(encrypted_deserialized_output)} bytes)")

    metadata = {
        "encrypted_output": encrypted_deserialized_output,
        "clear_output": clear_output,
        "weight_scale": weight_scale.cpu().numpy(),
        "weight_zp": weight_zp.cpu().numpy(),
        "sum_w": sum_w.cpu().numpy(),
        "weight_shape": numpy.array(weight_q.shape, dtype=numpy.int32),
        "transpose_inputs1": numpy.array([transpose_inputs1], dtype=numpy.bool_),
        "transpose_inputs2": numpy.array([transpose_inputs2], dtype=numpy.bool_),
        "input_n_bits": input_n_bits,
    }

    if bias is not None:
        metadata["bias"] = bias.cpu().numpy()

    def result_stream():
        buffer = io.BytesIO()
        numpy.savez_compressed(buffer, **metadata)
        buffer.seek(0)
        while chunk := buffer.read(4096):
            yield chunk

    return StreamingResponse(
        result_stream(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=server_response.npz"},
    )

if __name__ == "__main__":
    print(f"📡 [Server startup] {COMPILED_MODELS_PATH=}")
    uvicorn.run(app, host="127.0.0.1", port=8000)
