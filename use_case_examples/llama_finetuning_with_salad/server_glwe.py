import ast
import io
import uuid

import torch
import numpy
import time
from fastapi import UploadFile, File, Form

from typing import Optional
from pathlib import Path
import uvicorn
from utils_dev import COMPILED_MODELS_PAH
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from concrete.ml.torch.hybrid_model import HybridFHEModelServer

from common_variables import COMPILED_MODELS_PAH
import concrete_ml_extensions as fhext


app = FastAPI()
SERVER_DIR = 'server'

SERVER_EVAL_KEY_PATH = Path(f'{COMPILED_MODELS_PAH}/server/serialized_key.bin')
@app.post("/add_key")
async def add_key(
    key: UploadFile
):
    """Add public key.

    Arguments:
        key (UploadFile): public key

    Returns:
        Dict[str, str]
            - uid: uid a personal uid
    """
    print(f"📡 [Endpoint `add_key`]")

    path_to_server = Path(f'compiled_models/server')
    path_to_server.mkdir(exist_ok=True)

    # Load the serialized key from the uploaded file
    serialized_ckey = await key.read()

    # Deserialize the compressed key
    compression_key = fhext.deserialize_compression_key(serialized_ckey)
    print("🔐 Server key deserialized and set.", type(compression_key))

    with (path_to_server / "serialized_key.bin").open("wb") as binary_file:
        binary_file.write(compression_key.serialize())

    # Generate a UID for the key
    uid = str(uuid.uuid4())
    print(f"📦 [Endpoint add_key] {uid=}")
    return {"uid": uid}


# send the input data to the server
@app.post("/send_encrypted_input")
async def send_data(
    encrypted_input: UploadFile = File(...),
    linear_layer_name_path: str = Form(...),
    uid: Optional[str] = Form(None)
):
    """Send input data to the server.

    Arguments:
        input_data (str): Input data to be sent.
        uid (Optional[str]): Optional unique identifier for the request.

    Returns:
        Dict[str, str]: Acknowledgment message with the UID.
    """
    print(f"📡 [Endpoint `send_encrypted_input`] {uid=}")

    content = await encrypted_input.read()
    print("ciphertext_serialized size:", len(content), "bytes")


    dest_dir = Path(linear_layer_name_path)

    # Écriture du fichier
    with open(f'{dest_dir}/encrypted_input.bin', "wb") as f:
        f.write(content)


    return {"message": "Data received successfully", "uid": uid}


def _per_channel_weight_quantization(weight: numpy.ndarray, device: torch.device):
    """Quantize the weights, per-channel using symmetric (signed) quantization.

    Args:
        weight: Weight tensor to quantize
        q_module: Quantized module containing quantization parameters
        device: Device to place tensors on

    Returns:
        tuple: Quantized weights, scale, zero point and weight sum
    """
    # weight_float = torch.from_numpy(weight).to(device)
    weight_float = weight.to(device)

    # Get the signed integer range
    q_min, q_max = 0, 127

    w_min_vals, _ = weight_float.min(dim=0, keepdim=True)
    w_max_vals, _ = weight_float.max(dim=0, keepdim=True)

    weight_scale = (w_max_vals - w_min_vals) / (q_max - q_min)
    # Avoid division by zero.
    weight_scale = torch.where(
        (w_max_vals > w_min_vals), weight_scale, torch.ones_like(weight_scale)
    )
    weight_scale = weight_scale.squeeze(-1)  # shape: (out_dim,)

    # Quantization
    weight_zp = torch.round(q_min - w_min_vals / weight_scale).to(torch.float32)

    # Apply quantization with proper broadcasting
    weight_q = torch.round(weight_float / weight_scale) + weight_zp
    weight_q = torch.clamp(weight_q, q_min, q_max).to(torch.float32)
    sum_w = weight_q.sum(dim=0)  # sum over the input dimension

    return weight_q, weight_scale, weight_zp, sum_w



@app.post("/compute")
async def compute(
    uid: str = Form(),
    linear_layer_name_path: str = Form(...),
):
    """
    Computes the circuit over encrypted input.

    Args:
        model_input (UploadFile): Input of the circuit.
        uid (str): The UID of the public key to use for computations.
        linear_layer_name_path (str): Path to the directory containing 'encrypted_input.bin'.

    Returns:
        StreamingResponse: The result of the computation, streamed back in chunks.
    """
    print(f"📡 [Endpoint compute]")

    dest_dir = Path(linear_layer_name_path)
    encrypted_input_path = dest_dir / "encrypted_input.bin"
    remote_weights_path = dest_dir / "remote_weights.pth"
    encrypted_result_path = dest_dir / "encrypted_result.bin"

    if not encrypted_input_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {encrypted_input_path}")

    if not remote_weights_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {remote_weights_path}")

    with open(encrypted_input_path, "rb") as f:
        encrypted_input = f.read()
        print(f"📥 encrypted_input loaded ({len(encrypted_input)} bytes)")
        encrypted_input = fhext.EncryptedMatrix.deserialize(encrypted_input)


    private_remote_weights = torch.load(remote_weights_path)

    print(f"📥 private_remote_weights loaded {private_remote_weights.shape=}")
    # [2048, 2048]


    with open(SERVER_EVAL_KEY_PATH, "rb") as binary_file:
        serialized_ckey = binary_file.read()
        compression_key = fhext.deserialize_compression_key(serialized_ckey)
        print("🔐 Server key deserialized and set.", type(compression_key))

    weight_q, weight_scale, weight_zp, sum_w = _per_channel_weight_quantization(private_remote_weights, 'cpu')

    weight_q_int = weight_q.long().cpu().numpy().astype(numpy.int64).astype(numpy.uint64)

    print(weight_q_int.shape[1], 'poulfff')
    encrypted_result = fhext.matrix_multiplication(
        encrypted_matrix=encrypted_input,
        data=weight_q_int,
        compression_key=compression_key,
    )

    with open(encrypted_result_path, "wb") as binary_file:
            binary_file.write(encrypted_result.serialize())

    print(f"📤 encrypted_result saved ({len(encrypted_result.serialize())} bytes)")

    def result_stream():
        chunk_size = 4096
        for i in range(0, len(encrypted_result.serialize()), chunk_size):
            yield encrypted_result.serialize()[i:i+chunk_size]

    return StreamingResponse(
        result_stream(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename=encrypted_result.bin"}
    )

if __name__ == "__main__":

    print(f'📡 [Server.py] {COMPILED_MODELS_PAH=}')

    uvicorn.run(app, host="127.0.0.1", port=8000)
