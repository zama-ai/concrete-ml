import json
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
    remote_bias_path = dest_dir / "remote_bias.pth"
    encrypted_result_path = dest_dir / "encrypted_result.bin"
    remote_information_path = dest_dir / "information.json"

    if not encrypted_input_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {encrypted_input_path}")

    if not remote_weights_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {remote_weights_path}")

    with open(encrypted_input_path, "rb") as f:
        encrypted_input = f.read()
        print(f"📥 encrypted_input loaded ({len(encrypted_input)} bytes)")
        encrypted_input = fhext.EncryptedMatrix.deserialize(encrypted_input)

    with open(SERVER_EVAL_KEY_PATH, "rb") as binary_file:
        serialized_ckey = binary_file.read()
        compression_key = fhext.deserialize_compression_key(serialized_ckey)
        print("🔐 Server key deserialized and set.", type(compression_key))

    with open(remote_information_path, "r") as f:
        info = json.load(f)
        transpose_inputs1 = info.get("transpose_inputs1", False)
        transpose_inputs2 = info.get("transpose_inputs2", False)
        has_bias = info.get("bias", False)

    private_remote_weights = torch.load(remote_weights_path)
    print(f"📥 --------------  private_remote_weights loaded {private_remote_weights.shape=}")

    private_remote_bias = None
    if has_bias:
        assert remote_bias_path.exists(), "Bias file specified but not found"
        private_remote_bias = torch.load(remote_bias_path)
        print(f"📥 private_remote_bias loaded {private_remote_bias.shape=}")


    weight_q, weight_scale, weight_zp, sum_w = _per_channel_weight_quantization(private_remote_weights, 'cpu')
    weight_q_int = weight_q.long().cpu().numpy().astype(numpy.int64).astype(numpy.uint64)
    print(f'🐞 -------------- {weight_q_int.shape=}')

    encrypted_result = fhext.matrix_multiplication(
        encrypted_matrix=encrypted_input,
        data=weight_q_int,
        compression_key=compression_key,
    )

    with open(encrypted_result_path, "wb") as binary_file:
            binary_file.write(encrypted_result.serialize())
            print(f"📤 encrypted_result saved ({len(encrypted_result.serialize())} bytes)")

    save_dict = {
    "encrypted_result": encrypted_result.serialize(),
    "weight_scale": weight_scale.cpu().numpy(),
    "weight_zp": weight_zp.cpu().numpy(),
    "sum_w": sum_w.cpu().numpy(),
    "weight_shape": numpy.array(weight_q.shape, dtype=numpy.int32),
    "transpose_inputs1": numpy.array([transpose_inputs1], dtype=numpy.bool_),
    "transpose_inputs2": numpy.array([transpose_inputs2], dtype=numpy.bool_),
        }
    if private_remote_bias is not None:
        save_dict["bias"] = private_remote_bias.cpu().numpy()


    print(f"🐞 -------------- {weight_scale=}")
    print(f"🐞 -------------- {weight_zp=}")
    print(f"🐞 -------------- {sum_w=}")
    print(f"🐞 -------------- {weight_q.shape=}")


    def result_stream():
        chunk_size = 4096

        buffer = io.BytesIO()
        numpy.savez_compressed(buffer, **save_dict)
        buffer.seek(0)

        chunk_size = 4096
        while True:
            chunk = buffer.read(chunk_size)
            if not chunk:
                break
            yield chunk

    return StreamingResponse(
        result_stream(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename=encrypted_result_bundle.bin"}
    )

if __name__ == "__main__":

    print(f'📡 [Server.py] {COMPILED_MODELS_PAH=}')

    uvicorn.run(app, host="127.0.0.1", port=8000)
