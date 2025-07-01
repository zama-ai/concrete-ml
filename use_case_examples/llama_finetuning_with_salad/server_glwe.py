import tarfile
import numpy as np
import io
import json
import uuid
import torch
import numpy
from time import time
from pathlib import Path
from typing import Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
import uvicorn

import concrete_ml_extensions as fhext

from utils_server import *

COMPRESSION_KEY = None
archive_path = Path("./deployment_float/compiled_models.tar.gz")
target_dir = Path("./deployment/compiled_models")

app = FastAPI()

@app.post("/add_key")
async def add_key(key: UploadFile):
    """Upload and store the public evaluation key on the server side."""

    uid = str(uuid.uuid4())
    SERVER_DIR.mkdir(parents=True, exist_ok=True)

    print(f"📡 [Endpoint `add_key`] - `{uid=}`")

    serialized_public_key = await key.read()
    public_key = fhext.deserialize_compression_key(serialized_public_key)

    with KEY_PATH.open("wb") as f:
        f.write(public_key.serialize())
        print(f"🔐 {uid=}, Server key saved at `{KEY_PATH}`.")

    load_compression_key()

    return {"uid": uid}

@app.post("/send_encrypted_input")
async def send_data(
    encrypted_input: UploadFile = File(...),
    clear_input: Optional[UploadFile] = File(None),
    linear_layer_name_path: str = Form(...),
    uid: Optional[str] = Form(None)
):
    """Send the encrypted input to the server."""

    print(f"📡 [Endpoint `send_encrypted_input`] - `{uid=}`")

    path = Path(linear_layer_name_path)
    print(f'🔥🔥🔥🔥🔥')
    print(linear_layer_name_path)
    path.mkdir(parents=True, exist_ok=True)

    if clear_input is not None:
        clear_content = await clear_input.read()
        print(f"📥 Received clear input (`{len(clear_content)}` bytes)")
        with (path / CLEAR_FILENAME_INPUT).open("wb") as f:
            f.write(clear_content)

    encrypted_content = await encrypted_input.read()
    print(f"📥 Received encrypted input (`{len(encrypted_content)}` bytes)")

    with (path / ENCRYPTED_FILENAME_INPUT).open("wb") as f:
        f.write(encrypted_content)

    return {"uid": uid, "status": "Data received successfully."}


def load_compression_key():
    global COMPRESSION_KEY
    if not KEY_PATH.exists():
        raise RuntimeError(f"Compression key not found at `{KEY_PATH}`")
    with KEY_PATH.open("rb") as f:
        COMPRESSION_KEY = fhext.deserialize_compression_key(f.read())
    print("✅ Compression key loaded once and cached.")


@app.post("/compute")
async def compute(
    uid: str = Form(...),
    shape: Tuple[int, int] = Form(...),
    linear_layer_name_path: str = Form(...),
):
    """Computes the FHE matmul over encrypted input."""

    print(f"📡 [Endpoint `compute`] - `{uid=}`")

    layer_dir = Path(linear_layer_name_path)
    if not layer_dir.exists():
        raise HTTPException(status_code=404, detail=f"The layer `{layer_dir}` does not exist.")

    # Build paths
    path_encrypted_input = layer_dir / ENCRYPTED_FILENAME_INPUT
    path_bias = layer_dir / FILENAME_BIAS
    path_info = layer_dir / FILENAME_INFO
    # path_clear_input = layer_dir / CLEAR_FILENAME_INPUT
    # path_encrypted_output = layer_dir / ENCRYPTED_FILENAME_OUTPUT
    # path_clear_output = layer_dir / CLEAR_FILENAME_OUTPUT
    # path_weights_q = fetch_remote_weights(layer_dir, filename_weight_format='remote_quantized_weights')
    path_weights = fetch_remote_weights(layer_dir)

    # Validate required files
    required_files = [path_weights, path_encrypted_input, path_info, KEY_PATH]
    for p in required_files:
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Missing file: `{p}`")

    # Load clear input
    # clear_input = np.load(path_clear_input)

    # Deserialize encrypted input
    assert hasattr(fhext, "EncryptedMatrix")
    assert hasattr(fhext.EncryptedMatrix, "deserialize")
    with path_encrypted_input.open("rb") as f:
        encrypted_deserialized_input = fhext.EncryptedMatrix.deserialize(f.read())

    # Load the public compression key
    if COMPRESSION_KEY is None:
        raise RuntimeError("Compression key not loaded. Did you call `load_compression_key()`?")

    # Load metadata
    with path_info.open("r") as f:
        info = json.load(f)

    transpose_inputs1 = info.get("transpose_inputs1", False)
    transpose_inputs2 = info.get("transpose_inputs2", False)
    has_bias = info.get("bias", False)
    input_n_bits = info.get("input_n_bits", 7)
    weight_scale = info.get("weight_scale", None)
    weight_zp = info.get("weight_zp", None)
    sum_w = info.get("insum_wput_n_bits", None)

    # Load weights
    weights = np.load(path_weights)

    # Load bias if present
    bias = None
    if has_bias:
        if not path_bias.exists():
            raise HTTPException(status_code=404, detail=f"Bias expected but file missing: `{path_bias}`")
        bias = torch.load(path_bias, map_location=DEVICE)
        print(f"📥 Bias loaded: shape=`{bias.shape}`")

    # Transpose weights if needed
    if shape[1] == weights.shape[1]:
        weights = weights.T
    else:
        print(f'🔥🔥🔥🔥🔥 No transpose: input.shape: {shape}, weights.shape: {weights.shape}')

    # Quantize weights
    weight_q, weight_scale, weight_zp, sum_w = per_channel_weight_quantization(weights)
    weight_q_int = (
        weight_q.long()
        .numpy()
        .astype(np.int64)
        .astype(np.uint64)
    )

    # weight_q_2 = numpy.load(path_weights_q).astype(np.int64).astype(np.uint64)
    # assert all(weight_q_int.flatten() == weight_q_2.flatten())
    # assert all(weight_scale.flatten() == weight_scale.flatten())
    # assert all(weight_zp.flatten() == weight_zp.flatten())
    # assert all(sum_w.flatten() == sum_w.flatten())
    # print(f'🔥🔥🔥🔥🔥 (TYPE) {type(weight_q_2)=}, {type(weight_q_int)=}')

    # Clear matmul
    # clear_output = clear_input @ weight_q_int

    # Encrypted matmul
    encrypted_output = fhext.matrix_multiplication(
        encrypted_matrix=encrypted_deserialized_input,
        data=weight_q_int,
        compression_key=COMPRESSION_KEY,
    )
    encrypted_serialized_output = encrypted_output.serialize()

    # # Save clear output
    # clear_output_bytes = clear_output.tobytes()
    # with (layer_dir / CLEAR_FILENAME_OUTPUT).open("wb") as f:
    #     f.write(clear_output_bytes)
    #     print(f"📤 Clear result saved (`{len(clear_output_bytes)}` bytes)")

    # # Save encrypted output
    # with path_encrypted_output.open("wb") as f:
    #     f.write(encrypted_serialized_output)
    #     print(f"📤 Encrypted result saved (`{len(encrypted_serialized_output)}` bytes)")

    # # Save clear output
    # with path_clear_output.open("wb") as f:
    #     f.write(clear_output)
    #     print(f"📤 Clear result saved (`{len(clear_output)}` bytes)")

    # Prepare metadata
    metadata = {
        "encrypted_output": encrypted_serialized_output,
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

    # Build streaming response
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


def extract_archive():

    if not target_dir.exists():
        print("📦 Extracting compiled_models.tar.gz...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=target_dir)
        print("✅ Extraction complete.")
    else:
        print("✅ compiled_models directory already exists.")


if __name__ == "__main__":
    print(f"📡 [Server startup] - `{COMPILED_MODELS_PATH=}`")
    print(f"📡 [Server startup] -  Extract archive")
    extract_archive()
    uvicorn.run(app, host="127.0.0.1", port=8000)

