import io
import json
import logging
import re
import tarfile
import uuid
from glob import glob
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Tuple

import concrete_ml_extensions as fhext
import numpy
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from utils_server import *

logger = logging.getLogger("uvicorn")

COMPRESSION_KEY = None

app = FastAPI()


@app.post("/add_key")
async def add_key(key: UploadFile):
    """Upload and store the public evaluation key on the server side."""

    global COMPRESSION_KEY
    start = time()

    uid = str(uuid.uuid4())

    print(f"📡 [Endpoint `add_key`] - `{uid=}`")

    start_time = time()
    serialized_public_key = await key.read()
    time_read_key = time() - start_time
    print(f"⏱️ Key read in `{time_read_key:.2f}`s")

    start_time = time()
    public_key = fhext.deserialize_compression_key(serialized_public_key)
    time_deserialization_key = time() - start_time
    print(f"⏱️ Key deserialized in `{time_deserialization_key:.2f}`s")

    KEY_PATH.parent.mkdir(parents=True, exist_ok=True)

    start_time = time()
    serialized_public_key = public_key.serialize()
    time_serialization_key = time() - start_time

    with KEY_PATH.open("wb") as f:
        f.write(serialized_public_key)
    time_storage_key = time() - start_time
    print(f"⏱️ Key storage + serialized in `{time_storage_key:.2f}`s and saved at: `{KEY_PATH}`")

    total_add_key_func = time() - start

    save_benchmark_row(
        {   "endpoint": "Key",
            "uid": uid,
            "time_read_key": time_read_key,
            "time_deserialization_key": time_deserialization_key,
            "time_serialization_key": time_serialization_key,
            "time_storage_key": time_storage_key,
            "total_add_key_func": total_add_key_func,
        }
    )

    COMPRESSION_KEY = public_key

    return {"uid": uid}


@app.post("/send_encrypted_input")
async def send_data(
    encrypted_input: UploadFile = File(...),
    linear_layer_name_path: str = Form(...),
    uid: Optional[str] = Form(None),
):
    """Send the encrypted input to the server."""
    start = time()
    print(f"📡 [Endpoint `send_encrypted_input`] - `{uid=}`")

    encrypted_input_path = ROOT_SERVER_DIR / Path(linear_layer_name_path) / ENCRYPTED_FILENAME_INPUT
    path_weights = fetch_remote_weights(encrypted_input_path.parent)
    index = extract_layer_index(path_weights)

    start_time = time()
    encrypted_content = await encrypted_input.read()
    time_read_input = time() - start_time
    print(
        f"📥 Received encrypted input (`{len(encrypted_content)} bytes`) in `{time_read_input:.2f}`s"
    )

    start_time = time()
    with encrypted_input_path.open("wb") as f:
        f.write(encrypted_content)
    time_storage_input = time() - start_time
    print(
        f"⏱️ Encrypted input saved in `{time_storage_input:.2f}`s and saved at: `{encrypted_input_path}`"
    )

    total_send_input_func = time() - start

    save_benchmark_row(
        {   'endpoint': "Input",
            "uid": uid,
            "index": int(index),
            "time_read_input": time_read_input,
            "time_storage_input": time_storage_input,
            "total_send_input_func": total_send_input_func,
        }
    )

    return {"uid": uid, "status": "Data received successfully."}


@app.post("/compute")
async def compute(
    uid: str = Form(...),
    shape: Tuple[int, int] = Form(...),
    linear_layer_name_path: str = Form(...),
):
    """Computes the FHE matmul over encrypted input."""

    start = time()
    print(f"📡 [Endpoint `compute`] - `{uid=}`")

    layer_dir = ROOT_SERVER_DIR / Path(linear_layer_name_path)

    if not layer_dir.exists():
        raise HTTPException(status_code=404, detail=f"The layer `{layer_dir}` does not exist.")

    # Build paths
    path_encrypted_input = layer_dir / ENCRYPTED_FILENAME_INPUT
    path_encrypted_output = layer_dir / ENCRYPTED_FILENAME_OUTPUT
    path_bias = layer_dir / FILENAME_BIAS
    path_info = layer_dir / FILENAME_INFO
    path_weights = fetch_remote_weights(layer_dir)

    # Validate required files
    required_files = [path_weights, path_encrypted_input, path_info, KEY_PATH]
    for p in required_files:
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Missing file: `{p}`")

    # Deserialize encrypted input
    assert hasattr(fhext, "EncryptedMatrix")
    assert hasattr(fhext.EncryptedMatrix, "deserialize")
    start_time = time()
    with path_encrypted_input.open("rb") as f:
        encrypted_deserialized_input = fhext.EncryptedMatrix.deserialize(f.read())
    time_load_input = time() - start_time

    # Load the public compression key
    assert COMPRESSION_KEY is not None

    # Load metadata
    with path_info.open("r") as f:
        info = json.load(f)

    transpose_inputs1 = info.get("transpose_inputs1", False)
    transpose_inputs2 = info.get("transpose_inputs2", False)
    has_bias = info.get("bias", False)

    # Load weights
    weights = np.load(path_weights)

    # Load bias if present
    bias = None
    if has_bias:
        if not path_bias.exists():
            raise HTTPException(
                status_code=404, detail=f"Bias expected but file missing: `{path_bias}`"
            )
        bias = torch.load(path_bias, map_location=DEVICE)
        print(f"📥 Bias loaded: shape=`{bias.shape}`")

    # Transpose weights if needed
    if shape[1] == weights.shape[1]:
        weights = weights.T
    else:
        print(f"🔥🔥 No transpose: input.shape: {shape}, weights.shape: {weights.shape}")

    # Quantize weights
    start_time = time()
    weight_q, weight_scale, weight_zp, sum_w = per_channel_weight_quantization(weights)
    time_weight_quantization = time() - start_time

    # Encrypted matmul
    start_time = time()
    encrypted_output = fhext.matrix_multiplication(
        encrypted_matrix=encrypted_deserialized_input,
        data=weight_q.long().cpu().numpy().astype(np.int64).astype(np.uint64),
        compression_key=COMPRESSION_KEY,
    )
    time_matmul = time() - start_time
    print(f"⏱️ Encrypted Matmul done in `{time_matmul}`s using '{DEVICE=}'")

    start_time = time()
    encrypted_serialized_output = encrypted_output.serialize()
    time_serialization_output = time() - start_time

    # Save encrypted output
    start_time = time()
    with path_encrypted_output.open("wb") as f:
        f.write(encrypted_serialized_output)
    time_storage_output = time() - start_time
    print(f"📤 Encrypted output saved (`{len(encrypted_serialized_output)}` bytes)")
    print(
        f"⏱️ Encrypted output saved in `{time_storage_output}`s and saved at: `{path_encrypted_output}`"
    )

    # Prepare metadata
    metadata = {
        "encrypted_output": encrypted_serialized_output,
        "weight_scale": weight_scale.cpu().numpy(),
        "weight_zp": weight_zp.cpu().numpy(),
        "sum_w": sum_w.cpu().numpy(),
        "weight_shape": numpy.array(weight_q.shape, dtype=numpy.int32),
        "transpose_inputs1": numpy.array([transpose_inputs1], dtype=numpy.bool_),
        "transpose_inputs2": numpy.array([transpose_inputs2], dtype=numpy.bool_),
        "input_n_bits": 7,
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

    start_time = time()
    response = StreamingResponse(
        result_stream(),
        media_type="application/octet-stream",
        headers={"Content-Disposition": "attachment; filename=server_response.npz"},
    )
    time_packing_output_response = time() - start_time
    print(f"⏱️ Server response traited in `{time_packing_output_response}`s")

    total_compute_func = time() - start

    save_benchmark_row(
        {   "endpoint": "compute",
            "uid": uid,
            "layer_name": str(layer_dir).split("/")[-3],
            "index": extract_layer_index(path_weights),
            "input_shape": shape,
            "remote_weight_shape": weights.shape,
            "time_load_input": time_load_input,
            "time_weight_quantization": time_weight_quantization,
            "time_matmul": time_matmul,
            "time_storage_output": time_storage_output,
            "time_serialization_output": time_serialization_output,
            "time_packing_output_response": time_packing_output_response,
            "total_compute_func": total_compute_func,
        }
    )

    return response


@app.get("/ping")
async def ping():
    """
    curl http://localhost:8000/ping
    """
    print("📡 [Endpoint `send_encrypted_input`]")
    return {"status": "ok"}


@app.get("/download_benchmark")
async def download_benchmark():
    """
    Endpoint to download the benchmark CSV file.

    curl -o server_benchmarks.csv http://127.0.0.1:8000/download_benchmark

    """
    if not BENCHMARK_FILE_PATH.exists():
        raise HTTPException(
            status_code=404, detail=f"Benchmark file not found: `{BENCHMARK_FILE_PATH}`"
        )

    return FileResponse(path=BENCHMARK_FILE_PATH, media_type="text/csv", filename="benchmark.csv")


@app.get("/display_benchmark")
async def display_benchmark():
    """
    curl -v http://127.0.0.1:8000/display_benchmark
    curl -v http://localhost:8000/display_benchmark
    """

    if not BENCHMARK_FILE_PATH.exists():
        raise HTTPException(
            status_code=404, detail=f"Benchmark file not found: `{BENCHMARK_FILE_PATH}`"
        )

    try:
        df = pd.read_csv(BENCHMARK_FILE_PATH, sep=";")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading benchmark file: {e}")

    print("\n=== BENCHMARK CONTENT ===\n")
    print(df)
    print("\n=========================\n")

    return Response(status_code=204)


if __name__ == "__main__":
    print(f"📡 [Server startup]")
    extract_archive()
    print("--> Extraction done")
    init_benchmark_file()
    print("--> init benchmark file")

    # uvicorn.run("server_glwe:app", host="0.0.0.0", port=8000)
    # uvicorn.run("server_glwe:app", host="::", port=8000, log_level="debug")
    uvicorn.run("server_glwe:app", host="0.0.0.0", port=8000, log_level="debug")
