import ast
import io
import time
from typing import Optional
from pathlib import Path
import uvicorn
from utils_dev import COMPILED_MODELS_PAH
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from concrete.ml.torch.hybrid_model import HybridFHEModelServer


from common_variables import COMPILED_MODELS_PAH
app = FastAPI()


@app.get("/list_models")
def list_models():
    return server.modules


@app.get("/check_inputs")
def check_inputs(
    model_name: str,
    module_name: Optional[str] = None,
    input_shape: Optional[str] = None,
):
    try:
        server.check_inputs(model_name, module_name, input_shape)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return True


@app.get("/list_shapes")
def list_shapes(model_name: str = Form(), module_name: str = Form()):
    print(f"📡 [Endpoint list_shapes]\n{model_name=}, {module_name=}")
    server.check_inputs(model_name, module_name, None)
    string_list = list(server.modules[model_name][module_name].keys())
    tuple_list = [tuple(ast.literal_eval(s)) for s in string_list]
    return tuple_list


@app.get("/get_client")
def get_client(model_name: str = Form(), module_name: str = Form(), input_shape: str = Form()):
    """Get client.

    Returns:
        FileResponse: client.zip

    Raises:
        HTTPException: if the file can't be find locally
    """
    print(f"📡 [Endpoint get_client]\n{model_name=}, {module_name=}, {input_shape=}")
    path_to_client = server.get_client(model_name, module_name, input_shape)
    return path_to_client


@app.post("/add_key")
async def add_key(
    key: UploadFile,
    model_name: str = Form(),
    module_name: str = Form(),
    input_shape: str = Form(),
):
    """Add public key.

    Arguments:
        key (UploadFile): public key

    Returns:
        Dict[str, str]
            - uid: uid a personal uid
    """
    print(f"📡 [Endpoint add_key]")
    server.check_inputs(model_name, module_name, input_shape)
    return server.add_key(await key.read(), model_name, module_name, input_shape)


def stream_response(encrypted_results: bytes, chunk_size: int = 1024 * 1024):
    """Yields chunks of encrypted results.

    Args:
        encrypted_results (bytes): The byte data to be streamed.
        chunk_size (int): The size of the chunks in which encrypted_results should be streamed.
                        Defaults to 1MB.

    Returns:
        bytes: Chunks of encrypted_results of size chunk_size.
    """
    buffer = io.BytesIO(encrypted_results)
    while chunk := buffer.read(chunk_size):
        yield chunk


@app.post("/compute")
async def compute(
    model_input: UploadFile,
    uid: str = Form(),
    model_name: str = Form(),
    module_name: str = Form(),
    input_shape: str = Form(),
):
    """
    Computes the circuit over encrypted input.

    Args:
        model_input (UploadFile): Input of the circuit.
        uid (str): The UID of the public key to use for computations.
        model_name (str): The name of the model to be used.
        module_name (str): The name of the module containing the computation circuit.
        input_shape (str): The shape of the input data.

    Returns:
        StreamingResponse: The result of the computation, streamed back in chunks.
    """
    print(f"📡 [Endpoint compute]")
    start_time = time.time()
    uploaded_data = await model_input.read()
    end_time = time.time()

    try:
        start_time = time.time()
        encrypted_results = server.compute(
            uploaded_data,
            uid,
            model_name,
            module_name,
            input_shape,
        )
        end_time = time.time()

    except Exception as e:
        print(f"❌ Error during compute: `{e}`")

    return StreamingResponse(stream_response(encrypted_results))


if __name__ == "__main__":

    print(f'📡 [Server.py] {COMPILED_MODELS_PAH=}')

    server = HybridFHEModelServer(
        key_path=COMPILED_MODELS_PAH, model_dir=COMPILED_MODELS_PAH, logger=None
    )

    uvicorn.run(app, host="127.0.0.1", port=8000)
