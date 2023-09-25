"""Hybrid Model Deployment Server.

Routes:
    - Get all names
    - Get client.zip
    - Add a key
    - Compute
"""

import argparse
import ast
import io
import time
import uuid
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Generator, Optional, Tuple, Union

import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from loguru import logger

# No relative import here because when not used in the package itself
from concrete.ml.deployment import FHEModelServer


def underscore_str_to_tuple(tup):
    return ast.literal_eval(tup.replace("p_", "(").replace("_p", ")").replace("_", ", "))


if __name__ == "__main__":
    FILE_FOLDER = Path(__file__).parent

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", dest="port", type=int, default=8000)
    parser.add_argument(
        "--path-to-models",
        dest="path_to_models",
        type=Path,
        default=FILE_FOLDER / Path("compiled_models"),
    )
    parser.add_argument(
        "--path-to-keys",
        dest="path_to_keys",
        type=Path,
        default=FILE_FOLDER / Path("user_keys"),
    )
    args = parser.parse_args()
    app = FastAPI(debug=False)
    # Model-name -> Module-Name -> Input-shape

    KEY_PATH = args.path_to_keys
    KEY_PATH.mkdir(exist_ok=True)
    MODELS_PATH = args.path_to_models
    PORT = args.port
    MODULES = defaultdict(dict)
    # Populate modules at the beginning
    # this could also be done dynamically on each query if needed
    # We build the following mapping:
    # model_name -> module_name -> input_shape -> some information
    for model_path in MODELS_PATH.iterdir():  # Model
        if not model_path.is_dir():
            continue
        model_name = model_path.name
        MODULES[model_name] = defaultdict(dict)
        for module_path in model_path.iterdir():  # Module
            if not module_path.is_dir():
                continue
            module_name = module_path.name
            MODULES[model_name][module_name] = defaultdict(dict)
            for input_shape_path in module_path.iterdir():
                if not input_shape_path.is_dir():
                    continue
                input_shape = str(underscore_str_to_tuple(input_shape_path.name))
                MODULES[model_name][module_name][input_shape] = {
                    "path": input_shape_path.resolve(),
                    "module_name": module_name,
                    "model_name": model_name,
                    "shape": input_shape,
                }

    @lru_cache(maxsize=None)
    def load_key(uid) -> bytes:
        with open(KEY_PATH / str(uid), "rb") as file:
            return file.read()

    def dump_key(key_bytes: bytes, uid: Union[uuid.UUID, str]) -> None:
        with open(KEY_PATH / str(uid), "wb") as file:
            file.write(key_bytes)

    @lru_cache(maxsize=None)
    def get_circuit(model_name, module_name, input_shape):
        return FHEModelServer(str(MODULES[model_name][module_name][input_shape]["path"]))

    def check_inputs(model_name: str, module_name: Optional[str], input_shape: Optional[Tuple]):
        if model_name not in MODULES:
            raise HTTPException(
                status_code=500,
                detail=f"provided names '{model_name}' does not match any known name",
            )
        if module_name is not None and module_name not in MODULES[model_name]:
            raise HTTPException(
                status_code=500,
                detail=f"provided names '{module_name}' does not match any known name"
                f"{list(MODULES[model_name].keys())}",
            )
        if input_shape is not None and input_shape not in MODULES[model_name][module_name]:
            raise HTTPException(
                status_code=500,
                detail=f"provided names '{module_name}' does not match any known name"
                f"{list(MODULES[model_name][module_name].keys())}",
            )

    @app.get("/list_models")
    def list_models():
        return MODULES

    @app.get("/list_modules")
    def list_modules(model_name: str = Form()):
        check_inputs(model_name, None, None)
        return MODULES[model_name]

    @app.get("/list_shapes")
    def list_shapes(model_name: str = Form(), module_name: str = Form()):
        check_inputs(model_name, module_name, None)
        return MODULES[model_name][module_name]

    @app.get("/get_client")
    def get_client(model_name: str = Form(), module_name: str = Form(), input_shape: str = Form()):
        """Get client.

        Returns:
            FileResponse: client.zip

        Raises:
            HTTPException: if the file can't be find locally
        """
        check_inputs(model_name, module_name, input_shape)
        path_to_client = (
            MODULES[model_name][module_name][str(input_shape)]["path"] / "client.zip"
        ).resolve()
        if not path_to_client.exists():
            raise HTTPException(status_code=500, detail="Could not find client.")
        return FileResponse(path_to_client, media_type="application/zip")

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
        check_inputs(model_name, module_name, input_shape)
        uid = str(uuid.uuid4())
        key_bytes = await key.read()
        dump_key(key_bytes, uid)
        return {"uid": uid}

    def stream_response(
        encrypted_results: bytes, chunk_size: int = 1024 * 1024
    ) -> Generator[bytes, None, None]:
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
        check_inputs(model_name, module_name, input_shape)

        # Read the uploaded file first to avoid including this I/O time in FHE inference runtime measurement.
        logger.info("Reading uploaded data...")
        start_read = time.time()
        uploaded_data = await model_input.read()
        logger.info(f"Uploaded data read in {time.time() - start_read} seconds")

        logger.info("Loading key...")
        start = time.time()
        key_bytes = load_key(uid)
        logger.info(f"Key loaded in {time.time() - start} seconds")

        logger.info("Loading circuit...")
        start = time.time()
        fhe = get_circuit(model_name, module_name, input_shape)
        logger.info(f"Circuit loaded in {time.time() - start} seconds")

        logger.info("Running FHE inference...")
        start = time.time()
        encrypted_results = fhe.run(
            serialized_encrypted_quantized_data=uploaded_data,
            serialized_evaluation_keys=key_bytes,
        )
        logger.info(f"FHE inference completed in {time.time() - start} seconds")
        logger.info(f"Results size is {len(encrypted_results) / (1024 ** 2)} Mb")

        return StreamingResponse(stream_response(encrypted_results))

    uvicorn.run(app, host="0.0.0.0", port=int(PORT))
