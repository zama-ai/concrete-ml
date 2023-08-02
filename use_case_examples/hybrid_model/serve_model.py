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
from typing import Optional, Tuple, Union

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
        default=FILE_FOLDER / Path("user_keys"),
    )
    parser.add_argument(
        "--path-to-keys",
        dest="path_to_keys",
        type=Path,
        default=FILE_FOLDER / Path("compiled_models"),
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

    @app.post("/compute")
    async def compute(
        model_input: UploadFile,
        uid: str = Form(),
        model_name: str = Form(),
        module_name: str = Form(),
        input_shape: str = Form(),
    ):  # noqa: B008
        """Compute the circuit over encrypted input.

        Arguments:
            model_input (UploadFile): input of the circuit
            uid (str): uid of the public key to use

        Returns:
            StreamingResponse: the result of the circuit
        """
        check_inputs(model_name, module_name, input_shape)
        start = time.time()
        key_bytes = load_key(uid)
        end = time.time()
        logger.info(f"It took {end - start} seconds to load the key")

        start = time.time()
        fhe = get_circuit(model_name, module_name, input_shape)
        end = time.time()
        logger.info(f"It took {end - start} seconds to load the circuit")

        start = time.time()
        encrypted_results = fhe.run(
            serialized_encrypted_quantized_data=await model_input.read(),
            serialized_evaluation_keys=key_bytes,
        )
        end = time.time()
        logger.info(f"fhe inference of input of shape {input_shape} took {end - start}")
        logger.info(f"Results size is {len(encrypted_results)/(1024**2)} Mb")
        start = time.time()
        return StreamingResponse(
            io.BytesIO(encrypted_results),
        )

    uvicorn.run(app, host="0.0.0.0", port=int(PORT))
