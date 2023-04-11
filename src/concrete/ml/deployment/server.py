"""Deployment server.

Routes:
    - Get client.zip
    - Add a key
    - Compute
"""

import io
import os
import uuid

# pylint: disable=import-error
import zipfile
from pathlib import Path
from typing import Dict

import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from ..deployment import FHEModelServer

if __name__ == "__main__":
    app = FastAPI(debug=False)

    FILE_FOLDER = Path(__file__).parent

    KEY_PATH = Path(os.environ.get("KEY_PATH", FILE_FOLDER / Path("server_keys")))
    CLIENT_SERVER_PATH = Path(os.environ.get("PATH_TO_MODEL", FILE_FOLDER / Path("dev")))
    PORT = os.environ.get("PORT", "5000")

    fhe = FHEModelServer(str(CLIENT_SERVER_PATH.resolve()))

    KEYS: Dict[str, bytes] = {}

    PATH_TO_CLIENT = (CLIENT_SERVER_PATH / "client.zip").resolve()
    PATH_TO_SERVER = (CLIENT_SERVER_PATH / "server.zip").resolve()

    assert PATH_TO_CLIENT.exists()
    assert PATH_TO_SERVER.exists()

    # remove this hack once 1.x is released
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3251
    PATH_TO_PROCESSING = CLIENT_SERVER_PATH / "serialized_processing.json"
    if not PATH_TO_PROCESSING.exists():  # then try to extract it from client
        with zipfile.ZipFile(PATH_TO_CLIENT) as client_zip:
            path_in_client = zipfile.Path(client_zip) / "serialized_processing.json"
            if not path_in_client.exists():
                raise FileNotFoundError(f"{path_in_client} does not exists")
            with client_zip.open("serialized_processing.json", mode="r") as file_in_json, open(
                PATH_TO_PROCESSING, mode="wb"
            ) as file_out:
                file_out.write(file_in_json.read())
    # populate client with processing too
    with zipfile.ZipFile(PATH_TO_CLIENT, "a") as zip_file:
        zip_file.write(filename=PATH_TO_PROCESSING, arcname="serialized_processing.json")

    @app.get("/get_client")
    def get_client():
        """Get client.

        Returns:
            FileResponse: client.zip

        Raises:
            HTTPException: if the file can't be find locally
        """
        path_to_client = (CLIENT_SERVER_PATH / "client.zip").resolve()
        if not path_to_client.exists():
            raise HTTPException(status_code=500, detail="Could not find client.")
        return FileResponse(path_to_client, media_type="application/zip")

    # Needed for legacy reasons (to remove once 1.x is released)
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3225
    @app.get("/get_processing")
    def get_processing():
        """Get processing.

        Returns:
            FileResponse: serialized_processing.json

        Raises:
            HTTPException: if the file can't be find locally
        """
        # Backward compatibility issue
        # This root has to be removed once 1.x is released

        path_to_processing = (CLIENT_SERVER_PATH / "serialized_processing.json").resolve()
        if not path_to_processing.exists():
            raise HTTPException(status_code=500, detail="Could not find client.")
        return FileResponse(
            path_to_processing,
            media_type="application/json",
        )

    @app.post("/add_key")
    async def add_key(key: UploadFile):
        """Add public key.

        Arguments:
            key (UploadFile): public key

        Returns:
            Dict[str, str]
                - uid: uid a personal uid
        """
        uid = str(uuid.uuid4())
        KEYS[uid] = await key.read()
        return {"uid": uid}

    @app.post("/compute")
    async def compute(model_input: UploadFile, uid: str = Form()):  # noqa: B008
        """Compute the circuit over encrypted input.

        Arguments:
            model_input (UploadFile): input of the circuit
            uid (str): uid of the public key to use

        Returns:
            StreamingResponse: the result of the circuit
        """
        key = KEYS[uid]
        encrypted_results = fhe.run(
            serialized_encrypted_quantized_data=await model_input.read(),
            serialized_evaluation_keys=key,
        )
        return StreamingResponse(
            io.BytesIO(encrypted_results),
        )

    uvicorn.run(app, host="0.0.0.0", port=int(PORT))
