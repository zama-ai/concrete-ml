"""Deployment server.

Routes:
    - Get client.zip
    - Add a key
    - Compute
"""

import io
import os
import uuid
from pathlib import Path
from typing import Dict

import uvicorn
from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

# No relative import here because when not used in the package itself
from concrete.ml.deployment import FHEModelServer

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
