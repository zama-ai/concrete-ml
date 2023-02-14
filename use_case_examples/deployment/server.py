"""Deployment server

Routes:
    - Get client.zip
        - i.e. something to generate the key
    - Add a key
        - 
        - return an ID
    - Compute
        - give encrypted input and 

"""
import io
import os
import uuid
from pathlib import Path
from typing import Dict

import concrete.numpy as cnp
import uvicorn
from fastapi import FastAPI, Form, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from concrete.ml.deployment import FHEModelServer

app = FastAPI(debug=False)

FILE_FOLDER = Path(__file__).parent
CLIENT_SERVER_PATH = FILE_FOLDER / Path("dev")
KEY_PATH = FILE_FOLDER / Path("server_keys")
PORT = os.environ.get("PORT", "5000")

fhe = FHEModelServer(str(CLIENT_SERVER_PATH.resolve()))
client = cnp.Client.load(CLIENT_SERVER_PATH / "client.zip", KEY_PATH)

KEYS: Dict[str, bytes] = {}


@app.get("/get_client")
def get_client():
    return FileResponse((CLIENT_SERVER_PATH / "client.zip").resolve(), media_type="application/zip")


@app.get("/get_processing")
def get_processing():
    return FileResponse(
        (CLIENT_SERVER_PATH / "serialized_processing.json").resolve(),
        media_type="application/json",
    )


@app.post("/add_key")
async def add_key(key: UploadFile):
    uid = str(uuid.uuid4())
    KEYS[uid] = await key.read()
    return {"uid": uid}


@app.post("/compute")
async def compute(model_input: UploadFile, uid: str = Form()):
    key = KEYS[uid]
    encrypted_results = fhe.run(
        serialized_encrypted_quantized_data=await model_input.read(),
        serialized_evaluation_keys=key,
    )
    return StreamingResponse(
        io.BytesIO(encrypted_results),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(PORT))
