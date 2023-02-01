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
import base64
import os
import uuid
from pathlib import Path
from typing import Dict

import concrete.numpy as cnp
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from concrete.ml.deployment import FHEModelServer

app = FastAPI(debug=False)

CLIENT_SERVER_PATH = Path("./dev")
KEY_PATH = Path("./server_keys")
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


class EvaluationKey(BaseModel):
    key: str


@app.post("/add_key")
def add_key(key: EvaluationKey):
    key_bytes = base64.b64decode(key.key)
    uid = str(uuid.uuid4())
    KEYS[uid] = key_bytes
    return {"uid": uid}


class Inputs(BaseModel):
    uid: str
    inputs: str


@app.post("/compute")
def compute(inputs: Inputs):
    key = KEYS[inputs.uid]
    encrypted_results = fhe.run(
        serialized_encrypted_quantized_data=base64.b64decode(inputs.inputs),
        serialized_evaluation_keys=key,
    )
    return {"result": base64.b64encode(encrypted_results).decode("ascii")}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(PORT))
