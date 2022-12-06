"""Server that will listen for GET and POST requests from the client."""

from typing import List
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import time
from fastapi.responses import Response

from custom_client_server import CustomFHEServer
from common import FILTERS_PATH

class FilterRequest(BaseModel):
    filter: str

# Initialize an instance of FastAPI
app = FastAPI()

# Define the default route 
@app.get("/")
def root():
    return {"message": "Welcome to Your Image FHE Filter Server!"}

@app.post("/filter_image")
def filter_image(
    filter: str = Form(),
    files: List[UploadFile] = File(),
):
    encrypted_image = files[0].file.read()
    evaluation_key = files[1].file.read()

    #  Load the model
    fhe_model = CustomFHEServer(FILTERS_PATH / f"{filter}/deployment")
    
    # Run the FHE execution
    start = time.time()
    encrypted_output_image = fhe_model.run(encrypted_image, evaluation_key)
    print(f"FHE execution in {time.time() - start:0.2f}s")

    return Response(encrypted_output_image)
