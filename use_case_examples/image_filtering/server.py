"""Server that will listen for GET and POST requests from the client."""

from typing import List
from fastapi import FastAPI, File, UploadFile, Form
from pydantic import BaseModel
import time
from fastapi.responses import Response, JSONResponse

from custom_client_server import CustomFHEServer
from common import FILTERS_PATH, SERVER_TMP_PATH


def get_server_file_path(name, user_id, image_filter):
    """Get the correct temporary file path for the server.

    Args:
        name (str): The desired file name.
        user_id (int): The current user's ID.
        image_filter (str): The filter chosen by the user
    
    Returns:
        pathlib.Path: The file path.
    """
    return SERVER_TMP_PATH / f"{name}_{image_filter}_{user_id}"

class FilterRequest(BaseModel):
    filter: str

# Initialize an instance of FastAPI
app = FastAPI()

# Define the default route 
@app.get("/")
def root():
    return {"message": "Welcome to Your Image FHE Filter Server!"}

@app.post("/send_input")
def send_input(
    user_id: str = Form(),
    filter: str = Form(),
    files: List[UploadFile] = File(),
):
    encrypted_image_path = get_server_file_path("encrypted_image", filter, user_id)
    evaluation_key_path = get_server_file_path("evaluation_key", filter, user_id)

    with encrypted_image_path.open('wb') as encrypted_image, evaluation_key_path.open('wb') as evaluation_key:
        encrypted_image.write(files[0].file.read())
        evaluation_key.write(files[1].file.read())

@app.post("/run_fhe")
def run_fhe(
    user_id: str = Form(),
    filter: str = Form(),
):

    encrypted_image_path = get_server_file_path("encrypted_image", filter, user_id)
    evaluation_key_path = get_server_file_path("evaluation_key", filter, user_id)

    with encrypted_image_path.open("rb") as encrypted_image_file, evaluation_key_path.open("rb") as evaluation_key_file:
        encrypted_image = encrypted_image_file.read()
        evaluation_key = evaluation_key_file.read()

    #  Load the model
    fhe_model = CustomFHEServer(FILTERS_PATH / f"{filter}/deployment")
    
    # Run the FHE execution
    start = time.time()
    encrypted_output_image = fhe_model.run(encrypted_image, evaluation_key)
    fhe_execution_time = round(time.time() - start, 2)

    encrypted_output_path = get_server_file_path("encrypted_output", filter, user_id)

    with encrypted_output_path.open('wb') as encrypted_output:
        encrypted_output.write(encrypted_output_image)
    
    return JSONResponse(content=fhe_execution_time)

@app.post("/get_output")
def get_output(
    user_id: str = Form(),
    filter: str = Form(),
):
    encrypted_output_path = get_server_file_path("encrypted_output", filter, user_id)

    with encrypted_output_path.open("rb") as encrypted_output_file:
        encrypted_output = encrypted_output_file.read()

    return Response(encrypted_output)
