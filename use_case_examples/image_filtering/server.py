"""Server that will listen for GET and POST requests from the client."""

from fastapi import FastAPI
from pydantic import BaseModel
import base64
import time

from custom_client_server import CustomFHEServer
from common import FILTERS_PATH

class FilterRequest(BaseModel):
    filter: str
    evaluation_key: str
    encrypted_image: str

# Initialize an instance of FastAPI
app = FastAPI()

# Define the default route 
@app.get("/")
def root():
    return {"message": "Welcome to Your Image FHE Filter Server!"}

@app.post("/filter_image")
def transform_image(query: FilterRequest):

    # Decode the encrypted image and evaluation key
    encrypted_image = base64.b64decode(query.encrypted_image)
    evaluation_key = base64.b64decode(query.evaluation_key)

    start = time.time()

    #  Load the model
    fhe_model = CustomFHEServer(FILTERS_PATH / f"{query.filter}/deployment")
    
    # Run the FHE execution
    encrypted_output_image = fhe_model.run(encrypted_image, evaluation_key)
    print(f"FHE execution in {time.time() - start:0.2f}s")

    # Encode and decode the encrypted output
    encrypted_output_image = base64.b64encode(encrypted_output_image).decode()
    return {"encrypted_output_image": encrypted_output_image}
