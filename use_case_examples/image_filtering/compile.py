"A script to manually compile all filters."

import json
import shutil

import numpy as np
import onnx
from common import AVAILABLE_FILTERS, FILTERS_PATH, INPUT_SHAPE, INPUTSET, KEYS_PATH
from custom_client_server import CustomFHEClient, CustomFHEDev

print("Starting compiling the filters.")

for image_filter in AVAILABLE_FILTERS:
    print("\nCompiling filter:", image_filter)

    # Load the onnx model
    onnx_model = onnx.load(FILTERS_PATH / f"{image_filter}/server.onnx")

    deployment_path = FILTERS_PATH / f"{image_filter}/deployment"

    # Retrieve the client API related to the current filter
    model = CustomFHEClient(deployment_path, KEYS_PATH).model

    image_shape = INPUT_SHAPE + (3,)

    # Compile the model using the loaded onnx model
    model.compile(INPUTSET, onnx_model=onnx_model)

    processing_json_path = deployment_path / "serialized_processing.json"

    # Load the serialized_processing.json file
    with open(processing_json_path, "r") as f:
        serialized_processing = json.load(f)

    # Delete the deployment folder and its content if it exist
    if deployment_path.is_dir():
        shutil.rmtree(deployment_path)

    # Save the files needed for deployment
    fhe_api = CustomFHEDev(model=model, path_dir=deployment_path)
    fhe_api.save()

    # Write the serialized_processing.json file to the deployment folder
    with open(processing_json_path, "w") as f:
        json.dump(serialized_processing, f)

print("Done!")
