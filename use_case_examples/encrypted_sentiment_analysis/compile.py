import onnx
import pandas as pd
from concrete.ml.deployment import FHEModelDev, FHEModelClient
from concrete.ml.onnx.convert import get_equivalent_numpy_forward
import json
import os
import shutil
from pathlib import Path


script_dir = Path(__file__).parent

print("Compiling the model...")

# Load the onnx model
model_onnx = onnx.load(Path.joinpath(script_dir, "sentiment_fhe_model/server_model.onnx"))

# Load the data from the csv file to be used for compilation
data = pd.read_csv(
    Path.joinpath(script_dir, "sentiment_fhe_model/samples_for_compilation.csv"), index_col=0
).values

# Convert the onnx model to a numpy model
_tensor_tree_predict = get_equivalent_numpy_forward(model_onnx)

model = FHEModelClient(
    Path.joinpath(script_dir, "sentiment_fhe_model/deployment"), ".fhe_keys"
).model

# Assign the numpy model and compile the model
model._tensor_tree_predict = _tensor_tree_predict

# Compile the model
model.compile(data)

# Load the serialized_processing.json file
with open(
    Path.joinpath(script_dir, "sentiment_fhe_model/deployment/serialized_processing.json"), "r"
) as f:
    serialized_processing = json.load(f)

# Delete the deployment folder if it exist
if Path.joinpath(script_dir, "sentiment_fhe_model/deployment").exists():
    shutil.rmtree(Path.joinpath(script_dir, "sentiment_fhe_model/deployment"))

fhe_api = FHEModelDev(
    model=model, path_dir=Path.joinpath(script_dir, "sentiment_fhe_model/deployment")
)
fhe_api.save()

# Write the serialized_processing.json file to the deployment folder
with open(
    Path.joinpath(script_dir, "sentiment_fhe_model/deployment/serialized_processing.json"), "w"
) as f:
    json.dump(serialized_processing, f)

print("Done!")
