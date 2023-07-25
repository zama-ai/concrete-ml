import os
import time
from functools import partial
from pathlib import Path

import torch
from concrete.fhe.compilation.configuration import Configuration
from models import cnv_2w2a
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from concrete import fhe
from concrete.ml.deployment.fhe_client_server import FHEModelDev
from concrete.ml.quantization import QuantizedModule
from concrete.ml.torch.compile import compile_brevitas_qat_model

CURRENT_DIR = Path(__file__).resolve().parent
KEYGEN_CACHE_DIR = CURRENT_DIR.joinpath(".keycache")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", 1))


def measure_execution_time(func):
    """Run a function and return execution time and outputs.

    Usage:
        def f(x, y):
            return x + y
        output, execution_time = measure_execution_time(f)(x,y)
    """

    def wrapper(*args, **kwargs):
        # Get the current time
        start = time.time()

        # Call the function
        result = func(*args, **kwargs)

        # Get the current time again
        end = time.time()

        # Calculate the execution time
        execution_time = end - start

        # Return the result and the execution time
        return result, execution_time

    return wrapper


# Instantiate the model
torch_model = cnv_2w2a(pre_trained=False)
torch_model.eval()


# Load the saved parameters using the available checkpoint
checkpoint = torch.load(
    CURRENT_DIR.joinpath("experiments/CNV_2W2A_2W2A_20221114_131345/checkpoints/best.tar"),
    map_location=DEVICE,
)
torch_model.load_state_dict(checkpoint["state_dict"], strict=False)

# Get some data that will be used to compile the model to FHE standards
transform_to_tensor = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: 2.0 * x - 1.0),
    ]  # Normalizes data between -1 and +1
)

builder = CIFAR10

test_set = builder(
    root=CURRENT_DIR.joinpath(".datasets/"),
    train=False,
    download=True,
    transform=transform_to_tensor,
)

test_loader = DataLoader(test_set, batch_size=100, shuffle=False)

x, labels = next(iter(test_loader))

cfg = Configuration(
    dump_artifacts_on_unexpected_failures=False,
    enable_unsafe_features=True,  # Needed to use the insecure key cache location
    use_insecure_key_cache=True,  #  Needed to use the insecure key cache location
    insecure_key_cache_location=KEYGEN_CACHE_DIR,
)

print("Compiling the model.")
quantized_numpy_module: QuantizedModule
quantized_numpy_module, compilation_execution_time = measure_execution_time(
    compile_brevitas_qat_model
)(torch_model, x, configuration=cfg, rounding_threshold_bits=6, p_error=0.01)

# Save the client/server files to disk.
dev = FHEModelDev(path_dir="./client_server", model=quantized_numpy_module)
dev.save()

print(f"Compilation time took {compilation_execution_time} seconds")

# Display the max bit-width in the model
print(
    "Max bit-width used in the circuit: ",
    f"{quantized_numpy_module.fhe_circuit.graph.maximum_integer_bit_width()} bits",
)

# Save the graph and mlir
print("Saving graph and mlir to disk.")
open("cifar10.graph", "w").write(str(quantized_numpy_module.fhe_circuit))
open("cifar10.mlir", "w").write(quantized_numpy_module.fhe_circuit.mlir)

# Key generation
print("Creation of the private and evaluation keys.")
_, keygen_execution_time = measure_execution_time(quantized_numpy_module.fhe_circuit.keygen)(
    force=True
)
print(f"Keygen took {keygen_execution_time} seconds")

# Data torch to numpy
x_numpy = x.numpy()

# Initialize a list to store all the results
all_results = []

# Iterate through the NUM_SAMPLES
for image_index in range(NUM_SAMPLES):
    # Take one example
    test_x_numpy = x_numpy[image_index : image_index + 1]

    # Get the torch prediction
    torch_output = torch_model(x[image_index : image_index + 1])

    # Quantize the input
    q_x_numpy, quantization_execution_time = measure_execution_time(
        quantized_numpy_module.quantize_input
    )(test_x_numpy)

    print(f"Quantization of a single input (image) took {quantization_execution_time} seconds")
    print(f"Size of CLEAR input is {q_x_numpy.nbytes} bytes\n")

    # Use new VL with .simulate() once CP's multi-parameter/precision bug is fixed
    # TODO: https://github.com/zama-ai/concrete-ml-internal/issues/3856
    p_error = quantized_numpy_module.fhe_circuit.p_error
    expected_quantized_prediction, clear_inference_time = measure_execution_time(
        partial(quantized_numpy_module.fhe_circuit.graph, p_error=p_error)
    )(q_x_numpy)

    # Encrypt the input
    encrypted_q_x_numpy, encryption_execution_time = measure_execution_time(
        quantized_numpy_module.fhe_circuit.encrypt
    )(q_x_numpy)
    print(f"Encryption of a single input (image) took {encryption_execution_time} seconds\n")

    print(f"Size of ENCRYPTED input is {quantized_numpy_module.fhe_circuit.size_of_inputs} bytes")
    print(f"Size of ENCRYPTED output is {quantized_numpy_module.fhe_circuit.size_of_outputs} bytes")
    print(
        f"Size of keyswitch key is {quantized_numpy_module.fhe_circuit.size_of_keyswitch_keys} bytes"
    )
    print(
        f"Size of bootstrap key is {quantized_numpy_module.fhe_circuit.size_of_bootstrap_keys} bytes"
    )
    print(f"Size of secret key is {quantized_numpy_module.fhe_circuit.size_of_secret_keys} bytes")
    print(f"Complexity is {quantized_numpy_module.fhe_circuit.complexity}\n")

    print("Running FHE inference")
    fhe_output, fhe_execution_time = measure_execution_time(quantized_numpy_module.fhe_circuit.run)(
        encrypted_q_x_numpy
    )
    print(f"FHE inference over a single image took {fhe_execution_time}")

    # Decrypt print the result
    decrypted_fhe_output, decryption_execution_time = measure_execution_time(
        quantized_numpy_module.fhe_circuit.decrypt
    )(fhe_output)
    print(f"Expected prediction: {expected_quantized_prediction}")
    print(f"Decrypted prediction: {decrypted_fhe_output}")

    result = {
        "image_index": image_index,
        # Timings
        "quantization_time": quantization_execution_time,
        "encryption_time": encryption_execution_time,
        "fhe_time": fhe_execution_time,
        "decryption_time": decryption_execution_time,
        "inference_time": clear_inference_time,
        "label": labels[image_index].item(),
    }

    for prediction_index, prediction in enumerate(expected_quantized_prediction[0]):
        result[f"quantized_prediction_{prediction_index}"] = prediction
    for prediction_index, prediction in enumerate(decrypted_fhe_output[0]):
        result[f"prediction_{prediction_index}"] = prediction
    for prediction_index, prediction in enumerate(torch_output[0]):
        result[f"torch_prediction_{prediction_index}"] = prediction.item()

    all_results.append(result)


# Write the results to a CSV file
with open("inference_results.csv", "w", encoding="utf-8") as file:
    # Write the header row
    columns = list(all_results[0].keys())
    file.write(",".join(columns) + "\n")

    # Write the data rows
    for result in all_results:
        file.write(",".join(str(result[column]) for column in columns) + "\n")
