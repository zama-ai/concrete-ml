import os
import sys
import time
from pathlib import Path

import torch
from concrete.numpy.compilation.configuration import Configuration
from models import cnv_2w2a
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from concrete.ml.quantization import QuantizedModule
from concrete.ml.torch.compile import compile_brevitas_qat_model

CURRENT_DIR = Path(__file__).resolve().parent
KEYGEN_CACHE_DIR = CURRENT_DIR.joinpath(".keycache")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
torch_model = cnv_2w2a(pretrained=False)
torch_model.eval()


# Load the saved parameters using the available checkpoint
checkpoint = torch.load(
    CURRENT_DIR.joinpath("experiments/CNV_2W2A_2W2A_20221114_131345/checkpoints/best.tar"),
    map_location=DEVICE,
)
torch_model.load_state_dict(checkpoint["state_dict"], strict=False)

# Let's get some data that will be used to compile the model to FHE standards
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

test_loader = DataLoader(test_set, batch_size=100, shuffle=True)

x = next(iter(test_loader))[0]

cfg = Configuration(
    dump_artifacts_on_unexpected_failures=False,
    enable_unsafe_features=True,  # Needed to use the insecure key cache location
    use_insecure_key_cache=True,  #  Needed to use the insecure key cache location
    insecure_key_cache_location=KEYGEN_CACHE_DIR,
    p_error=None,  # To avoid any confusion: we are always using kwarg p_error
    global_p_error=None,  # To avoid any confusion: we are always using kwarg global_p_error
)

print("Compiling the model.")
quantized_numpy_module: QuantizedModule
quantized_numpy_module, execution_time = measure_execution_time(compile_brevitas_qat_model)(
    torch_model,
    x,
    use_virtual_lib=False,
    configuration=cfg,
)

print(f"Compilation time took {execution_time} seconds")

# Display the max bitwidth in the model
print(
    "Max bitwidth used in the circuit: ",
    f"{quantized_numpy_module.fhe_circuit.graph.maximum_integer_bit_width()} bits",
)

# Save the graph and mlir
print("Saving graph and mlir to disk.")
open("cifar10.graph", "w").write(str(quantized_numpy_module.fhe_circuit))
open("cifar10.mlir", "w").write(quantized_numpy_module.fhe_circuit.mlir)

# Key generation
print("Creation of the private and evaluation keys.")
_, execution_time = measure_execution_time(quantized_numpy_module.fhe_circuit.keygen)(force=True)
print(f"Keygen took {execution_time} seconds")

# List the files in the directory
files = KEYGEN_CACHE_DIR.glob("**/*")

# Print the size of each file
for file in files:
    file_size = file.stat().st_size
    print(f"{file}: {file_size} bytes")

# Data torch to numpy
x_numpy = x.numpy()

# Take only one example
one_x_numpy = x_numpy[:1]

# Quantize the input
q_x_numpy, execution_time = measure_execution_time(quantized_numpy_module.quantize_input)(
    one_x_numpy
)
print(f"Quantization of a single input (image) took {execution_time} seconds")
print(f"Size of CLEAR input is {sys.getsizeof(q_x_numpy)}KB\n")

expected_prediction = quantized_numpy_module.forward(q_x_numpy)

# Encrypt the input
encrypted_q_x_numpy, execution_time = measure_execution_time(
    quantized_numpy_module.fhe_circuit.encrypt
)(q_x_numpy)
print(f"Encryption of a single input (image) took {execution_time} seconds")
serialized_encrypted_q_x_numpy = encrypted_q_x_numpy.serialize()
print(f"Size of ENCRYPTED input is {sys.getsizeof(serialized_encrypted_q_x_numpy)}KB\n")


print("Running FHE inference")
prediction, execution_time = measure_execution_time(quantized_numpy_module.fhe_circuit.run)(
    encrypted_q_x_numpy
)
print(f"FHE inference over a single image took {execution_time} seconds")
serialized_prediction = prediction.serialize()
print(f"Size of the prediction is {sys.getsizeof(serialized_prediction)}KB\n")

# Decrypt print the result
decrypted_prediction = quantized_numpy_module.fhe_circuit.decrypt(prediction)
print(f"Expected prediction: {expected_prediction}")
print(f"Decrypted prediction: {decrypted_prediction}")
