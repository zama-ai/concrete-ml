import json
import os
import time
from functools import partial
from importlib.metadata import version
from pathlib import Path

import torch
from concrete.compiler import check_gpu_available, check_gpu_enabled
from concrete.fhe import Exactness
from concrete.fhe.compilation.configuration import Configuration
from models import cnv_2w2a
from torch.utils.data import DataLoader
from trainer import get_test_set

from concrete.ml.quantization import QuantizedModule
from concrete.ml.torch.compile import compile_brevitas_qat_model

CURRENT_DIR = Path(__file__).resolve().parent
KEYGEN_CACHE_DIR = CURRENT_DIR.joinpath(".keycache")

# Add MPS (for macOS with Apple Silicon or AMD GPUs) support when error is fixed. For now, we
# observe a decrease in torch's top1 accuracy when using MPS devices
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3953

# For PyTorch operations, we use CPU (simpler and avoids device mismatch issues)
DEVICE = "cpu"
# For FHE compilation and execution, we use GPU if available
COMPILATION_DEVICE = "cuda" if check_gpu_available() else "cpu"

NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", 1))
P_ERROR = float(os.environ.get("P_ERROR", 0.01))

# GPU Verification Logging
print("=" * 50)
print("🔍 GPU VERIFICATION & DEVICE INFO")
print("=" * 50)
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"Concrete GPU enabled: {check_gpu_enabled()}")
print(f"Concrete GPU available: {check_gpu_available()}")
print(f"PyTorch DEVICE: {DEVICE} (CPU for PyTorch operations)")
print(f"FHE COMPILATION_DEVICE: {COMPILATION_DEVICE} (GPU used for FHE operations)")
print(f"CML_USE_GPU environment variable: {os.environ.get('CML_USE_GPU', 'Not set')}")

if torch.cuda.is_available():
    try:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Available GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        print(f"CUDA Version: {torch.version.cuda}")
        if COMPILATION_DEVICE == "cuda":
            print("✅ GPU will be used for FHE operations")
        else:
            print("⚠️  GPU available but Concrete GPU not enabled")
    except Exception as e:
        print(f"Error getting GPU info: {e}")
else:
    print("No CUDA GPU detected")

print("=" * 50)


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

# Keep model on CPU for PyTorch operations
# (GPU will be used only for FHE operations)

# Import and load the CIFAR test dataset
test_set = get_test_set(dataset="CIFAR10", datadir=CURRENT_DIR.joinpath(".datasets/"))
test_loader = DataLoader(test_set, batch_size=100, shuffle=False)

# Get the first sample
x, labels = next(iter(test_loader))

# Parameter `enable_unsafe_features` and `use_insecure_key_cache` are needed in order to be able to
# cache generated keys through `insecure_key_cache_location`. As the name suggests, these
# parameters are unsafe and should only be used for debugging in development
# Multi-parameter strategy is used in order to speed-up the FHE executions
configuration = Configuration(
    dump_artifacts_on_unexpected_failures=False,
    enable_unsafe_features=True,
    use_insecure_key_cache=True,
    insecure_key_cache_location=KEYGEN_CACHE_DIR,
)

print("🔧 Compiling the model...")
print(f"   FHE target device: {COMPILATION_DEVICE}")
print(f"   Ensuring model and input are on CPU for ONNX export")

# Explicitly ensure model is on CPU before compilation
# The ONNX export requires both model and input to be on the same device
torch_model_cpu = torch_model.cpu()
x_cpu = x.cpu()

quantized_numpy_module, compilation_execution_time = measure_execution_time(
    compile_brevitas_qat_model
)(
    torch_model_cpu,
    x_cpu,
    configuration=configuration,
    rounding_threshold_bits={"method": Exactness.APPROXIMATE, "n_bits": 6},
    p_error=P_ERROR,
    device=COMPILATION_DEVICE,  # This controls where FHE circuit runs, not ONNX export
)
assert isinstance(quantized_numpy_module, QuantizedModule)

print(
    f"✅ Compilation completed in {compilation_execution_time:.2f} seconds using {COMPILATION_DEVICE.upper()}"
)

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
print("🔑 Creating private and evaluation keys...")
_, keygen_execution_time = measure_execution_time(quantized_numpy_module.fhe_circuit.keygen)(
    force=True
)
print(f"✅ Keygen completed in {keygen_execution_time:.2f} seconds")

# Data torch to numpy
x_numpy = x.numpy()

# Initialize a list to store all the results
all_results = []

# Iterate through the NUM_SAMPLES
print(f"\n🚀 Starting FHE inference for {NUM_SAMPLES} sample(s)...")
for image_index in range(NUM_SAMPLES):
    print(f"\n--- Processing sample {image_index + 1}/{NUM_SAMPLES} ---")

    # Take one example
    test_x_numpy = x_numpy[image_index : image_index + 1]

    # Get the torch prediction
    torch_output = torch_model(x[image_index : image_index + 1])

    # Quantize the input
    q_x_numpy, quantization_execution_time = measure_execution_time(
        quantized_numpy_module.quantize_input
    )(test_x_numpy)

    print(f"⚙️  Quantization: {quantization_execution_time:.3f}s")
    print(f"📊 Clear input size: {q_x_numpy.nbytes} bytes")

    expected_quantized_prediction, clear_inference_time = measure_execution_time(
        partial(quantized_numpy_module.fhe_circuit.simulate)
    )(q_x_numpy)
    print(f"🧪 Simulation: {clear_inference_time:.3f}s")

    # Encrypt the input
    encrypted_q_x_numpy, encryption_execution_time = measure_execution_time(
        quantized_numpy_module.fhe_circuit.encrypt
    )(q_x_numpy)
    print(f"🔒 Encryption: {encryption_execution_time:.3f}s")

    print(f"📈 Encrypted input size: {quantized_numpy_module.fhe_circuit.size_of_inputs} bytes")
    print(f"📉 Encrypted output size: {quantized_numpy_module.fhe_circuit.size_of_outputs} bytes")
    print(
        f"🔑 Key sizes - Switch: {quantized_numpy_module.fhe_circuit.size_of_keyswitch_keys} bytes, Bootstrap: {quantized_numpy_module.fhe_circuit.size_of_bootstrap_keys} bytes"
    )
    print(f"🔍 Circuit complexity: {quantized_numpy_module.fhe_circuit.complexity}")

    print(f"\n🔥 Running FHE inference (using {COMPILATION_DEVICE.upper()})...")
    fhe_output, fhe_execution_time = measure_execution_time(quantized_numpy_module.fhe_circuit.run)(
        encrypted_q_x_numpy
    )
    print(f"⚡ FHE inference completed: {fhe_execution_time:.3f}s")

    # Decrypt the result
    decrypted_fhe_output, decryption_execution_time = measure_execution_time(
        quantized_numpy_module.fhe_circuit.decrypt
    )(fhe_output)
    print(f"🔓 Decryption: {decryption_execution_time:.3f}s")
    print(f"✅ Sample {image_index + 1} completed!")

    result = {
        "image_index": image_index,
        # Timings
        "quantization_time": quantization_execution_time,
        "encryption_time": encryption_execution_time,
        "fhe_time": fhe_execution_time,
        "decryption_time": decryption_execution_time,
        "inference_time": clear_inference_time,
        "label": labels[image_index].item(),
        "p_error": P_ERROR,
    }

    for prediction_index, prediction in enumerate(expected_quantized_prediction[0]):
        result[f"quantized_prediction_{prediction_index}"] = prediction
    for prediction_index, prediction in enumerate(decrypted_fhe_output[0]):
        result[f"prediction_{prediction_index}"] = prediction
    for prediction_index, prediction in enumerate(torch_output[0]):
        result[f"torch_prediction_{prediction_index}"] = prediction.item()

    all_results.append(result)

print("\n" + "=" * 50)
print("🎯 CIFAR-10 FHE BENCHMARK COMPLETED")
print("=" * 50)
print(f"📊 Processed {NUM_SAMPLES} sample(s)")
print(f"🔧 FHE compilation device: {COMPILATION_DEVICE}")
print(f"🖥️  PyTorch device: {DEVICE}")
print(f"🔐 P-error: {P_ERROR}")
if torch.cuda.is_available() and COMPILATION_DEVICE == "cuda":
    print(f"🎮 GPU used for FHE: {torch.cuda.get_device_name(0)}")
elif torch.cuda.is_available():
    print(f"⚠️  GPU available but not used: {torch.cuda.get_device_name(0)}")
print("=" * 50)

# Write the results to a CSV file
with open("inference_results.csv", "w", encoding="utf-8") as file:
    # Write the header row
    columns = list(all_results[0].keys())
    file.write(",".join(columns) + "\n")

    # Write the data rows
    for result in all_results:
        file.write(",".join(str(result[column]) for column in columns) + "\n")

metadata = {
    "p_error": P_ERROR,
    "cml_version": version("concrete-ml"),
    "cnp_version": version("concrete-python"),
    # Device and GPU information for benchmark differentiation
    "fhe_compilation_device": COMPILATION_DEVICE,
    "pytorch_device": DEVICE,  # Always CPU in this setup
    "cuda_available": torch.cuda.is_available(),
    "gpu_enabled": check_gpu_enabled(),
    "gpu_available": check_gpu_available(),
    "cml_use_gpu": os.environ.get("CML_USE_GPU", "Not set"),
    "gpu_usage": "fhe_only",  # GPU used only for FHE operations, not PyTorch
}

# Add GPU-specific information if available
if torch.cuda.is_available():
    try:
        metadata.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
                "cuda_version": torch.version.cuda,
            }
        )
    except Exception as e:
        metadata["gpu_info_error"] = str(e)
with open("metadata.json", "w") as file:
    json.dump(metadata, file)
