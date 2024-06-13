import argparse
import json
import time
from pathlib import Path

import torch
from concrete.fhe import Configuration
from resnet import ResNet18_Weights, resnet18_custom
from utils_resnet import TinyImageNetProcessor
import numpy as np

from concrete.ml.torch.compile import compile_torch_model

parser = argparse.ArgumentParser(description="Run ResNet18 model with FHE execution.")
parser.add_argument("--run_fhe", action="store_true", help="Run the actual FHE execution.")
parser.add_argument(
    "--export_statistics", action="store_true", help="Export the circuit statistics."
)
args = parser.parse_args()
BASE_DIR = Path(__file__).resolve().parent

# Load the ResNet18 model with pretrained weights
resnet18 = resnet18_custom(weights=ResNet18_Weights.IMAGENET1K_V1)

CALIBRATION_SAMPLES = 10
NUM_TEST_SAMPLES = 100

imagenet_classes_path = BASE_DIR / "LOC_synset_mapping.txt"
processor = TinyImageNetProcessor(imagenet_classes_path)
all_images, all_labels = processor.get_image_label_tensors(num_samples=NUM_TEST_SAMPLES + CALIBRATION_SAMPLES)

# Split into calibration and test sets
calib_images, _ = all_images[:CALIBRATION_SAMPLES], all_labels[:CALIBRATION_SAMPLES]
images, labels = all_images[CALIBRATION_SAMPLES:], all_labels[CALIBRATION_SAMPLES:]

# Forward pass through the model to get the predictions
with torch.no_grad():
    outputs = resnet18(images)

# Compute and print accuracy
accuracy = processor.compute_accuracy(outputs, labels)
print(f"Accuracy of the ResNet18 model on the images: {accuracy*100:.2f}%")

topk_accuracy = processor.compute_topk_accuracy(outputs, labels, k=5)
print(f"Top-5 Accuracy of the ResNet18 model on the images: {topk_accuracy*100:.2f}%")

# Enable TLU fusing to optimize the number of TLUs in the residual connections
config = Configuration(enable_tlu_fusing=True, print_tlu_fusing=False)

# Compile the model
print("Compiling the model...")
q_module = compile_torch_model(
    resnet18,
    torch_inputset=images,
    n_bits={"model_inputs": 8, "op_inputs": 6, "op_weights": 6, "model_outputs": 8},
    rounding_threshold_bits={"n_bits": 7, "method": "APPROXIMATE"},
    p_error=0.05,
    configuration=config,
)

if args.export_statistics:
    open("resnet.graph", "w").write(q_module.fhe_circuit.graph.format(show_locations=True))
    open("resnet.mlir", "w").write(q_module.fhe_circuit.mlir)

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {str(key): make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(element) for element in obj]
        elif isinstance(obj, tuple):
            return tuple(make_serializable(element) for element in obj)
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

    statistics = make_serializable(q_module.fhe_circuit.statistics)

    with open("resnet_statistics.json", "w") as f:
        json.dump(statistics, f)

print("Model compiled successfully.")

# Forward pass with FHE disabled
with torch.no_grad():
    outputs_disable = q_module.forward(images.detach().numpy(), fhe="disable")

# Compute and print accuracy for quantized model
accuracy = processor.compute_accuracy(torch.from_numpy(outputs_disable), labels)
print(f"Quantized Model Accuracy of the FHEResNet18 on the images: {accuracy*100:.2f}%")
topk_accuracy = processor.compute_topk_accuracy(torch.from_numpy(outputs_disable), labels, k=5)
print(f"Quantized Model Top-5 Accuracy of the FHEResNet18 on the images: {topk_accuracy*100:.2f}%")

# Forward pass with FHE simulation
with torch.no_grad():
    outputs_simulate = q_module.forward(images.detach().numpy(), fhe="simulate")

# Compute and print accuracy for FHE simulation
accuracy = processor.compute_accuracy(torch.from_numpy(outputs_simulate), labels)
print(f"FHE Simulation Accuracy of the FHEResNet18 on the images: {accuracy*100:.2f}%")
topk_accuracy = processor.compute_topk_accuracy(torch.from_numpy(outputs_simulate), labels, k=5)
print(f"FHE Simulation Top-5 Accuracy of the FHEResNet18 on the images: {topk_accuracy*100:.2f}%")

if args.run_fhe:
    # Run FHE execution and measure time on a single image
    # q_module.fhe_circuit.keygen()
    single_image = images[0:1].detach().numpy()
    
    start = time.time()
    fhe_output = q_module.forward(single_image, fhe="execute")
    end = time.time()
    print(f"Time taken for one FHE execution: {end - start:.4f} seconds")

else:
    print("FHE execution was not run. Use --run_fhe to enable it.")