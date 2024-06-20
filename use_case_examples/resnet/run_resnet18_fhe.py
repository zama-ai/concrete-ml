import argparse
import json
import time
from pathlib import Path

import torch
from concrete.fhe import Configuration
from resnet import ResNet18_Weights, resnet18_custom
from utils_resnet import TinyImageNetProcessor

from concrete.ml.torch.compile import compile_torch_model

BASE_DIR = Path(__file__).resolve().parent
CALIBRATION_SAMPLES = 10
NUM_TEST_SAMPLES = 100


def load_model():
    # Load the ResNet18 model with pre-trained weights
    return resnet18_custom(weights=ResNet18_Weights.IMAGENET1K_V1)


def load_data():
    imagenet_classes_path = BASE_DIR / "LOC_synset_mapping.txt"
    processor = TinyImageNetProcessor(imagenet_classes_path)
    all_images, all_labels = processor.get_image_label_tensors(
        num_samples=NUM_TEST_SAMPLES + CALIBRATION_SAMPLES
    )
    calib_images, _ = all_images[:CALIBRATION_SAMPLES], all_labels[:CALIBRATION_SAMPLES]
    images, labels = all_images[CALIBRATION_SAMPLES:], all_labels[CALIBRATION_SAMPLES:]
    return processor, calib_images, images, labels


def evaluate_model(model, processor, images, labels):
    with torch.no_grad():
        outputs = model(images)
    accuracy = processor.compute_accuracy(outputs, labels)
    topk_accuracy = processor.compute_topk_accuracy(outputs, labels, k=5)
    print(f"Accuracy of the ResNet18 model on the images: {accuracy*100:.2f}%")
    print(f"Top-5 Accuracy of the ResNet18 model on the images: {topk_accuracy*100:.2f}%")


def compile_model(model, images, use_gpu=False):
    # Enable TLU fusing to optimize the number of TLUs in the residual connections
    config = Configuration(enable_tlu_fusing=True, print_tlu_fusing=False, use_gpu=use_gpu)
    print("Compiling the model...")
    return compile_torch_model(
        model,
        torch_inputset=images,
        n_bits={"model_inputs": 8, "op_inputs": 6, "op_weights": 6, "model_outputs": 8},
        rounding_threshold_bits={"n_bits": 7, "method": "APPROXIMATE"},
        p_error=0.05,
        configuration=config,
    )


def export_statistics(q_module):
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


def evaluate_model_cml(q_module, processor, images, labels, fhe):
    assert fhe in ["disable", "simulate"], "fhe must be either 'disable' or 'simulate'"

    with torch.no_grad():
        outputs = q_module.forward(images.detach().numpy(), fhe=fhe)
    accuracy = processor.compute_accuracy(torch.from_numpy(outputs), labels)
    topk_accuracy = processor.compute_topk_accuracy(torch.from_numpy(outputs), labels, k=5)
    if fhe == "simulate":
        print("FHE simulation Accuracy of the FHEResNet18 on the images: " f"{accuracy*100:.2f}%")
        print(
            "FHE simulation Top-5 Accuracy of the FHEResNet18 on the images: "
            f"{topk_accuracy*100:.2f}%"
        )
    else:
        print("Quantized Model Accuracy of the FHEResNet18 on the images: " f"{accuracy*100:.2f}%")
        print(
            "Quantized Model Top-5 Accuracy of the FHEResNet18 on the images: "
            f"{topk_accuracy*100:.2f}%"
        )


def run_fhe_execution(q_module, images):
    single_image = images[0:1].detach().numpy()
    start = time.time()
    _ = q_module.forward(single_image, fhe="execute")
    end = time.time()
    print(f"Time taken for one FHE execution: {end - start:.4f} seconds")


def main():
    parser = argparse.ArgumentParser(description="Run ResNet18 model with FHE execution.")
    parser.add_argument("--run_fhe", action="store_true", help="Run the actual FHE execution.")
    parser.add_argument(
        "--export_statistics", action="store_true", help="Export the circuit statistics."
    )
    parser.add_argument(
        "--use_gpu", action="store_true", help="Use the available GPU at FHE runtime."
    )
    args = parser.parse_args()

    resnet18 = load_model()
    processor, calib_images, images, labels = load_data()

    evaluate_model(resnet18, processor, images, labels)

    q_module = compile_model(resnet18, calib_images, use_gpu=args.use_gpu)

    if args.export_statistics:
        export_statistics(q_module)

    print("Model compiled successfully.")

    evaluate_model_cml(q_module, processor, images, labels, fhe="disable")
    evaluate_model_cml(q_module, processor, images, labels, fhe="simulate")

    if args.run_fhe:
        run_fhe_execution(q_module, images)
    else:
        print("FHE execution was not run. Use --run_fhe to enable it.")


if __name__ == "__main__":
    main()
