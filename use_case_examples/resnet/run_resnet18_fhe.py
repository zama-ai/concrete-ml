import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from concrete.fhe import Configuration
from resnet import ResNet18_Weights, resnet18_custom
from utils_resnet import ImageNetProcessor

from concrete.ml.torch.compile import build_quantized_module, compile_torch_model

BASE_DIR = Path(__file__).resolve().parent
CALIBRATION_SAMPLES = 10
NUM_TEST_SAMPLES = 100


def load_model():
    # Load the ResNet18 model with pre-trained weights
    return resnet18_custom(weights=ResNet18_Weights.IMAGENET1K_V1)


def load_data():
    processor = ImageNetProcessor()
    calib_images = processor.get_calibration_data(CALIBRATION_SAMPLES)
    return processor, calib_images


def evaluate_model(model, processor):
    with torch.no_grad():
        device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        outputs = []
        all_labels = []
        for batch in processor.dataloader:
            batch_images = batch["pixel_values"].to(device)
            batch_outputs = model(batch_images)
            outputs.append(batch_outputs)
            all_labels.append(batch["labels"])

        outputs = torch.cat(outputs)
        outputs = outputs.cpu()
        all_labels = torch.cat(all_labels)
        all_labels = all_labels.cpu()

    accuracy = processor.accuracy(outputs, all_labels)
    topk_accuracy = processor.accuracy_top5(outputs, all_labels)
    print(f"Accuracy of the ResNet18 model on the images: {accuracy:.2f}%")
    print(f"Top-5 Accuracy of the ResNet18 model on the images: {topk_accuracy:.2f}%")


def compile_model(
    model, images, n_bits, rounding_threshold_bits=None, fhe_mode="disable", use_gpu=False
):
    """
    Compile the model using either build_quantized_module or compile_torch_model.

    Args:
        model: The PyTorch model to compile.
        images: The calibration images.
        n_bits: The number of bits for quantization (int). Can be a dictionary:
                    {
                        "model_inputs": 8,
                        "op_inputs": 6,
                        "op_weights": 6,
                        "model_outputs": 8
                    }
        rounding_threshold_bits: The rounding threshold bits.
        fhe_mode: The FHE mode ('disable' or 'simulate').
        use_gpu: Whether to use GPU for compilation.

    Returns:
        The compiled quantized module.
    """
    compile_config = {
        "n_bits": n_bits,
        "rounding_threshold_bits": (
            {"n_bits": rounding_threshold_bits, "method": "APPROXIMATE"}
            if rounding_threshold_bits is not None
            else None
        ),
    }

    if fhe_mode != "disable":
        config = Configuration(enable_tlu_fusing=True, print_tlu_fusing=False, use_gpu=use_gpu)
        compile_config.update(
            {
                "p_error": 0.05,
                "configuration": config,
            }
        )
        compile_func = compile_torch_model
    else:
        compile_func = build_quantized_module

    print(f"Compiling the model with {compile_func.__name__}...")
    return compile_func(model, torch_inputset=images, **compile_config)


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


def evaluate_model_cml(q_module, processor, fhe):
    assert fhe in ["disable", "simulate"], "fhe must be either 'disable' or 'simulate'"

    with torch.no_grad():
        outputs = []
        all_labels = []
        for batch in processor.dataloader:
            batch_images = batch["pixel_values"].detach().numpy()
            batch_outputs = q_module.forward(batch_images, fhe=fhe)
            outputs.append(batch_outputs)
            all_labels.append(batch["labels"].detach().numpy())

        outputs = torch.from_numpy(np.concatenate(outputs))
        all_labels = torch.from_numpy(np.concatenate(all_labels))

    accuracy = processor.accuracy(outputs, all_labels)
    topk_accuracy = processor.accuracy_top5(outputs, all_labels)
    if fhe == "simulate":
        print("FHE simulation Accuracy of the FHEResNet18 on the images: " f"{accuracy:.2f}%")
        print(
            "FHE simulation Top-5 Accuracy of the FHEResNet18 on the images: "
            f"{topk_accuracy:.2f}%"
        )
    else:
        print("Quantized Model Accuracy of the FHEResNet18 on the images: " f"{accuracy:.2f}%")
        print(
            "Quantized Model Top-5 Accuracy of the FHEResNet18 on the images: "
            f"{topk_accuracy:.2f}%"
        )


def run_fhe_execution(q_module, images, num_images=1):
    images = images[:num_images].detach().numpy()
    n_features = images.shape[1:]  # Get the shape of features (channels, height, width)

    def get_top5_labels(output):
        return np.argsort(output.flatten())[-5:][::-1]

    print(f"Processing {num_images} image(s)...")
    total_fhe_time = 0

    for i in range(num_images):
        print(f"\nImage {i+1}:")
        img = images[i].reshape(1, *n_features)  # Reshape to (1, *n_features)

        print("  Running FHE execution...")
        start = time.time()
        output_fhe = q_module.forward(img, fhe="execute")
        fhe_end = time.time()
        fhe_time = fhe_end - start
        total_fhe_time += fhe_time
        print(f"  FHE execution completed in {fhe_time:.4f} seconds")

        fhe_top5 = get_top5_labels(output_fhe)
        print("  FHE top 5 labels:", ", ".join(map(str, fhe_top5)))

        print("  Running simulation...")
        output_simulate = q_module.forward(img, fhe="simulate")

        sim_top5 = get_top5_labels(output_simulate)
        print("  Simulation top 5 labels:", ", ".join(map(str, sim_top5)))

    print(f"\nTotal FHE execution time for {num_images} image(s): {total_fhe_time:.4f} seconds")


def run_experiment(resnet18, calib_images, processor, fhe_mode="disable"):

    # Define ranges for n_bits and rounding_threshold_bits
    n_bits_range = range(2, 16)
    rounding_threshold_bits_range = list(range(2, 9)) + [None]  # 2 to 8 and None

    # Initialize a dictionary to store accuracies for each combination
    accuracies = {}

    total_combinations = len(n_bits_range) * len(rounding_threshold_bits_range)
    current_combination = 0

    # Loop over the ranges of n_bits and rounding_threshold_bits
    for n_bits in n_bits_range:
        for rounding_threshold_bits in rounding_threshold_bits_range:
            current_combination += 1
            print(f"\nProcessing combination {current_combination}/{total_combinations}")
            print(f"n_bits: {n_bits}, rounding_threshold_bits: {rounding_threshold_bits}")

            q_module = compile_model(
                resnet18, calib_images, n_bits, rounding_threshold_bits, fhe_mode
            )

            outputs = []
            all_labels = []

            print("Evaluating model...")
            for batch in processor.dataloader:
                batch_images = batch["pixel_values"].detach().numpy()
                batch_outputs = q_module.forward(batch_images, fhe=fhe_mode)
                outputs.append(batch_outputs)
                all_labels.append(batch["labels"].detach().numpy())

            outputs = torch.from_numpy(np.concatenate(outputs))
            all_labels = torch.from_numpy(np.concatenate(all_labels))

            # Calculate and store accuracy
            fhe_accuracy = processor.accuracy(outputs, all_labels)
            accuracies[(n_bits, rounding_threshold_bits)] = fhe_accuracy
            print(f"Accuracy: {fhe_accuracy:.4f}")

    # Convert accuracies to a 2D array for plotting
    accuracy_matrix = np.zeros((len(n_bits_range), len(rounding_threshold_bits_range)))
    for i, n_bits in enumerate(n_bits_range):
        for j, rounding_threshold_bits in enumerate(rounding_threshold_bits_range):
            accuracy_matrix[i, j] = accuracies[(n_bits, rounding_threshold_bits)]

    # Save the accuracy matrix to disk
    np.save("accuracy_matrix.npy", accuracy_matrix)

    print("\nGenerating plot...")

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(accuracy_matrix, cmap="viridis")
    fig.colorbar(cax)

    # Set ticks and labels
    ax.set_xticks(range(len(rounding_threshold_bits_range)))
    ax.set_xticklabels([str(x) for x in rounding_threshold_bits_range], rotation=45)
    ax.set_yticks(range(len(n_bits_range)))
    ax.set_yticklabels([str(x) for x in n_bits_range])
    ax.set_xlabel("Rounding Threshold Bits")
    ax.set_ylabel("N Bits")
    ax.set_title(f"Accuracy of FHE ({fhe_mode})")

    # Annotate each cell with the accuracy percentage
    for i in range(len(n_bits_range)):
        for j in range(len(rounding_threshold_bits_range)):
            ax.text(j, i, f"{accuracy_matrix[i, j]:.2f}", va="center", ha="center", color="white")

    plt.tight_layout()
    plt.savefig("accuracy_matrix.png", dpi=300)
    print("Plot saved as accuracy_matrix.png")


def main():
    parser = argparse.ArgumentParser(description="Run ResNet18 model with FHE execution.")
    parser.add_argument("--run_fhe", action="store_true", help="Run the actual FHE execution.")
    parser.add_argument(
        "--export_statistics", action="store_true", help="Export the circuit statistics."
    )
    parser.add_argument(
        "--use_gpu", action="store_true", help="Use the available GPU at FHE runtime."
    )
    parser.add_argument(
        "--run_experiment",
        action="store_true",
        help="Run the experiment with different n_bits and rounding_threshold_bits.",
    )
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default=None,
        help="Path to the directory where the dataset is cached.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=1,
        help="Number of images to process in FHE execution (default: 1)",
    )
    args = parser.parse_args()

    resnet18 = load_model()
    processor = ImageNetProcessor(
        NUM_TEST_SAMPLES, CALIBRATION_SAMPLES, cache_dir=args.dataset_cache_dir
    )
    calib_images = processor.get_calibration_tensor()

    evaluate_model(resnet18, processor)

    if args.run_experiment:
        # Get the test images and labels
        run_experiment(resnet18, calib_images, processor)
    else:
        q_module = compile_model(
            resnet18,
            calib_images,
            n_bits={"model_inputs": 8, "op_inputs": 7, "op_weights": 7, "model_outputs": 9},
            rounding_threshold_bits=7,
            fhe_mode="simulate",
            use_gpu=args.use_gpu,
        )

        if args.export_statistics:
            export_statistics(q_module)

        print("Model compiled successfully.")

        evaluate_model_cml(q_module, processor, fhe="disable")
        evaluate_model_cml(q_module, processor, fhe="simulate")

    if args.run_fhe:
        num_images = args.num_images
        run_fhe_execution(q_module, calib_images, num_images)
    else:
        print("FHE execution was not run. Use --run_fhe to enable it.")


if __name__ == "__main__":
    main()
