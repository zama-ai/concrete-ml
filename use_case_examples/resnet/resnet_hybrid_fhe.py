"""
ResNet18 Hybrid FHE implementation using HybridFHEModel.

This script demonstrates how to run ResNet18 with specific layers executing in FHE mode
while others remain in clear text. All linear and conv layers are executed in FHE.
"""

import argparse
import json
import platform
import socket
import time
from datetime import datetime
from pathlib import Path

import cpuinfo
import psutil
import torch
from resnet import ResNet18_Weights, resnet18_custom
from utils_resnet import ImageNetProcessor

from concrete.ml.torch.hybrid_model import HybridFHEModel

BASE_DIR = Path(__file__).resolve().parent
CALIBRATION_SAMPLES = 10
NUM_TEST_SAMPLES = 20


def get_system_info():
    """Gather system information for metrics."""
    cpu_info_data = cpuinfo.get_cpu_info()

    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cpu": {
            "name": cpu_info_data.get("brand_raw", "Unknown"),
            "count": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "memory_gb": psutil.virtual_memory().total // (1024**3),
        },
        "device": "cpu",
    }


def load_model():
    """Load the ResNet18 model with pre-trained weights."""
    return resnet18_custom(weights=ResNet18_Weights.IMAGENET1K_V1)


def load_data(dataset_cache_dir=None):
    """Load the ImageNet data processor and calibration data."""
    print(f"Creating ImageNetProcessor with cache_dir: {dataset_cache_dir}")
    processor = ImageNetProcessor(
        NUM_TEST_SAMPLES, CALIBRATION_SAMPLES, cache_dir=dataset_cache_dir
    )

    calib_images = processor.get_calibration_tensor()

    if calib_images is None:
        raise ValueError("get_calibration_tensor() returned None")
    if len(calib_images) == 0:
        raise ValueError(
            f"get_calibration_tensor() returned empty tensor with shape {calib_images.shape}"
        )

    print(f"Loaded calibration data: shape={calib_images.shape}, dtype={calib_images.dtype}")
    return processor, calib_images


def create_hybrid_model(model, target_modules):
    """
    Create a hybrid FHE model with specified modules running in FHE.

    Args:
        model: PyTorch model
        target_modules: List of module names to execute in FHE

    Returns:
        HybridFHEModel instance
    """
    print("\nCreating HybridFHEModel...")
    hybrid_model = HybridFHEModel(
        model=model,
        module_names=target_modules,
        model_name="resnet18_hybrid",
    )
    return hybrid_model


def compile_hybrid_model(hybrid_model, calibration_data):
    """
    Compile the hybrid model for FHE execution.

    Args:
        hybrid_model: HybridFHEModel instance
        calibration_data: Calibration tensor data

    Returns:
        Compilation time in seconds
    """
    print("\nCompiling hybrid model for FHE...")
    start_time = time.time()

    hybrid_model.compile_model(x=calibration_data, n_bits=8, device="cpu")  # 8-bit quantization

    compile_time = time.time() - start_time
    print(f"\nCompilation completed in {compile_time:.2f} seconds")
    return compile_time


def run_single_fhe_inference(hybrid_model, test_data, num_samples=1):
    """
    Run inference in FHE mode and measure timing.

    Args:
        hybrid_model: Compiled HybridFHEModel
        test_data: Test tensor data
        num_samples: Number of samples to process

    Returns:
        Dictionary with results and timing information
    """
    print(f"\nRunning inference on {num_samples} sample(s)...")

    # Validate inputs
    if test_data is None or len(test_data) == 0:
        print("ERROR: Test data is None or empty!")
        return {}

    if num_samples > len(test_data):
        print(f"WARNING: Requested {num_samples} samples but only {len(test_data)} available")
        num_samples = len(test_data)

    # Use specified number of samples
    test_subset = test_data[:num_samples]
    print(f"Test subset shape: {test_subset.shape}")

    def get_top5_labels(output):
        """Extract top 5 predicted labels."""
        return torch.topk(output, 5, dim=1)[1].cpu().numpy()

    results = {}

    # Test clear mode first
    print(f"\n--- Running in CLEAR mode ---")
    try:
        total_clear_time = 0
        clear_predictions = []

        for i in range(num_samples):
            sample = test_subset[i : i + 1]
            start_time = time.time()
            with torch.no_grad():
                output_clear = hybrid_model(sample, fhe="disable")
            clear_time = time.time() - start_time
            total_clear_time += clear_time

            top5_clear = get_top5_labels(output_clear)
            clear_predictions.append(top5_clear[0])

            print(f"  Sample {i+1}: {clear_time:.4f}s, Top-5: {top5_clear[0]}")

        avg_clear_time = total_clear_time / num_samples
        results["clear"] = {
            "total_time": total_clear_time,
            "avg_time": avg_clear_time,
            "predictions": clear_predictions,
        }

        print(f"  Total clear time: {total_clear_time:.4f}s")
        print(f"  Average clear time: {avg_clear_time:.4f}s")

    except Exception as e:
        print(f"  Error in clear mode: {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()

    # Test FHE execute mode
    print(f"\n--- Running in FHE EXECUTE mode ---")
    print("\n⚠️  IMPORTANT: The first FHE inference run will include key generation time")
    print("   Subsequent runs will be faster as they reuse the generated keys.")
    print("   This is normal behavior and only affects the first run.")

    try:
        total_fhe_time = 0
        fhe_predictions = []

        for i in range(num_samples):
            sample = test_subset[i : i + 1]
            print(f"  Starting FHE execution for sample {i+1}/{num_samples}...")
            start_time = time.time()
            with torch.no_grad():
                output_fhe = hybrid_model(sample, fhe="execute")
            fhe_time = time.time() - start_time
            total_fhe_time += fhe_time

            top5_fhe = get_top5_labels(output_fhe)
            fhe_predictions.append(top5_fhe[0])

            print(f"  ✅ Sample {i+1} completed: {fhe_time:.4f}s, Top-5: {top5_fhe[0]}")
            print(f"     Elapsed total time: {total_fhe_time:.4f}s")

        avg_fhe_time = total_fhe_time / num_samples
        results["fhe"] = {
            "total_time": total_fhe_time,
            "avg_time": avg_fhe_time,
            "predictions": fhe_predictions,
        }

        print(f"  Total FHE time: {total_fhe_time:.4f}s")
        print(f"  Average FHE time: {avg_fhe_time:.4f}s")

    except Exception as e:
        print(f"  Error in FHE execute mode: {type(e).__name__}: {str(e)}")
        import traceback

        traceback.print_exc()

    # Compare results
    print(f"\n--- Results Summary ---")
    if "clear" in results and "fhe" in results:
        clear_time = results["clear"]["avg_time"]
        fhe_time = results["fhe"]["avg_time"]
        overhead = fhe_time / clear_time if clear_time > 0 else float("inf")

        print(f"Clear execution time: {clear_time:.4f}s per sample")
        print(f"FHE execution time: {fhe_time:.4f}s per sample")
        print(f"FHE overhead: {overhead:.1f}x")

        # Check prediction consistency
        correct_predictions = sum(
            1
            for i in range(num_samples)
            if results["clear"]["predictions"][i][0] == results["fhe"]["predictions"][i][0]
        )
        consistency = correct_predictions / num_samples * 100

        print(
            f"Top-1 prediction consistency: {consistency:.1f}% ({correct_predictions}/{num_samples})"
        )

    elif "fhe" in results:
        print(f"FHE execution time: {results['fhe']['avg_time']:.4f}s per sample")
    else:
        print("No successful inference results")

    return results


def evaluate_model_accuracy(hybrid_model, processor, mode="disable"):
    """
    Evaluate model accuracy on the full test set.

    Args:
        hybrid_model: HybridFHEModel instance
        processor: ImageNetProcessor
        mode: FHE mode for evaluation

    Returns:
        Tuple of (top1_accuracy, top5_accuracy)
    """
    print(f"\nEvaluating model accuracy in {mode} mode...")

    total_correct = 0
    total_top5_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in processor.dataloader:
            batch_images = batch["pixel_values"]
            batch_labels = batch["labels"]

            # Run inference
            outputs = hybrid_model(batch_images, fhe=mode)

            # Calculate top-1 and top-5 accuracy
            _, top1_pred = torch.max(outputs, 1)
            _, top5_pred = torch.topk(outputs, 5, dim=1)

            total_correct += (top1_pred == batch_labels).sum().item()

            # Check if true label is in top-5
            for i, label in enumerate(batch_labels):
                if label in top5_pred[i]:
                    total_top5_correct += 1

            total_samples += batch_labels.size(0)

    top1_accuracy = (total_correct / total_samples) * 100
    top5_accuracy = (total_top5_correct / total_samples) * 100

    print(f"Top-1 Accuracy ({mode}): {top1_accuracy:.2f}%")
    print(f"Top-5 Accuracy ({mode}): {top5_accuracy:.2f}%")

    return top1_accuracy, top5_accuracy


def save_metrics(
    results, system_info, compilation_time, target_modules, args, accuracy_results=None
):
    """Save benchmark metrics in the expected format for the database."""

    # Get git information (placeholder for now)
    git_info = {"hash": "unknown_resnet_hash", "timestamp": datetime.utcnow().timestamp()}

    # 1. Prepare the 'machine' part
    machine_data = {
        "machine_name": socket.gethostname(),
        "machine_specs": system_info,
    }

    # 2. Prepare the 'experiment_metadata' part
    experiment_metadata = {
        "num_samples": args.num_samples,
        "target_all_layers": args.target_all_layers,
        "evaluate_accuracy": args.evaluate_accuracy,
        "save_model": args.save_model,
        "num_fhe_layers": len(target_modules),
        "fhe_layers_targeted": target_modules,
        "debug_mode": args.debug_mode,
        "device_used": system_info.get("device", "cpu"),
        "framework": "concrete-ml",
        "model": "resnet18",
        "task": "image_classification",
        "dataset": "imagenet",
        "quantization_bits": 8,
        "run_type": "benchmark",
    }

    # 3. Prepare the 'metrics' list
    metrics_list = []

    # Add compilation time
    metrics_list.append({"metric_name": "compilation_time_seconds", "value": compilation_time})

    # Add timing results
    if "clear" in results:
        metrics_list.append(
            {"metric_name": "clear_total_time_seconds", "value": results["clear"]["total_time"]}
        )
        metrics_list.append(
            {
                "metric_name": "clear_avg_time_per_sample_seconds",
                "value": results["clear"]["avg_time"],
            }
        )

    if "fhe" in results:
        metrics_list.append(
            {"metric_name": "fhe_total_time_seconds", "value": results["fhe"]["total_time"]}
        )
        metrics_list.append(
            {"metric_name": "fhe_avg_time_per_sample_seconds", "value": results["fhe"]["avg_time"]}
        )

        # Calculate overhead if both clear and FHE results exist
        if "clear" in results and results["clear"]["avg_time"] > 0:
            overhead = results["fhe"]["avg_time"] / results["clear"]["avg_time"]
            metrics_list.append({"metric_name": "fhe_overhead_factor", "value": overhead})

            # Calculate prediction consistency
            if "predictions" in results["clear"] and "predictions" in results["fhe"]:
                correct = sum(
                    1
                    for i in range(args.num_samples)
                    if results["clear"]["predictions"][i][0] == results["fhe"]["predictions"][i][0]
                )
                consistency = (correct / args.num_samples) * 100
                metrics_list.append(
                    {"metric_name": "top1_prediction_consistency_percent", "value": consistency}
                )

        # Add performance per FHE layer
        if len(target_modules) > 0:
            metrics_list.append(
                {
                    "metric_name": "avg_time_per_fhe_layer_seconds",
                    "value": results["fhe"]["avg_time"] / len(target_modules),
                }
            )

    # Add accuracy results if available
    if accuracy_results:
        for mode, acc_data in accuracy_results.items():
            if "top1" in acc_data:
                metrics_list.append(
                    {"metric_name": f"accuracy_{mode}_top1_percent", "value": acc_data["top1"]}
                )
            if "top5" in acc_data:
                metrics_list.append(
                    {"metric_name": f"accuracy_{mode}_top5_percent", "value": acc_data["top5"]}
                )

    # 4. Build the complete structure
    session_data = {
        "machine": machine_data,
        "experiments": [
            {
                "experiment_name": "resnet18_hybrid_fhe_benchmark",
                "experiment_metadata": experiment_metadata,
                "git_hash": git_info["hash"],
                "git_timestamp": git_info["timestamp"],
                "experiment_timestamp": datetime.utcnow().timestamp(),
                "metrics": metrics_list,
            }
        ],
    }

    # Save to file
    output_file = BASE_DIR / "to_upload.json"
    with open(output_file, "w") as f:
        json.dump(session_data, f, indent=2)

    print(f"\nMetrics saved to {output_file}")
    return session_data


def main():
    parser = argparse.ArgumentParser(description="Run ResNet18 with hybrid FHE execution")
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default=None,
        help="Path to the directory where the dataset is cached",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples for inference timing comparison (default: 1)",
    )
    parser.add_argument(
        "--evaluate_accuracy", action="store_true", help="Evaluate model accuracy on full test set"
    )
    parser.add_argument("--save_model", action="store_true", help="Save the compiled hybrid model")
    parser.add_argument(
        "--target_all_layers",
        action="store_true",
        help="Target all conv and linear layers instead of just linear layers",
    )
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Only select the first non-conv1 layer for FHE execution",
    )

    args = parser.parse_args()

    print("=== ResNet18 Hybrid FHE Execution ===")
    print(
        "\n⚠️  NOTE: The first convolutional layer ('conv1') is currently excluded from FHE execution"
    )

    # Get system information
    system_info = get_system_info()
    print(f"\nRunning on CPU")
    print(f"CPU: {system_info['cpu']['name']}")
    print(
        f"CPU Cores: {system_info['cpu']['count']} physical, {system_info['cpu']['threads']} logical"
    )
    print(f"Memory: {system_info['cpu']['memory_gb']} GB")

    # Load model and data
    print("\nLoading model and data...")
    resnet18 = load_model()
    processor, calib_images = load_data(args.dataset_cache_dir)

    print(f"\nDebug: Calibration tensor shape: {calib_images.shape}")
    print(f"Debug: Calibration tensor dtype: {calib_images.dtype}")
    print(f"Debug: Calibration tensor device: {calib_images.device}")

    # Select all Conv2d and Linear layers except 'conv1'
    print("\nSelecting layers for FHE execution...")
    target_modules = []
    for name, module in resnet18.named_modules():
        if name != "conv1" and (
            isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear)
        ):
            target_modules.append(name)
            print(f"  ✓ Adding layer: {name} ({type(module).__name__})")
            if args.debug_mode:
                break

    print(f"\nSelected {len(target_modules)} target modules for FHE execution:")
    for i, module_name in enumerate(target_modules, 1):
        print(f"  {i}. {module_name}")

    # Create hybrid model
    hybrid_model = create_hybrid_model(resnet18, target_modules)

    # Compile for FHE
    compilation_time = compile_hybrid_model(hybrid_model, calib_images)

    # Run FHE inference BEFORE saving the model
    test_data = processor.get_calibration_tensor()
    print(f"\nDebug: Test data shape: {test_data.shape}")
    print(f"Debug: Test data dtype: {test_data.dtype}")
    print(f"Debug: Test data min/max: {test_data.min():.3f}/{test_data.max():.3f}")

    results = run_single_fhe_inference(hybrid_model, test_data, args.num_samples)

    # Evaluate accuracy if requested
    accuracy_results = {}
    if args.evaluate_accuracy:
        for mode in ["disable", "simulate"]:
            top1, top5 = evaluate_model_accuracy(hybrid_model, processor, mode)
            accuracy_results[mode] = {"top1": top1, "top5": top5}

    # Save model AFTER inference is complete
    if args.save_model:
        save_path = BASE_DIR / "compiled_hybrid_model"
        save_path.mkdir(exist_ok=True, parents=True)
        print(f"\nSaving compiled model to {save_path}...")
        try:
            hybrid_model.save_and_clear_private_info(save_path, via_mlir=True)
            print("Model saved successfully!")
            # Verify the save
            if save_path.exists() and any(save_path.iterdir()):
                print(f"Verified: Model directory contains {len(list(save_path.iterdir()))} files")
            else:
                print("WARNING: Model directory appears to be empty after save")
        except Exception as e:
            print(f"Error saving model: {type(e).__name__}: {str(e)}")
            import traceback

            traceback.print_exc()

    # Save metrics for the benchmark database
    metrics = save_metrics(
        results, system_info, compilation_time, target_modules, args, accuracy_results
    )

    # Print comprehensive final statistics
    print("\n" + "=" * 60)
    print("                FINAL EXECUTION STATISTICS")
    print("=" * 60)

    # FHE Configuration Summary
    print(f"FHE Configuration:")
    print(f"  • Number of layers executed in FHE: {len(target_modules)}")
    print(f"  • FHE layer types: ALL Linear + Conv2d layers")
    print(f"  • Number of samples processed: {args.num_samples}")

    # Timing Results
    if "fhe" in results and "clear" in results:
        fhe_total = results["fhe"]["total_time"]
        fhe_avg = results["fhe"]["avg_time"]
        clear_total = results["clear"]["total_time"]
        clear_avg = results["clear"]["avg_time"]
        overhead = fhe_avg / clear_avg if clear_avg > 0 else float("inf")

        print(f"\nTiming Results:")
        print(f"  • Total FHE execution time: {fhe_total:.4f}s")
        print(f"  • Average FHE time per sample: {fhe_avg:.4f}s")
        print(f"  • Total clear execution time: {clear_total:.4f}s")
        print(f"  • Average clear time per sample: {clear_avg:.4f}s")
        print(f"  • FHE overhead factor: {overhead:.1f}x")

        # Performance per FHE layer
        time_per_fhe_layer = fhe_avg / len(target_modules) if len(target_modules) > 0 else 0
        print(f"  • Average time per FHE layer: {time_per_fhe_layer:.4f}s")

    elif "fhe" in results:
        fhe_total = results["fhe"]["total_time"]
        fhe_avg = results["fhe"]["avg_time"]
        print(f"\nTiming Results:")
        print(f"  • **TOTAL FHE EXECUTION TIME: {fhe_total:.4f}s**")
        print(f"  • Average FHE time per sample: {fhe_avg:.4f}s")

        # Performance per FHE layer
        time_per_fhe_layer = fhe_avg / len(target_modules) if len(target_modules) > 0 else 0
        print(f"  • Average time per FHE layer: {time_per_fhe_layer:.4f}s")
    else:
        print(f"\nTiming Results:")
        print(f"  • ❌ No FHE execution results available")

    # Accuracy Results
    if "fhe" in results and "clear" in results:
        correct_predictions = sum(
            1
            for i in range(args.num_samples)
            if results["clear"]["predictions"][i][0] == results["fhe"]["predictions"][i][0]
        )
        consistency = correct_predictions / args.num_samples * 100

        print(f"\nAccuracy Results:")
        print(f"  • Top-1 prediction consistency: {consistency:.1f}%")
        print(f"  • Correct predictions: {correct_predictions}/{args.num_samples}")
        if consistency == 100:
            print(f"  • ✅ Perfect accuracy maintained in FHE mode")
        else:
            print(f"  • ⚠️  Some accuracy loss in FHE mode")

    print("\n" + "=" * 60)

    # Exit with error if no results
    if not results:
        print("\nERROR: No inference results obtained. Exiting with error code.")
        exit(1)


if __name__ == "__main__":
    main()
