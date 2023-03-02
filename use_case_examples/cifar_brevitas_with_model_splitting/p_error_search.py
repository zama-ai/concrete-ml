#!/usr/bin/env python
# coding: utf-8

# Run p_error search for the model.

# Steps
# - Load model
# - Load data
# - Pre-process data according to the model
#   - In our case that means computing the features maps
# - Run the search (for a given number of steps here but we could also consider something like the delta in p-error between 2 steps as a stop condition)
#   - Run the inference using the Virtual Library and p_error=0.0 for reference
#   - Take a lower and upper bound of p-error (start = [0, 1])
#   - Consider a p-error between the bounds
#   - Run the inference using the Virtual Library and this p_error
#   - If our objective is matched, here we compute the difference in accuracy between the reference and the result of the inference
#       - Update the lower bound to be the current p-error
#   - Else
#       - Update the upper bound to be the current p-error
#   - Update the current p-error with the mean of the bounds

import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from brevitas.nn.quant_layer import acc
from concrete.numpy import Configuration
from model import CNV
from sklearn.metrics import accuracy_score, top_k_accuracy_score
from tqdm import tqdm

from concrete.ml.quantization import QuantizedModule
from concrete.ml.torch.compile import compile_brevitas_qat_model


def acc_diff_objective(
    model_output, reference, ground_truth, tolerance=0.01, k=1
) -> Tuple[bool, Dict]:
    # Compute accuracy
    reference_accuracy = top_k_accuracy_score(y_true=ground_truth, y_score=reference, k=k)
    estimated_accuracy = top_k_accuracy_score(y_true=ground_truth, y_score=model_output, k=k)

    # Compute inference error
    difference = abs(reference_accuracy - estimated_accuracy)
    raw_difference = reference - model_output
    l_1_error = np.abs(raw_difference).sum()
    l_inf_error = np.abs(raw_difference).max()
    count_error = (reference != model_output).sum()

    # Check if match
    if difference <= tolerance:
        match = True
    else:
        match = False

    meta_output = OrderedDict(
        {
            "accuracy_difference": difference,
            "match": match,
            "l1_error": l_1_error,
            "linf_error": l_inf_error,
            "count_error": count_error,
        }
    )
    return match, meta_output


def compile(model, inputs, p_error: float) -> QuantizedModule:
    configuration = Configuration(
        dump_artifacts_on_unexpected_failures=True,
        enable_unsafe_features=True,  # This is for our tests in Virtual Library only
        show_graph=False,
        show_mlir=False,
        show_optimizer=False,
    )
    quantized_numpy_module = compile_brevitas_qat_model(
        torch_model=model,
        torch_inputset=inputs,
        p_error=p_error,
        use_virtual_lib=True,
        configuration=configuration,
    )
    return quantized_numpy_module


def infer(quantized_module, inputs, progress_bar=True) -> Tuple[np.ndarray, Dict]:
    # We need to iterate to do the inference
    inferences = list()
    quantized_inferences = list()
    for img_feature_map in tqdm(inputs, leave=False, disable=not progress_bar):
        quantized_feature_maps = quantized_module.quantize_input(img_feature_map)
        quantized_output = quantized_module.fhe_circuit.simulate(quantized_feature_maps[None, ...])
        dequantized_output = quantized_module.dequantize_output(quantized_output)
        inferences.append(dequantized_output[0])
        quantized_inferences.append(quantized_output[0])
    inferences = np.array(inferences)
    quantized_inferences = np.array(quantized_inferences)
    return inferences, {"outputs": inferences, "quantized_output": quantized_inferences}


def output_to_csv(output: Dict, file_path: Path):
    for label_index in range(output["outputs"].shape[1]):
        output[f"pred_{label_index}"] = output["outputs"][:, label_index]
    for label_index in range(output["quantized_output"].shape[1]):
        output[f"pred_quant_{label_index}"] = output["quantized_output"][:, label_index]

    del output["outputs"]
    del output["quantized_output"]

    pd.DataFrame(output).to_csv(file_path)


def search(
    model: torch.nn.Module,
    inputs: np.ndarray,
    ground_truth: np.ndarray,
    objective: Callable[..., Tuple[bool, Dict]],
    max_iter: int = 100,
    bootstrap_samples: int = 3,
):
    search_folder = Path("./search")
    search_folder.mkdir(exist_ok=True)
    assert bootstrap_samples >= 1

    # Run predictions
    reference_quantized_module = compile(model=model, inputs=inputs, p_error=0.0)

    # Compute reference
    reference_output, meta_output = infer(reference_quantized_module, inputs)
    meta_output["label"] = ground_truth
    output_to_csv(meta_output, search_folder / "reference_output.csv")

    # Search
    lower = 0.0  # 0
    upper = 1.0  # 1
    current_p_error = (lower + upper) / 2

    log_file = search_folder / "p_error_search.csv"

    # Binary search algorithm
    for index in tqdm(range(max_iter)):
        # Run the inference with given p-error
        # Run predictions
        current_quantized_module = compile(model=model, inputs=inputs, p_error=current_p_error)

        all_matches = []
        # Compute reference
        for _ in range(bootstrap_samples):
            current_output, current_meta_output = infer(current_quantized_module, inputs)

            p_error_as_str = str(current_p_error).replace(".", "_")
            output_to_csv(current_meta_output, search_folder / "{p_error_as_str}_output.csv")

            is_matched_objective, objective_meta = objective(
                model_output=current_output,
                reference=reference_output,
                ground_truth=ground_truth,
            )
            all_matches.append(is_matched_objective)

            # CSV header
            if index == 0:
                with open(log_file, "w", encoding="utf-8") as file:
                    file.write(",".join(objective_meta.keys()) + ",lower,upper,p_error\n")
            with open(log_file, "a", encoding="utf-8") as file:
                file.write(
                    ",".join(str(elt) for elt in objective_meta.values())
                    + f",{str(lower)},{str(upper)},{str(current_p_error)}\n"
                )

        # Update p-error
        if all(all_matches):
            lower = current_p_error
        else:
            upper = current_p_error
        current_p_error = (lower + upper) / 2

        print(current_p_error, lower, upper)


def main():
    # Load model
    model = CNV(num_classes=10, weight_bit_width=2, act_bit_width=2, in_bit_width=3, in_ch=3)
    loaded = torch.load(Path(__file__).parent / "8_bit_model.pt")
    model.load_state_dict(loaded["model_state_dict"])
    model = model.eval()

    # Get dataset
    IMAGE_TRANSFORM = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    try:
        train_set = torchvision.datasets.CIFAR10(
            root=".data/",
            train=True,
            download=False,
            transform=IMAGE_TRANSFORM,
            target_transform=None,
        )
    except:
        train_set = torchvision.datasets.CIFAR10(
            root=".data/",
            train=True,
            download=True,
            transform=IMAGE_TRANSFORM,
            target_transform=None,
        )

    # Sub-sample
    num_samples = 1000
    train_sub_set = torch.stack(
        [train_set[index][0] for index in range(min(num_samples, len(train_set)))]
    )
    # Pre-processing -> images -> feature maps (used for compilation)
    with torch.no_grad():
        train_features_sub_set = model.clear_module(train_sub_set).numpy()
    train_sub_set_labels = np.array(
        [train_set[index][1] for index in range(min(num_samples, len(train_set)))]
    )

    search(
        model=model.encrypted_module,
        inputs=train_features_sub_set,
        ground_truth=train_sub_set_labels,
        objective=acc_diff_objective,
        max_iter=100,
        bootstrap_samples=1,
    )


if __name__ == "__main__":
    main()
