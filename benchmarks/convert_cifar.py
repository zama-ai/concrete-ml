"""Takes the csv dumped by infer_fhe and generates a proper json for the benchmark DB"""

import datetime
import json
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from convert import get_git_hash, get_git_hash_date, git_iso_to_python_iso, is_git_diff


def chaos_distance(seq_1, seq_2):
    index_mapping = {element: index for index, element in enumerate(seq_1)}
    return minimum_bribes(list(map(index_mapping.get, seq_2)))


def minimum_bribes(q):
    # Inspired from https://www.hackerrank.com/challenges/new-year-chaos/problem
    q = [i - 1 for i in q]  # set queue to start at 0
    bribes = 0
    for i, o in enumerate(q):
        for k in q[max(o - 1, 0) : i]:
            if k > o:
                bribes += 1
    return bribes


def main():
    # Get metrics
    results = pd.read_csv("./inference_results.csv")
    assert isinstance(results, pd.DataFrame)
    timing_columns = [col for col in results.columns if col.endswith("_time")]
    timings = results[timing_columns]
    assert isinstance(timings, pd.DataFrame)
    timing_means = timings.mean().to_dict()
    timing_stds = timings.std().to_dict()
    num_samples = len(results)
    torch_cols = list(
        sorted([col for col in results.columns if col.startswith("torch_prediction")])
    )
    fhe_cols = list(sorted([col for col in results.columns if col.startswith("prediction_")]))
    torch_preds, fhe_preds = results[torch_cols], results[fhe_cols]
    ground_truth = results["label"]
    assert isinstance(torch_preds, pd.DataFrame)
    assert isinstance(fhe_preds, pd.DataFrame)

    torch_classes, fhe_classes = np.argmax(torch_preds.values, axis=1), np.argmax(
        fhe_preds.values, axis=1
    )
    torch_order, fhe_order = np.argsort(torch_preds.values, axis=1), np.argsort(
        fhe_preds.values, axis=1
    )
    chaos_distance_mean = np.mean(
        [chaos_distance(elt_1, elt_2) for elt_1, elt_2 in zip(torch_order, fhe_order)]
    )
    top_1_acc = np.sum(torch_classes == fhe_classes) / num_samples
    top_1_acc_diff = (
        np.sum(torch_classes == ground_truth) - np.sum(fhe_classes == ground_truth)
    ) / num_samples

    # Repository specifics
    path_to_repository = (Path(__file__) / "..").resolve()
    assert not is_git_diff(path_to_repository)
    current_git_hash = get_git_hash(path_to_repository)
    current_git_hash_timestamp = datetime.datetime.fromisoformat(
        git_iso_to_python_iso(get_git_hash_date(current_git_hash, path_to_repository))
    ).timestamp()
    current_timestamp = datetime.datetime.now().timestamp()

    # Collect everything
    session_data: Dict[str, Union[Dict, List]] = {}

    # Create machine
    session_data["machine"] = {
        "machine_name": None,
        "machine_specs": {
            "cpu": None,
            "ram": None,
            "os": None,
        },
    }

    # Create experiments
    experiments = []
    model_name, dataset_name = "8-bit-split-v0", "CIFAR-10"
    experiment_representation: Dict[str, Any] = {}
    experiment_representation["experiment_name"] = "cifar-10-8-bit-split-v0"
    experiment_representation["experiment_metadata"] = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "cml_version": version("concrete-ml"),
        "cnp_version": version("concrete-numpy"),
    }
    experiment_representation["git_hash"] = current_git_hash
    experiment_representation["git_timestamp"] = current_git_hash_timestamp
    experiment_representation["experiment_timestamp"] = current_timestamp

    experiment_representation["metrics"] = []
    for key, value in timing_means.items():
        experiment_representation["metrics"].append({"metric_name": f"{key}_mean", "value": value})
    for key, value in timing_stds.items():
        experiment_representation["metrics"].append({"metric_name": f"{key}_std", "value": value})
    experiment_representation["metrics"].append(
        {"metric_name": "num_samples", "value": num_samples}
    )
    experiment_representation["metrics"].append({"metric_name": "top_1_acc", "value": top_1_acc})
    experiment_representation["metrics"].append(
        {"metric_name": "top_1_acc_diff", "value": top_1_acc_diff}
    )
    experiment_representation["metrics"].append(
        {"metric_name": "chaos_distance_mean", "value": chaos_distance_mean}
    )
    experiments.append(experiment_representation)
    session_data["experiments"] = experiments

    # Dump modified file
    with open("./to_upload.json", "w", encoding="utf-8") as file:
        json.dump(session_data, file, indent=4)

    return session_data


if __name__ == "__main__":
    main()
