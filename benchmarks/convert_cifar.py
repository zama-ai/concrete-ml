"""Takes a csv dumped and generates a proper json for the benchmark DB"""

import argparse
import datetime
import json
import platform
import socket
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Union

import cpuinfo
import numpy as np
import pandas as pd
import psutil
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


def get_size(bytes_count: float, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes_count < factor:
            return f"{bytes_count:.2f} {unit}{suffix}"
        bytes_count /= factor
    return f"{bytes_count:.2f} {suffix}"


def get_system_information():
    # From https://stackoverflow.com/questions/3103178/how-to-get-the-system-info-with-python
    info = {}
    # What is naturally dumped by python-progress-tracker
    info["ram"] = get_size(psutil.virtual_memory().total)
    info["cpu"] = cpuinfo.get_cpu_info()["brand_raw"]
    info["os"] = f"{platform.system()} {platform.release()}"

    # Added metadata about the system
    info["platform"] = platform.system()
    info["platform-release"] = platform.release()
    info["platform-version"] = platform.version()
    info["architecture"] = platform.machine()
    info["hostname"] = socket.gethostname()
    info["processor"] = platform.processor()
    info["physical_cores"] = psutil.cpu_count(logical=False)
    info["total_cores"] = psutil.cpu_count(logical=True)
    uname = platform.uname()
    info["machine"] = uname.machine
    info["processor"] = uname.processor
    info["system"] = uname.system
    info["node_name"] = uname.node
    info["release"] = uname.release
    info["version"] = uname.version
    info["swap"] = get_size(psutil.swap_memory().total)

    return info


def get_ec2_metadata():
    res = {}
    try:
        output = subprocess.check_output("ec2metadata", shell=True, encoding="utf-8")
        for line in output.split("\n"):
            if line:
                splitted = line.split(": ")
                if len(splitted) == 2:
                    key, value = splitted
                    res[key] = value
            else:
                print(line)
        return res
    except subprocess.CalledProcessError as exception:
        print(exception)
        return res


def value_else_none(value):
    if value != value:  # pylint: disable=comparison-with-itself
        return None
    return value


def main(model_name):
    # Get metrics
    results = pd.read_csv("./inference_results.csv")
    with open("./metadata.json", "r", encoding="utf-8") as file:
        metadata = json.load(file)
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

    ec2_metadata = get_ec2_metadata()

    # Create machine
    # We should probably add the platform to the DB too
    session_data["machine"] = {
        "machine_name": ec2_metadata.get("instance-type", socket.gethostname()),
        "machine_specs": get_system_information(),
    }

    # Create experiments
    experiments = []
    dataset_name = "CIFAR-10"
    experiment_data: Dict[str, Any] = {}
    experiment_data["experiment_name"] = f"cifar-10-{model_name}"
    experiment_data["experiment_metadata"] = metadata
    experiment_data["experiment_metadata"].update(
        {
            "model_name": model_name,
            "dataset_name": dataset_name,
        }
    )
    experiment_data["git_hash"] = current_git_hash
    experiment_data["git_timestamp"] = current_git_hash_timestamp
    experiment_data["experiment_timestamp"] = current_timestamp

    experiment_data["metrics"] = []
    for key, value in timing_means.items():
        experiment_data["metrics"].append(
            {"metric_name": f"{key}_mean", "value": value_else_none(value)}
        )
    for key, value in timing_stds.items():
        experiment_data["metrics"].append(
            {"metric_name": f"{key}_std", "value": value_else_none(value)}
        )
    experiment_data["metrics"].append(
        {"metric_name": "num_samples", "value": value_else_none(num_samples)}
    )
    experiment_data["metrics"].append(
        {"metric_name": "top_1_acc", "value": value_else_none(top_1_acc)}
    )
    experiment_data["metrics"].append(
        {"metric_name": "top_1_acc_diff", "value": value_else_none(top_1_acc_diff)}
    )
    experiment_data["metrics"].append(
        {
            "metric_name": "chaos_distance_mean",
            "value": value_else_none(chaos_distance_mean),
        }
    )
    experiments.append(experiment_data)
    session_data["experiments"] = experiments

    # Dump modified file
    with open("./to_upload.json", "w", encoding="utf-8") as file:
        json.dump(session_data, file, indent=4)

    return session_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FHE Inference on CIFAR10")
    parser.add_argument("--model-name", help="Model name for the experiment.")
    args = parser.parse_args()
    main(args.model_name)
