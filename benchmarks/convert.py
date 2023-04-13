#!/usr/bin/env python
# coding: utf-8
"""Used to convert output format from python-progress-tracker to new postgres DB format"""


import argparse
import datetime
import json
import math
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from common import benchmark_name_to_config


def is_git_diff(path: Union[None, Path, str]) -> bool:
    """Check if there is a diff in a repository."""
    path = path if path is not None else "."
    completed_process = subprocess.run(
        ["git", "diff", "HEAD"], capture_output=True, cwd=path, check=True
    )
    if completed_process.stderr:
        raise ValueError("Check git diff raised an error:\n" f"{completed_process.stderr.decode()}")
    return bool(completed_process.stdout)


def get_git_hash(path: Union[None, Path, str]) -> str:
    """Get git hash of repository."""
    path = path if path is not None else "."
    completed_process = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, cwd=path, check=True
    )
    if completed_process.stderr:
        raise ValueError("Check git hash raised an error:\n" f"{completed_process.stderr.decode()}")
    return completed_process.stdout.decode().strip()


def get_git_hash_date(hash_str: str, path: Union[None, Path, str]) -> str:
    """Get repository git hash date."""
    path = path if path is not None else "."
    # We get the author date (%ai) and not the commit date (%ci)
    # for more details please refer to https://git-scm.com/docs/git-show
    completed_process = subprocess.run(
        ["git", "show", "-s", "--date=iso-strict", "--format=%ai", hash_str],
        capture_output=True,
        cwd=path,
        check=True,
    )
    if completed_process.stderr:
        raise ValueError("Check git hash raised an error:\n" f"{completed_process.stderr.decode()}")
    print(completed_process.stdout.decode().strip())
    return completed_process.stdout.decode().strip()


def git_iso_to_python_iso(date_str: str) -> str:
    """Transform git iso into Python iso."""
    splitted = date_str.split()
    return f"{splitted[0]}T{splitted[1]}{splitted[2][:3]}:{splitted[2][3:]}"


def find_element_in_zip(elements: List[Tuple[str, Any]], key: str) -> Any:
    """Find the element in a dict represented as a zip."""
    for key_, value in elements:
        if key_ == key:
            return value
    raise ValueError(f"Couldn't find key {key} in {[key for key, _ in elements]}")


def convert_to_new_postgres(
    source: Path, target: Path, path_to_repository: Path, machine_name: Optional[str] = None
):
    """Convert json file generated via python-progress-tracker to new format."""
    # Load from direct result of script
    assert source.exists(), source
    with open(source, "r", encoding="utf-8") as file:
        progress = json.load(file)

    # Get git information
    assert not is_git_diff(path_to_repository)
    current_git_hash = get_git_hash(path_to_repository)
    current_git_hash_timestamp = datetime.datetime.fromisoformat(
        git_iso_to_python_iso(get_git_hash_date(current_git_hash, path_to_repository))
    ).timestamp()
    current_timestamp = datetime.datetime.now().timestamp()

    session_data: Dict[str, Union[Dict, List]] = {}

    # Create machine
    session_data["machine"] = {
        "machine_name": progress["machine"]["name"] if machine_name is None else machine_name,
        "machine_specs": {
            "cpu": find_element_in_zip(progress["machine"]["specs"], "CPU"),
            "ram": find_element_in_zip(progress["machine"]["specs"], "RAM"),
            "os": find_element_in_zip(progress["machine"]["specs"], "OS"),
        },
    }

    # Create experiments
    experiments = []
    for experiment_name, experiment_values in progress["targets"].items():
        model_name, dataset_name, hyper_args = benchmark_name_to_config(experiment_name)
        experiment_representation = {}
        experiment_representation["experiment_name"] = experiment_name
        experiment_representation["experiment_metadata"] = {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "cml_version": version("concrete-ml"),
            "cnp_version": version("concrete-python"),
        }
        experiment_representation["experiment_metadata"].update(hyper_args)
        experiment_representation["git_hash"] = current_git_hash
        experiment_representation["git_timestamp"] = current_git_hash_timestamp
        experiment_representation["experiment_timestamp"] = current_timestamp
        experiment_representation["metrics"] = []
        if "measurements" in experiment_values:
            for metric_name, metric_value in experiment_values["measurements"].items():
                if math.isnan(metric_value):  # Nan
                    metric_value = None
                experiment_representation["metrics"].append(
                    {"metric_name": metric_name, "value": metric_value}
                )
        experiment_representation["metrics"].append(
            {"metric_name": "working", "value": int(experiment_values["working"])}
        )
        experiments.append(experiment_representation)
    session_data["experiments"] = experiments

    # Dump modified file
    with open(target, "w", encoding="utf-8") as file:
        json.dump(session_data, file, indent=4)


def main():
    """Main function to convert json into new format."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        dest="source",
        type=Path,
        default=Path("./source.json"),
        help="Path to json file to convert.",
    )
    parser.add_argument(
        "--target",
        dest="target",
        type=Path,
        default=Path("./target.json"),
        help="Path to converted json file.",
    )
    parser.add_argument(
        "--path_to_repository",
        dest="path_to_repository",
        type=Path,
        default=Path("./"),
        help="Path to repository used to run the benchmark",
    )
    parser.add_argument(
        "--machine_name",
        dest="machine_name",
        type=str,
        default=None,
        help="Overwrite machine_name (default is None)",
    )
    args = parser.parse_args(sys.argv[1:])
    convert_to_new_postgres(
        source=args.source,
        target=args.target,
        path_to_repository=args.path_to_repository,
        machine_name=args.machine_name,
    )


if __name__ == "__main__":
    main()
