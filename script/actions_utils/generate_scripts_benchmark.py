"""Script to generate the list of commands to run all benchmarks"""
import argparse
import datetime
import json
import subprocess

MAX_VALUE = 100
MIN_VALUE = 0


def int_range(x: str) -> int:
    """Cast string to int and raises an error if not in range."""
    x_int = int(x)
    if x_int > MAX_VALUE:
        raise ValueError(f"{x_int} > {MAX_VALUE}")
    if x_int < MIN_VALUE:
        raise ValueError(f"{x_int} < {MIN_VALUE}")
    return x_int


def main():
    """Generate the list of commands to be launched.

    One command == one benchmark (one model + one set of hyper-parameters + one dataset)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--list_length", dest="list_length", choices=["short", "long"])
    parser.add_argument("--classification", dest="classification", choices=["true", "false"])
    parser.add_argument("--regression", dest="regression", choices=["true", "false"])
    parser.add_argument("--glm", dest="glm", choices=["true", "false"])
    parser.add_argument("--fhe_samples", dest="fhe_samples", type=int_range, default=100)

    args = parser.parse_args()

    list_length = args.list_length
    if list_length == "short":
        list_arg = "--short_list"
    elif list_length == "long":
        list_arg = "--long_list"
    else:
        raise ValueError(f"Unrecognized list length: {list_length}")

    scripts = []
    if args.classification == "true":
        scripts.append("classification")
    if args.regression == "true":
        scripts.append("regression")
    if args.glm == "true":
        scripts.append("glm")

    # Get all commands to benchmark each models using python benchmarks/BENCHMARK_FILE
    commands = []
    for script in scripts:
        command_start = f"python3 benchmarks/{script}.py"
        now = datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S")
        script_commands = [
            elt.replace('"', '\\"')  # Needed to escape " when calling eval
            for elt in subprocess.check_output(
                f"{command_start} {list_arg} --fhe_samples {args.fhe_samples}",
                shell=True,
            )
            .decode("utf-8")
            .split("\n")
            if elt.startswith("--")  # avoid getting trash logs in there
        ]
        commands += [f"{command_start} {elt}" for elt in script_commands]

    result = []
    for index, command in enumerate(commands):
        element = {"label": f"ml_bench_{now}_{index}", "index": index, "command": command}
        result.append(element)

    # Print to stdout to put in environment variable
    print(json.dumps(result))


if __name__ == "__main__":
    main()
