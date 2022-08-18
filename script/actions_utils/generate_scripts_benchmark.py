"""Script to generate the list of commands to run all benchmarks"""
import datetime
import json
import subprocess


def main():
    """Generate the list of commands to be launched.

    One command == one benchmark (one model + one set of hyper-parameters + one dataset)
    """
    # Get all commands to benchmark each models using python benchmarks/BENCHMARK_FILE
    result = []

    # FIXME: add --short-list option to regression.py
    # FIXME: add regression.py generated commands
    # FIXME: add glm.py generated commands
    # FIXME: expose --short-list option in workflow yaml
    # https://github.com/zama-ai/concrete-ml-internal/issues/1652
    command_start = "python3 benchmarks/classification.py"
    now = datetime.datetime.now().strftime("%Y/%m/%d_%H:%M:%S")
    commands = [
        elt.replace('"', '\\"')  # Needed to escape " when calling eval
        for elt in subprocess.check_output(
            f"{command_start} --short_list",
            shell=True,
        )
        .decode("utf-8")
        .split("\n")
        if elt.startswith("--")  # avoid getting trash logs in there
    ]
    commands = [f"{command_start} {elt}" for elt in commands]

    for index, command in enumerate(commands):

        element = {"label": f"ml_bench_{now}_{index}", "index": index, "command": command}
        result.append(element)

    # Print to stdout to put in env var
    print(json.dumps(result))


if __name__ == "__main__":
    main()
