"""Small helper utility to preprocess benchmark results locally"""

import json
from typing import Set


def main():
    """Reads the benchmark result json and compiles all results to a CSV file"""

    # Load the json output of the benchmarks
    benchmark = None
    with open("progress.json", "r", encoding="utf8") as inp_fp:
        benchmark = json.load(inp_fp)

    # Get all unique measurement names
    unique_measurements: Set[str] = set()
    for target in benchmark["targets"].values():
        if "measurements" not in target:
            continue
        target_measurements = set(target["measurements"])
        unique_measurements = unique_measurements.union(target_measurements)

    # Open the output CSV file
    with open("progress.csv", "wt", encoding="utf8") as out_fp:
        # Add the classifier column and write the CSV header
        columns = ["classifier"] + sorted(list(unique_measurements))
        out_fp.write(",".join(columns) + "\n")

        # Re-scan all targets
        for target in benchmark["targets"].values():
            if "measurements" not in target:
                continue

            # Prepare an empty list of values, fill in the classifier column
            vals = ["" for _ in range(len(columns))]
            vals[0] = target["name"]

            # Extract measurements for this target and save in the right column
            for m in target["measurements"]:
                meas = target["measurements"][m]
                vals[columns.index(m)] = meas

            # Write the data line to the CSV
            out_fp.write(",".join(map(str, vals)) + "\n")


if __name__ == "__main__":
    main()
