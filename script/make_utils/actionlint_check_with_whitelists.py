""" Check an actionlint log against some whitelists """
import sys
from typing import Set

# Exact lines which are whitelisted
whitelisted_lines: Set[str] = set()

# Pattern which are whitelisted
whitelisted_pattern: Set[str] = {
    "matrix.runs_on",
    "matrix.python_version",
    "matrix: ${{ fromJSON(format('{{\"include\":{0}}}', "
    + "needs.start-runner-linux.outputs.matrix)) }}",
    "matrix: ${{ fromJSON(format('{{\"include\":{0}}}', "
    + "needs.matrix-preparation.outputs.macos-matrix)) }}",
}


def main():
    """Do the test"""
    status = 0

    for line in sys.stdin:
        if line in whitelisted_lines:
            continue

        is_bad_line = True

        for pattern in whitelisted_pattern:
            if pattern in line:
                is_bad_line = False
                break

        if is_bad_line:
            print("->", line)
            status = 1

    if status:
        print("Some non whitelisted errors, look at full log file")
        raise ValueError


if __name__ == "__main__":
    main()
