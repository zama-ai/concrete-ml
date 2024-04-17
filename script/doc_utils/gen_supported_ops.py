"""Update list of supported functions in the doc."""

import argparse
from pathlib import Path

from concrete.ml.onnx.onnx_utils import ONNX_OPS_TO_NUMPY_IMPL

SCRIPT_NAME = Path(__file__).name

CONVERSION_OPS_BEGIN_HEADER = (
    f"<!--- {SCRIPT_NAME}: inject supported operations for evaluation [BEGIN] -->"
)
CONVERSION_OPS_END_HEADER = (
    f"<!--- {SCRIPT_NAME}: inject supported operations for evaluation [END] -->"
)


def main(file_to_update):
    """Update list of supported functions in file_to_update

    Args:
        file_to_update (str): file to update
    """
    supported_ops = sorted(ONNX_OPS_TO_NUMPY_IMPL)

    with open(file_to_update, "r", encoding="utf-8") as file:
        lines = file.readlines()

    newlines = []
    keep_line = True

    for line in lines:
        if line.startswith(CONVERSION_OPS_BEGIN_HEADER):
            keep_line = False
            newlines.append(line)
            newlines.append("\n")
            newlines.append("<!--- do not edit, auto generated part by `make supported_ops` -->\n")
            newlines.append("\n")
            newlines.extend(f"- {op}\n" for op in supported_ops)
        elif line.startswith(CONVERSION_OPS_END_HEADER):
            keep_line = True
            newlines.append("\n")
            newlines.append(line)
        elif line.startswith("<!---"):
            pass
        else:
            assert f"{SCRIPT_NAME}" not in line, (
                f"Error: not expected to have '{SCRIPT_NAME}' at line {line} "
                f"of {file_to_update}"
            )

            if keep_line:
                newlines.append(line)

    # This code is seen as a duplicate with other scripts we have
    # pylint: disable=duplicate-code
    if args.check:

        with open(file_to_update, "r", encoding="utf-8") as file:
            oldlines = file.readlines()

        assert (
            oldlines == newlines
        ), "List of supported functions is not up to date. Please run `make supported_ops`. "
    # pylint: disable=duplicate-code

    else:
        with open(file_to_update, "w", encoding="utf-8") as file:
            file.writelines(newlines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update list of supported functions in the doc")
    parser.add_argument("--check", action="store_true", help="flag to enable just checking mode")

    parser.add_argument("file_to_update", type=str, help=".md file to update")
    args = parser.parse_args()
    main(args.file_to_update)
