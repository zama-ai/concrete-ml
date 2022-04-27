"""Update list of supported functions in the doc."""
import argparse
import tempfile
from pathlib import Path

import mdformat._cli as mdformat_cli

from concrete.ml.onnx.onnx_utils import ONNX_OPS_TO_NUMPY_IMPL
from concrete.ml.quantization.base_quantized_op import ONNX_OPS_TO_QUANTIZED_IMPL

SCRIPT_NAME = Path(__file__).name

CONVERSION_OPS_BEGIN_HEADER = (
    f"<!--- {SCRIPT_NAME}: inject supported operations for evaluation [BEGIN] -->"
)
CONVERSION_OPS_END_HEADER = (
    f"<!--- {SCRIPT_NAME}: inject supported operations for evaluation [END] -->"
)

PTQ_OPS_BEGIN_HEADER = f"<!--- {SCRIPT_NAME}: inject supported operations for PTQ [BEGIN] -->"

PTQ_OPS_END_HEADER = f"<!--- {SCRIPT_NAME}: inject supported operations for PTQ [END] -->"


def main(file_to_update):
    """Update list of supported functions in file_to_update"""
    supported_ops = sorted(ONNX_OPS_TO_NUMPY_IMPL)
    supported_ptq_ops = sorted(
        f"{op_name}: {quantized_op.__name__}"
        for op_name, quantized_op in ONNX_OPS_TO_QUANTIZED_IMPL.items()
    )

    with open(file_to_update, "r", encoding="utf-8") as file:
        lines = file.readlines()

    newlines = []
    keep_line = True

    for line in lines:
        if line.startswith(CONVERSION_OPS_BEGIN_HEADER):
            keep_line = False
            newlines.append(line)
            newlines.append(
                f"<!--- do not edit, auto generated part by `python3 {SCRIPT_NAME}` in docker -->\n"
            )
            newlines.extend(f"- {op}\n" for op in supported_ops)
        elif line.startswith(CONVERSION_OPS_END_HEADER):
            keep_line = True
            newlines.append(line)
        elif line.startswith(PTQ_OPS_BEGIN_HEADER):
            keep_line = False
            newlines.append(line)
            newlines.append(
                f"<!--- do not edit, auto generated part by `python3 {SCRIPT_NAME}` in docker -->\n"
            )
            newlines.extend(f"- {op}\n" for op in supported_ptq_ops)
        elif line.startswith(PTQ_OPS_END_HEADER):
            keep_line = True
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

    # We create a tempfile to format according to mdformat or we'll have issues with our tools
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp_f:
        tmp_file_path = Path(tmp_f.name)
        tmp_f.writelines(newlines)

    # Format with mdformat
    exit_code = mdformat_cli.run([str(tmp_file_path)])

    # Check formatting was ok
    if exit_code != 0:
        tmp_file_path.unlink(missing_ok=True)
        raise RuntimeError(f"Unable to format {str(tmp_file_path)}")

    # Get the formatted newlines
    with open(tmp_file_path, "r", encoding="utf-8") as f:
        newlines = f.readlines()

    # Remove the tmp file
    tmp_file_path.unlink(missing_ok=True)

    if args.check:

        with open(file_to_update, "r", encoding="utf-8") as file:
            oldlines = file.readlines()

        assert (
            oldlines == newlines
        ), "List of supported functions is not up to date. Please run `make supported_ops`."

    else:
        with open(file_to_update, "w", encoding="utf-8") as file:
            file.writelines(newlines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update list of supported functions in the doc")
    parser.add_argument("--check", action="store_true", help="flag to enable just checking mode")

    parser.add_argument("file_to_update", type=str, help=".md file to update")
    args = parser.parse_args()
    main(args.file_to_update)
