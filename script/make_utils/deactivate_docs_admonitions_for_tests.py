"""Helper script to disable admonitions but keep line numbers untouched."""

import argparse
import multiprocessing
import re
import sys
from pathlib import Path


def process_file(file_str: str):
    """Deactivate admonitions in a file.

    Args:
        file_str (str): the path to the file to process.

    Returns:
        True if everything went alright.
    """
    verbose = False
    file_path = Path(file_str).resolve()

    if verbose:
        print(f"Removing admonitions for: {str(file_path)}")

    file_content = None
    with open(file_path, "r", encoding="utf-8") as f:
        file_content = "".join(f.readlines())

    # Check https://regex101.com/ with python gms flags to understand why this works
    processed_content = re.sub(
        r"(```{)(.*?)(```)", r"ignore\1\2ignore\3", file_content, flags=re.DOTALL
    )
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(processed_content)

    return True


def main(args):
    """Entry point.

    Args:
        args: a Namespace object
    """
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        res = pool.map(process_file, args.files)
        # Exit 0 if all went well as True == 1
        sys.exit(not all(res))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Deactivate admonitions without changing line numbers", allow_abbrev=False
    )

    parser.add_argument(
        "--files", type=str, nargs="+", required=True, help="The files to modify in place"
    )

    cli_args = parser.parse_args()
    main(cli_args)
