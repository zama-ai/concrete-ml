"""Helper script to replace Replace `references/api/README.md` with `_apidoc/modules.html`."""

import argparse
import multiprocessing
import re
import sys
from pathlib import Path


def process_file(file_str: str) -> bool:
    """Replace `references/api/README.md` with `_apidoc/modules.html`.

    Also changes "references/api" pattern to Sphinx's "_apidoc"

    Args:
        file_str (str): the path to the file to process.

    Returns:
        bool: True once done
    """
    verbose = False

    file_path = Path(file_str).resolve()
    file_path_output = Path(file_str).resolve()

    if verbose:
        print(f"Fix api reference problems for {str(file_path)} into {str(file_path_output)}")

    api_pattern = "references/api/README.md"
    apidoc_pattern = "_apidoc/modules.html"

    processed_content = ""
    with open(file_path, "r", encoding="utf-8") as f:

        for line in f:
            newline = re.sub(api_pattern, apidoc_pattern, line)
            processed_content += newline

    with open(file_path_output, "w", encoding="utf-8") as fout:
        print(processed_content, file=fout, end="")

    return True


def main(args):
    """Entry point.

    Args:
        args (List[str]): a list of arguments
    """
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        res = pool.map(process_file, args.files)
        # Exit 0 if all went well as True == 1
        sys.exit(not all(res))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Replace `references/api/README.md` with `_apidoc/modules.html`.",
        allow_abbrev=False,
    )

    parser.add_argument(
        "--files", type=str, nargs="+", required=True, help="The files to modify in place"
    )

    cli_args = parser.parse_args()
    main(cli_args)
