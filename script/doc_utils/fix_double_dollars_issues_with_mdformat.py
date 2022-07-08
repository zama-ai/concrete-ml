"""Helper script to replace $$, $/$ and /$$ by $$ or $, because of problems between mdformat
and GitBook."""

import argparse
import multiprocessing
import sys
from functools import partial
from pathlib import Path


def process_file(file_str: str, args=None):
    """Replace $$, $/$ and /$$ by $$ or $, because of problems between mdformat and GitBook.

    Args:
        file_str (str): the path to the file to process.
        args: the arguments of the call.
    """

    file_path = Path(file_str).resolve()
    file_path_output = Path(file_str).resolve()

    print(f"Fix double-dollar problems for {str(file_path)} into {str(file_path_output)}")
    processed_content = ""

    with open(file_path, "r", encoding="utf-8") as f:

        for line in f:

            newline = line.replace("$\\$", "$$").replace("\\$$", "$$")

            if args.single_dollar:
                newline = newline.replace("$$", "$")

            processed_content += newline

    processed_content += "\n"

    with open(file_path_output, "w", encoding="utf-8") as fout:
        print(processed_content, file=fout)

    return True


def main(args):
    """Entry point."""
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        res = pool.map(partial(process_file, args=args), args.files)
        # Exit 0 if all went well as True == 1
        sys.exit(not all(res))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Replace $$, $/$ and /$$ by $$ or $, because of problems between mdformat and GitBook",
        allow_abbrev=False,
    )

    parser.add_argument(
        "--files", type=str, nargs="+", required=True, help="The files to modify in place"
    )

    parser.add_argument(
        "--single_dollar", action="store_true", help="Replace to $ (default is replacing to $$)"
    )

    cli_args = parser.parse_args()
    main(cli_args)
