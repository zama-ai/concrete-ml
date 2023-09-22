"""Helper script to turn GitBook card boards into something sphinx-style."""

import argparse
import multiprocessing
import sys
from functools import partial
from pathlib import Path


def process_file(file_str: str):
    """Change table-car from gitbook-style to sphinx-style

    Args:
        file_str (str): the path to the file to process.

    Raises:
        ValueError: value error

    Returns:
        True if everything went alright.
    """

    file_path = Path(file_str).resolve()
    file_path_output = Path(file_str).resolve()

    processed_content = ""
    has_started = False
    which_line = 0
    how_many = 0

    with open(file_path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.rstrip()

            if "<!--- start -->" in line:
                assert not has_started
                has_started = True
                which_line = 0

                processed_content += line

            elif "<!--- end -->" in line:
                assert has_started
                has_started = False
                how_many += 1

                processed_content += line
            elif has_started:
                if which_line == 0:
                    # Find the png
                    which_line += 1
                    png = line.split(".gitbook/assets/")[1].split('"')[0]
                elif which_line == 1:
                    # Find link
                    link = line.split('a href="')[1].split('">')[0]
                    # Add the line
                    processed_content += (
                        f'<td><a href="{link}"><img src="../_images/{png}" width=200></a></td>'
                    )

                else:
                    raise ValueError
            else:
                processed_content += line

            processed_content += "\n"

    with open(file_path_output, "w", encoding="utf-8") as fout:
        print(processed_content, file=fout)

    assert how_many == 8

    return True


def main(args):
    """Entry point.

    Args:
        args: a Namespace object

    """
    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        res = pool.map(partial(process_file), args.files)
        # Exit 0 if all went well as True == 1
        sys.exit(not all(res))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Change table-cards from gitbook-style to sphinx-style", allow_abbrev=False
    )

    parser.add_argument(
        "--files", type=str, nargs="+", required=True, help="The files to modify in place"
    )

    cli_args = parser.parse_args()
    main(cli_args)
