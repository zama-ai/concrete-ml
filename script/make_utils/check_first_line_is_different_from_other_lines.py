"""Helper to check the first line is different from all other lines."""

import argparse
from pathlib import Path


def process_file(file_str: str):
    """Helper to check the first line is different from all other lines.

    Args:
        file_str (str): the path to the file to process.

    Returns:
        True if everything went alright.

    Raises:
        ValueError: if the first line is equal to another line in the file

    """
    file_path = Path(file_str).resolve()

    with open(file_path, "r", encoding="utf-8") as f:
        file_content = f.readlines()

    first_line = file_content[0]

    for new_line in file_content[1:]:
        is_equal_line = new_line == first_line
        if is_equal_line:
            raise ValueError("Error, the first line and another line are equal")

    assert len(file_content) > 1
    print(f"{file_path} looks good (number of lines: {len(file_content)})!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Helper to check the first line is different from all other lines", allow_abbrev=False
    )

    parser.add_argument("--file", type=str, required=True, help="The log files to check")

    cli_args = parser.parse_args()
    process_file(cli_args.file)
