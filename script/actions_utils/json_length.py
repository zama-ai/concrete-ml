"""Script to evaluate the length of a json file"""

import argparse
import json
from pathlib import Path


def main():
    """Main function: computes the length of a json file.

    Raises:
        ValueError: if the json content is not a list.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-file", dest="json_file", type=Path, required=True)
    args = parser.parse_args()
    with open(args.json_file, mode="r", encoding="utf-8") as file:
        json_content = json.load(file)
    if not isinstance(json_content, list):
        raise ValueError(f"Content of {args.json_file} is not a list")
    print(len(json_content))


if __name__ == "__main__":
    main()
