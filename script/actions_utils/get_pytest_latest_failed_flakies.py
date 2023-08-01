import json 
import argparse
import sys
from pathlib import Path


def main(args):
    json_path = Path(args.json_path)
    if json_path.is_file():
        with json_path.open('r') as f:
            failed_tests = json.load(f)
        
        if failed_tests:
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Check if Pytest's latest failed tests are known flakies", allow_abbrev=False)

    parser.add_argument(
        "json_path", type=str, required=True, help="The path to the json storing Pytest's latest failed tests."
    )

    cli_args = parser.parse_args()
    main(cli_args)

