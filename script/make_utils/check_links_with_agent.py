"""Check external web links."""

import argparse
import re
import sys
from pathlib import Path
from typing import List

import requests


def check_links(file_path: Path, verbose: bool) -> List[str]:
    """Check the content of a markdown file for dead links.

    Args:
        file_path (Path): The path to the file.
        verbose (bool): Enable verbose output.

    Returns:
        List[str]: a list of errors (dead-links) found.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
    }

    broken_links = []

    # Read the file
    content = file_path.read_text(encoding="utf-8")

    # Use a regular expression to extract all links starting with https
    raw_links = re.findall(r"https://\S+", content)

    # Clean the links by removing everything after any of these characters: '", ], }, ), >'
    links_to_check = [re.split(r'["\]\}\)>]', link)[0] for link in raw_links]

    # Check each link
    for link in links_to_check:
        try:
            response = requests.get(link, headers=headers, timeout=10)
            if response.status_code == 200:
                status_message = f"OK: {link}"
            else:
                status_message = f"Failed: {link} (Status Code: {response.status_code})"
                broken_links.append(status_message)

        except requests.exceptions.RequestException as e:
            # Extract only the relevant part of the error message
            status_message = (
                f"Failed: {link} ({e.__class__.__name__}: {str(e).rsplit(':', maxsplit=1)[-1]})"
            )
            broken_links.append(status_message)

        if verbose:
            print(status_message)

    return broken_links


def main():
    """Main function"""

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Check web links in a file.")
    parser.add_argument("filename", help="The path to the file to check")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    args = parser.parse_args()

    if args.verbose:
        print(f"checking external links {args.filename}")

    # Create a Path object for the file
    root = Path(".")
    file_path = root / args.filename

    broken_links = check_links(file_path, args.verbose)

    # Exit with status code 1 if there are broken links
    if broken_links:
        print("\nBroken links:")
        sys.exit("\n".join(broken_links))


if __name__ == "__main__":

    main()
