"""Script to escape double quotes within brackets or curly braces."""

import argparse

parser = argparse.ArgumentParser(description="Escape double quotes in a string")
parser.add_argument("commands", help="String containing Python commands")
parser.add_argument(
    "--curly-braces-only",
    dest="curly_braces_only",
    action="store_true",
    help="Escape double quotes within curly braces",
)


def escape_quotes(raw_str, curly_braces_only=False):
    """Prints out an escaped version for double quotes of ``raw_str``.

    Args:
        raw_str: string containing double quotes
        curly_braces_only: escape quotes contained only within curly braces
    """
    out_string = ""
    delimiter = {"start": "[", "end": "]"} if not curly_braces_only else {"start": "{", "end": "}"}
    in_delimiters = False

    for c in raw_str:
        if not in_delimiters:
            out_string += c
        else:
            if c != '"':
                out_string += c
            else:
                out_string += r"\\\""

        if c == delimiter["start"]:
            in_delimiters = True
        elif c == delimiter["end"]:
            in_delimiters = False

    print(out_string)


if __name__ == "__main__":
    args = parser.parse_args()
    escape_quotes(args.commands, args.curly_braces_only)
