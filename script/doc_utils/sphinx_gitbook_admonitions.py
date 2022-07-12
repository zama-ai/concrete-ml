"""Helper script to change admonitions from sphinx-style to gitbook-style, and vice-versa."""

import argparse
import multiprocessing
import sys
from functools import partial
from pathlib import Path


def process_file(file_str: str, args=None):
    """Change admonitions from sphinx-style to gitbook-style, and vice-versa.

    Args:
        file_str (str): the path to the file to process.
        args: the arguments of the call.
    """
    verbose = False

    file_path = Path(file_str).resolve()
    file_path_output = Path(file_str).resolve()

    if verbose:
        print(f"Changing admonitions for {str(file_path)} into {str(file_path_output)}")

    sphinx_to_gitbook_admonition = {"danger": "danger", "note": "info", "warning": "danger"}

    gitbook_to_sphinx_admonition = {
        value: key for key, value in sphinx_to_gitbook_admonition.items()
    }

    admonition = ""
    processed_content = ""

    with open(file_path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.rstrip()

            if args.gitbook_to_sphinx:
                # Gitbook to Sphinx
                if admonition != "" and line.startswith("{% endhint %}"):
                    # Closing admonition
                    admonition = ""
                    processed_content += "```"

                elif line.startswith("{% hint style='"):
                    # Starting admonition
                    admonition = line.replace("{% hint style='", "").replace("' %}", "")
                    gitbook_admonition = gitbook_to_sphinx_admonition[admonition]
                    processed_content += "```{" + gitbook_admonition + "}"

                else:
                    processed_content += line

            else:
                # Sphinx to Gitbook
                if admonition != "" and line.startswith("```"):
                    # Closing admonition
                    admonition = ""
                    processed_content += "{% endhint %}"

                elif line.startswith("```{"):
                    # Starting admonition
                    admonition = line.replace("```{", "").replace("}", "")
                    gitbook_admonition = sphinx_to_gitbook_admonition[admonition]
                    processed_content += "{% hint style='" + gitbook_admonition + "' %}"

                else:
                    processed_content += line

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
        "Change admonitions from sphinx-style to gitbook-style, and vice-versa", allow_abbrev=False
    )

    parser.add_argument(
        "--files", type=str, nargs="+", required=True, help="The files to modify in place"
    )

    parser.add_argument(
        "--gitbook_to_sphinx",
        action="store_true",
        help="Do from gitbook to sphinx (default is sphinx to gitbook)",
    )

    cli_args = parser.parse_args()
    main(cli_args)
