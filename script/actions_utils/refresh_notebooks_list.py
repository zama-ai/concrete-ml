"""Update the list of available notebooks for the refresh_one_notebook GitHib action."""
import argparse
from pathlib import Path

SCRIPT_NAME = Path(__file__).name
NOTEBOOKS_DIR = Path(r"docs/advanced_examples")

TAB = "        "

NOTEBOOKS_LIST_BEGIN_HEADER = (
    TAB + f"# --- {SCRIPT_NAME}: refresh list of notebooks currently available [BEGIN] ---"
)
NOTEBOOKS_LIST_END_HEADER = (
    TAB + f"# --- {SCRIPT_NAME}: refresh list of notebooks currently available [END] ---"
)


def main(file_to_update):
    """Update list of currently available notebooks in file_to_update"""
    notebook_names = [notebook_path.stem for notebook_path in NOTEBOOKS_DIR.glob("*.ipynb")]

    with open(file_to_update, "r", encoding="utf-8") as file:
        lines = file.readlines()

    newlines = []
    keep_line = True

    for line in lines:
        if line.startswith(NOTEBOOKS_LIST_BEGIN_HEADER):
            keep_line = False
            newlines.append(line)
            newlines.append(
                TAB
                + "# --- do not edit, auto generated part by `make refresh_notebooks_list` ---\n"
            )
            newlines.append(TAB + 'description: "Notebook file name only in: ' + r"\n" + "\n")
            newlines.extend(
                TAB + f"- {notebook_name} " + r"\n" + "\n"
                for notebook_name in sorted(notebook_names)
            )
            newlines.append(TAB + '"\n')

        elif line.startswith(NOTEBOOKS_LIST_END_HEADER):
            keep_line = True
            newlines.append(line)

        elif line.startswith(TAB + "# ---"):
            pass

        else:
            assert f"{SCRIPT_NAME}" not in line, (
                f"Error: not expected to have '{SCRIPT_NAME}' at line {line} "
                f"of {file_to_update}"
            )

            if keep_line:
                newlines.append(line)

    if args.check:
        with open(file_to_update, "r", encoding="utf-8") as file:
            oldlines = file.readlines()

        assert (
            oldlines == newlines
        ), "List of notebooks is not up to date. Please run `make refresh_notebooks_list`."

    else:
        with open(file_to_update, "w", encoding="utf-8") as file:
            file.writelines(newlines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update list of currently available notebooks")
    parser.add_argument("--check", action="store_true", help="flag to enable just checking mode")

    parser.add_argument("file_to_update", type=str, help=".yaml file to update")
    args = parser.parse_args()
    main(args.file_to_update)
