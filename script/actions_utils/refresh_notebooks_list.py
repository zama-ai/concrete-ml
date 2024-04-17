"""Update the list of available notebooks for the refresh_one_notebook GitHib action."""

import argparse
from pathlib import Path

SCRIPT_NAME = Path(__file__).name
NOTEBOOKS_DIRS = [Path(r"docs/advanced_examples"), Path(r"use_case_examples")]

SMALL_TAB = "  "
TAB = SMALL_TAB * 4

# Header start and end for updating the current list of available notebooks
NOTEBOOKS_LIST_HEADER_START = (
    TAB + f"# --- {SCRIPT_NAME}: refresh list of notebooks currently available [START] ---"
)
NOTEBOOKS_LIST_HEADER_END = (
    TAB + f"# --- {SCRIPT_NAME}: refresh list of notebooks currently available [END] ---"
)

# Header start and end for updating the current path list of available notebooks
NOTEBOOK_PATHS_LIST_HEADER_START = (
    SMALL_TAB
    + f"# --- {SCRIPT_NAME}: refresh list of notebook paths currently available [START] ---"
)
NOTEBOOK_PATHS_LIST_HEADER_END = (
    SMALL_TAB + f"# --- {SCRIPT_NAME}: refresh list of notebook paths currently available [END] ---"
)

# Additional message indicating not to edit the headers
NO_EDIT_MESSAGE = "# --- do not edit, auto generated part by `make refresh_notebooks_list` ---\n"


# pylint: disable-next=too-many-statements
def main(file_to_update):
    """Update list of currently available notebooks in file_to_update

    Args:
        file_to_update (str): file to update
    """

    # Look for all available notebooks needing a refresh
    notebook_paths = [
        notebook_path
        for notebooks_path in NOTEBOOKS_DIRS
        for notebook_path in notebooks_path.rglob("*.ipynb")
        if not str(notebook_path).endswith("-checkpoint.ipynb")
    ]

    # Sort the paths by name for better display
    notebook_paths.sort(key=lambda path: path.stem.lower())

    # Open the file to update
    with open(file_to_update, "r", encoding="utf-8") as file:
        lines = file.readlines()

    notebooks_completed = False
    notebook_paths_completed = False
    current_header = None
    newlines = []
    keep_line = True

    for line in lines:
        # If the current line is the "notebook list" header start, update the list
        if line.startswith(NOTEBOOKS_LIST_HEADER_START):

            assert current_header is None, (
                "Start of header for refreshing the notebook list found while another header is "
                "being processed. A header start of end probably got (re)moved."
            )

            # Mark this section as currently being processed
            current_header = NOTEBOOKS_LIST_HEADER_START

            # The following lines should not be kept until the header's end is found
            keep_line = False

            # Keep the header line
            newlines.append(line)

            # Add the "no edit" warning message
            newlines.append(TAB + NO_EDIT_MESSAGE)

            # Append the complete list of current available notebooks with a specific format
            newlines.append(TAB + 'description: "Notebook file name only in: ' + r"\n" + "\n")
            newlines.extend(
                TAB + f"- {notebook_path.stem} " + r"\n" + "\n" for notebook_path in notebook_paths
            )
            newlines.append(TAB + '"\n')

        # Else, if the current line is the "notebook path list" header start, update the list
        elif line.startswith(NOTEBOOK_PATHS_LIST_HEADER_START):

            assert current_header is None, (
                "Start of header for refreshing the notebook path list found while another header "
                "is being processed. A header start of end probably got (re)moved."
            )

            # Mark this section as currently being processed
            current_header = NOTEBOOK_PATHS_LIST_HEADER_START

            # The following lines should not be kept until the header's end is found
            keep_line = False

            # Keep the header line
            newlines.append(line)

            # Add the "no edit" warning message
            newlines.append(SMALL_TAB + NO_EDIT_MESSAGE)

            # Append the complete list of current available notebook paths with a specific format
            newlines.extend(
                SMALL_TAB + f"{notebook_path.stem}: " + f'"{notebook_path}" \n'
                for notebook_path in notebook_paths
            )

        # Else, if the current line is a header end, stop the update
        elif line.startswith(NOTEBOOKS_LIST_HEADER_END):

            # Make sure the last header found is the header for refreshing the notebook list
            assert current_header == NOTEBOOKS_LIST_HEADER_START, (
                "End of header for refreshing the notebook list found before its start. Header "
                "start probably got (re)moved."
            )

            # Mark this section as completed
            notebooks_completed = True
            current_header = None

            # The following lines should be kept until a new header is found or the end of the file
            # is reached
            keep_line = True
            newlines.append(line)

        elif line.startswith(NOTEBOOK_PATHS_LIST_HEADER_END):

            # Make sure the last header found is the header for refreshing the notebook path list
            assert current_header == NOTEBOOK_PATHS_LIST_HEADER_START, (
                "End of header for refreshing the notebook path list found before its start. "
                "Header start probably got (re)moved."
            )

            # Mark this section as completed
            notebook_paths_completed = True
            current_header = None

            # The following lines should be kept until a new header is found or the end of the file
            # is reached
            keep_line = True
            newlines.append(line)

        # Else, if the current line is a comment, ignore it
        elif line.startswith(TAB + "# ---"):
            pass

        # Else, make sure the current line is correctly outside all header sections and then proceed
        # to the next line
        else:
            assert (
                f"{SCRIPT_NAME}" not in line
            ), f"'{SCRIPT_NAME}' at line {line} of {file_to_update} is not expected."

            if keep_line:
                newlines.append(line)

    assert notebooks_completed is True and notebook_paths_completed is True, (
        "Refreshing the notebook lists has not been properly completed. Some headers probably got "
        + "(re)moved. Failed refreshes: "
        + "Notebook list " * (not notebooks_completed)
        + "Notebook path list " * (not notebook_paths_completed)
    )

    # If the script was called for checking if the list has correctly been updated, call an assert
    # without updating the file
    if args.check:
        with open(file_to_update, "r", encoding="utf-8") as file:
            oldlines = file.readlines()

        assert (
            oldlines == newlines
        ), "List of notebooks is not up to date. Please run `make refresh_notebooks_list`."

    # Else, update the file
    else:
        with open(file_to_update, "w", encoding="utf-8") as file:
            file.writelines(newlines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update list of currently available notebooks")
    parser.add_argument("--check", action="store_true", help="flag to enable just checking mode")

    parser.add_argument("file_to_update", type=str, help=".yaml file to update")
    args = parser.parse_args()
    main(args.file_to_update)
