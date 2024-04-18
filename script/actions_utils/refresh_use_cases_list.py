"""Updates or checks the run_one_use_case_example.yaml workflow with available use cases."""

import argparse
from pathlib import Path

SCRIPT_NAME = Path(__file__).name
USE_CASES_DIRS = [Path(r"use_case_examples")]

SMALL_TAB = "  "
TAB = SMALL_TAB * 5

# Headers for updating sections
SECTION_HEADERS = {
    "use_cases": {
        "start": TAB
        + f"# --- {SCRIPT_NAME}: refresh list of use cases currently available [START] ---",
        "end": TAB
        + f"# --- {SCRIPT_NAME}: refresh list of use cases currently available [END] ---",
    },
    "paths": {
        "start": SMALL_TAB
        + f"# --- {SCRIPT_NAME}: refresh list of use case paths currently available [START] ---",
        "end": SMALL_TAB
        + f"# --- {SCRIPT_NAME}: refresh list of use case paths currently available [END] ---",
    },
}


def no_edit_message(indent):
    """Generates a no edit message with appropriate indentation.

    Args:
        indent (str): The indentation to apply before the message.

    Returns:
        str: The formatted no edit message.
    """
    return indent + "# --- do not edit, auto generated part by `make refresh_use_cases_list` ---\n"


def find_use_cases_with_makefile():
    """Finds directories containing a Makefile, indicating a valid use case.

    Returns:
        list[Path]: A list of paths to directories with a Makefile.
    """
    use_case_paths = []
    for use_cases_dir in USE_CASES_DIRS:
        for use_cases_path in use_cases_dir.rglob("*"):
            if use_cases_path.is_dir() and (use_cases_path / "Makefile").exists():
                relative_path = Path(*use_cases_path.parts[1:])
                use_case_paths.append(relative_path)
    use_case_paths.sort(key=lambda path: path.as_posix().lower())
    return use_case_paths


def update_section(lines, start_header, end_header, new_content, indent):
    """Updates specific sections of the file with new content between start and end headers.

    Args:
        lines (list[str]): The original lines of the file.
        start_header (str): The starting header after which content should be updated.
        end_header (str): The ending header before which content should be updated.
        new_content (list[str]): The new content to insert between the headers.
        indent (str): The indentation used for formatting the new content.

    Returns:
        list[str]: The updated list of lines.
    """
    new_lines = []
    in_update_section = False

    for line in lines:
        if line.startswith(start_header):
            in_update_section = True
            new_lines.append(line)
            new_lines.append(no_edit_message(indent))
            new_lines.extend(new_content)
        elif line.startswith(end_header):
            in_update_section = False
            new_lines.append(line)
        elif not in_update_section:
            new_lines.append(line)

    return new_lines


def main(file_to_update, check_mode=False):
    """Updates or checks the specified file with the current list of use cases and their paths.

    Args:
        file_to_update (str): The path to the file to be updated.
        check_mode (bool): If True, checks the file for needed updates without writing changes.
    """
    use_case_paths = find_use_cases_with_makefile()

    with open(file_to_update, "r", encoding="utf-8") as file:
        lines = file.readlines()

    # Update use case descriptions
    use_cases_content = [TAB + f"- {path.as_posix()}\n" for path in use_case_paths]
    lines = update_section(
        lines,
        SECTION_HEADERS["use_cases"]["start"],
        SECTION_HEADERS["use_cases"]["end"],
        use_cases_content,
        TAB,
    )

    # Update use case paths
    use_case_paths_content = [
        SMALL_TAB + f'{path.stem}: "{path.as_posix()}"\n' for path in use_case_paths
    ]
    lines = update_section(
        lines,
        SECTION_HEADERS["paths"]["start"],
        SECTION_HEADERS["paths"]["end"],
        use_case_paths_content,
        SMALL_TAB,
    )

    if check_mode:
        with open(file_to_update, "r", encoding="utf-8") as file:
            assert (
                file.readlines() == lines
            ), "List of use cases is not up to date. Please run `make refresh_use_cases_list`."
    else:
        with open(file_to_update, "w", encoding="utf-8") as file:
            file.writelines(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update list of currently available use cases")
    parser.add_argument("--check", action="store_true", help="flag to enable just checking mode")
    parser.add_argument("file_to_update", type=str, help=".yaml file to update")
    args = parser.parse_args()
    main(args.file_to_update, check_mode=args.check)
