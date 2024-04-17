"""Check linked github issues states"""

import json
import re
import subprocess
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict

ISSUE_URL_PATTERN = re.compile(
    r"(?:FIXME|TODO): https\:\/\/github\.com\/zama-ai\/concrete-ml-internal\/issues\/([0-9]+)"
)
EXTENSIONS_TO_CHECK = [".py", ".md"]
FOLDERS_TO_CHECK = ["src", "docs", "use_case_examples", "tests", "script"]
NUMBER_OF_CORES = max(cpu_count() - 2, 1)


class PathEncoder(json.JSONEncoder):
    """PathEncoder: a custom json encoder that supports Path"""

    def default(self, o):
        """

        Args:
            o: object to encode

        Returns:
            str

        """
        if isinstance(o, Path):
            return str(o.resolve())
        return super().default(o)


def check_file(path, root):
    """check_file: checks for issues linked in the file.

    Args:
        path (pathlib.Path): the path to the file to check
        root (pathlib.Path): the path to the project for which to check the state of the issue

    Returns:
        Dict[str, Dict[str, Any]]

    """
    issues = {}
    with open(path, "r", encoding="utf-8") as file:
        file_content = file.read()
    found_indexes = ISSUE_URL_PATTERN.findall(file_content)
    for index in found_indexes:
        if index not in issues:
            output = subprocess.check_output(
                shell=True,
                args=f'gh issue view {index} --json state --repo "zama-ai/concrete-ml-internal"',
                cwd=root,
            ).decode()
            issues[index] = json.loads(output)
            issues[index]["path"] = path

    return issues


def main():
    """

    Raises:
        Exception: if some links to not-opened github issues

    """
    root = Path(__file__).parent / "../.."  # Set correct path

    paths = (
        (path, root)
        for sub_folder in FOLDERS_TO_CHECK
        for path in (root / sub_folder).rglob("*")
        if any(path.name.endswith(extension) for extension in EXTENSIONS_TO_CHECK)
    )

    with Pool(NUMBER_OF_CORES) as pool:
        issues_list = pool.starmap(check_file, paths)

    issues: Dict[str, Dict[str, Any]] = {}
    for path_issues in issues_list:
        for issue_index, issue in path_issues.items():
            if issue_index in issues:
                issues[issue_index]["files"].append(issue["path"])
            else:
                issues[issue_index] = issue
                issues[issue_index]["files"] = [issue["path"]]
                del issues[issue_index]["path"]

    errors = {key: value for key, value in issues.items() if value["state"] != "OPEN"}
    if errors:
        raise Exception(
            "Some files refer to not opened issues:\n"
            f"{json.dumps(errors, indent=4, cls=PathEncoder)}"
        )


if __name__ == "__main__":
    main()
