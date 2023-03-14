"""Script to generate custom GitHub actions test matrices."""

import argparse
import enum
import json
from pathlib import Path


class EnumEncoder(json.JSONEncoder):
    """JSON encoder for python enumerates"""

    def default(self, o):
        if isinstance(o, enum.Enum):
            return o.value
        return json.JSONEncoder.default(self, o)


class PythonVersion(enum.Enum):
    """Enumerate for python versions"""

    V_3_8 = "3.8"
    V_3_9 = "3.9"
    V_3_10 = "3.10"


class OS(enum.Enum):
    """Enumerate for OS"""

    LINUX = "linux"
    MACOS = "macos"


OS_VERSIONS = {
    OS.LINUX: "ubuntu-20.04",
    OS.MACOS: "macos-12",
}


def main(args):
    """Entry point to generate CI test matrix.

    Args:
        args (List[str]): a list of arguments
    """
    github_action_matrix = []

    for python_version in args.linux_python_versions:
        github_action_matrix.append(
            {
                "os_kind": OS.LINUX,
                "runs_on": OS_VERSIONS[OS.LINUX],
                "python_version": python_version,
            }
        )
    for python_version in args.macos_python_versions:
        github_action_matrix.append(
            {
                "os_kind": OS.MACOS,
                "runs_on": OS_VERSIONS[OS.MACOS],
                "python_version": python_version,
            }
        )

    print(json.dumps(github_action_matrix, indent=4, cls=EnumEncoder))

    output_json_path = args.output_json.resolve()

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(github_action_matrix, f, cls=EnumEncoder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Generate GHA test matrices", allow_abbrev=False)
    parser.add_argument(
        "--macos-python-versions",
        dest="macos_python_versions",
        nargs="*",
        required=True,
        type=PythonVersion,
        default=[],
        choices=set(PythonVersion),
        help="Python versions to use on mac systems",
    )
    parser.add_argument(
        "--linux-python-versions",
        dest="linux_python_versions",
        nargs="*",
        required=True,
        type=PythonVersion,
        default=[],
        choices=set(PythonVersion),
        help="Python versions to use on linux systems",
    )
    parser.add_argument(
        "--output-json", type=Path, required=True, help="Where to output the matrix as json data"
    )
    cli_args = parser.parse_args()
    main(cli_args)
