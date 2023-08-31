"""Tool to manage version in the project"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

import tomlkit
from semver import VersionInfo


def strip_leading_v(version_str: str):
    """Strip leading v of a version which is not SemVer compatible."

    Args:
        version_str: version as string

    Returns:
        str: version without v

    """
    return version_str[1:] if version_str.startswith("v") else version_str


def is_latest_entry(args):
    """ "is_latest command entry point.

    Args:
        args: a Namespace object

    Raises:
        RuntimeError: If the given version is not valid.
    """
    print(args, file=sys.stderr)

    # This is the safest default
    result = {"is_latest": False, "is_prerelease": True}

    new_version_str = strip_leading_v(args.new_version)
    if VersionInfo.isvalid(new_version_str):
        new_version_info = VersionInfo.parse(new_version_str)

        # If it is not a release candidate
        if new_version_info.prerelease is None:
            all_versions_str = (
                strip_leading_v(version_str) for version_str in args.existing_versions
            )

            # Keep versions that are not release candidate
            all_non_prerelease_version_infos = [
                VersionInfo.parse(version_str)
                for version_str in all_versions_str
                if VersionInfo.isvalid(version_str)
                and VersionInfo.parse(version_str)
                and VersionInfo.parse(version_str).prerelease is None
            ]

            all_non_prerelease_version_infos.append(new_version_info)

            new_version_is_latest = max(all_non_prerelease_version_infos) == new_version_info
            result["is_latest"] = new_version_is_latest
            result["is_prerelease"] = False

        print(json.dumps(result))

    else:
        raise RuntimeError(f"Version {args.version} is not valid.")


def is_prerelease_entry(args):
    """ "is_prerelease command entry point.

    Args:
        args: a Namespace object

    Raises:
        RuntimeError: If the given version is not valid.
    """
    version_str = strip_leading_v(args.version)
    if VersionInfo.isvalid(version_str):
        new_version_info = VersionInfo.parse(version_str)
        if new_version_info.prerelease is None:
            is_prerelease = False
        else:
            is_prerelease = True

        print(str(is_prerelease).lower())

    else:
        raise RuntimeError(f"Version {args.version} is not valid.")


def is_patch_release_entry(args):
    """ "is_patch_release command entry point.

    Args:
        args: a Namespace object

    Raises:
        RuntimeError: If the given version is not valid.
    """
    version_str = strip_leading_v(args.version)
    if VersionInfo.isvalid(version_str):
        new_version_info = VersionInfo.parse(version_str)
        if new_version_info.patch == 0 or new_version_info.prerelease is not None:
            is_patch_release = False
        else:
            is_patch_release = True

        print(str(is_patch_release).lower())

    else:
        raise RuntimeError(f"Version {args.version} is not valid.")


def update_variable_in_py_file(file_path: Path, var_name: str, version_str: str):
    """Update the version in a .py file.

    Args:
        file_path: path to file
        var_name: variable name
        version_str: version as string

    """

    file_content = None
    with open(file_path, encoding="utf-8") as f:
        file_content = f.read()

    updated_file_content = re.sub(
        rf'{var_name} *[:=] *["\'](.+)["\']',
        rf'{var_name} = "{version_str}"',
        file_content,
    )

    with open(file_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(updated_file_content)


def update_variable_in_toml_file(file_path: Path, var_name: str, version_str: str):
    """Update the version in a .toml file.

    Args:
        file_path: path to file
        var_name: variable name
        version_str: version as string

    """
    toml_content = None
    with open(file_path, encoding="utf-8") as f:
        toml_content = tomlkit.loads(f.read())

    toml_keys = var_name.split(".")
    current_content = toml_content  # type: ignore
    for toml_key in toml_keys[:-1]:
        current_content = current_content[toml_key]  # type: ignore
    last_toml_key = toml_keys[-1]
    current_content[last_toml_key] = version_str  # type: ignore

    with open(file_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(tomlkit.dumps(toml_content))


def load_file_vars_set(pyproject_path: os.PathLike, cli_file_vars: Optional[List[str]]):
    """Load files and their version variables set-up in pyproject.toml and passed as arguments.

    Args:
        pyproject_path: path to pyproject
        cli_file_vars: cli file variables

    Returns:
        set[unknown]: file variables

    """

    file_vars_set = set()
    if cli_file_vars is not None:
        file_vars_set.update(cli_file_vars)

    pyproject_path = Path(pyproject_path).resolve()

    # Check if there is a semantic release configuration
    if pyproject_path.exists():
        pyproject_content = None
        with open(pyproject_path, encoding="utf-8") as f:
            pyproject_content = tomlkit.loads(f.read())

        try:
            tool = pyproject_content["tool"]
            sr_conf = tool["semantic_release"]  # type: ignore
            sr_version_toml: str = sr_conf.get("version_toml", "")  # type: ignore
            file_vars_set.update(sr_version_toml.split(","))
            sr_version_variable: str = sr_conf.get("version_variable", "")  # type: ignore
            file_vars_set.update(sr_version_variable.split(","))
        except KeyError:
            print("No configuration for semantic release in pyproject.toml")

    return file_vars_set


def set_version(args):
    """set-version command entry point.

    Args:
        args: a Namespace

    Raises:
        RuntimeError: If unable to validate version

    """

    version_str = strip_leading_v(args.version)
    if not VersionInfo.isvalid(version_str):
        raise RuntimeError(f"Unable to validate version: {args.version}")

    file_vars_set = load_file_vars_set(args.pyproject_file, args.file_vars)

    for file_var_str in sorted(file_vars_set):
        print(f"Processing {file_var_str}")
        file, var_name = file_var_str.split(":", 1)
        file_path = Path(file).resolve()

        if file_path.suffix == ".py":
            update_variable_in_py_file(file_path, var_name, version_str)
        elif file_path.suffix == ".toml":
            update_variable_in_toml_file(file_path, var_name, version_str)
        else:
            raise RuntimeError(f"Unsupported file extension: {file_path.suffix}")


def get_variable_from_py_file(file_path: Path, var_name: str):
    """Read variable value from a .py file.

    Args:
        file_path: path to file
        var_name: variable name

    Returns:
        Set[str]: all values of the variables

    """
    file_content = None
    with open(file_path, encoding="utf-8") as f:
        file_content = f.read()

    variable_values_set = set()

    start_pos = 0
    while True:
        file_content = file_content[start_pos:]
        match = re.search(
            rf'{var_name} *[:=] *["\'](.+)["\']',
            file_content,
        )
        if match is None:
            break

        variable_values_set.add(match.group(1))
        start_pos = match.end()

    return variable_values_set


def get_variable_from_toml_file(file_path: Path, var_name: str):
    """Read variable value from a .toml file.

    Args:
        file_path: path to file
        var_name: variable name

    Returns:
        value of the variable

    """

    toml_content = None
    with open(file_path, encoding="utf-8") as f:
        toml_content = tomlkit.loads(f.read())

    toml_keys = var_name.split(".")
    current_content = toml_content
    for toml_key in toml_keys:
        current_content = current_content[toml_key]  # type: ignore

    return current_content


def check_version(args):
    """check-version command entry point.

    Args:
        args: Namespace

    Raises:
        RuntimeError: either wrong file extension or error with the version

    """

    version_str_set = set()

    file_vars_set = load_file_vars_set(args.pyproject_file, args.file_vars)

    for file_var_str in sorted(file_vars_set):
        print(f"Processing {file_var_str}")
        file, var_name = file_var_str.split(":", 1)
        file_path = Path(file).resolve()

        if file_path.suffix == ".py":
            version_str_set.update(get_variable_from_py_file(file_path, var_name))
        elif file_path.suffix == ".toml":
            version_str_set.add(get_variable_from_toml_file(file_path, var_name))
        else:
            raise RuntimeError(f"Unsupported file extension: {file_path.suffix}")

    if len(version_str_set) == 0:
        raise RuntimeError(f"No versions found in {', '.join(sorted(file_vars_set))}")
    if len(version_str_set) > 1:
        raise RuntimeError(
            f"Found more than one version: {', '.join(sorted(version_str_set))}\n"
            "Re-run make set-version"
        )
    # Now version_str_set len == 1
    version = next(iter(version_str_set))
    if not VersionInfo.isvalid(version):
        raise RuntimeError(f"Unable to validate version: {version}")

    print(f"Found version {version} in all processed locations.")

    if args.version is not None:
        if args.version != version:
            raise RuntimeError(
                f"Versions does not match. Found version {version} but got {args.version}"
            )

        print("Version found in files is correct.")


def main(args):
    """Entry point

    Args:
        args: a Namespace object

    """
    args.entry_point(args)


if __name__ == "__main__":
    main_parser = argparse.ArgumentParser("Version utils", allow_abbrev=False)

    sub_parsers = main_parser.add_subparsers(dest="sub-command", required=True)

    parser_is_latest = sub_parsers.add_parser("is_latest")
    parser_is_latest.add_argument(
        "--new-version", type=str, required=True, help="The new version to compare"
    )
    parser_is_latest.add_argument(
        "--existing-versions",
        type=str,
        nargs="+",
        required=True,
        help="The list of existing versions",
    )
    parser_is_latest.set_defaults(entry_point=is_latest_entry)

    parser_is_prerelease = sub_parsers.add_parser("is_prerelease")
    parser_is_prerelease.add_argument(
        "--version", type=str, required=True, help="The version to consider"
    )
    parser_is_prerelease.set_defaults(entry_point=is_prerelease_entry)

    parser_is_patch_release = sub_parsers.add_parser("is_patch_release")
    parser_is_patch_release.add_argument(
        "--version", type=str, required=True, help="The version to consider"
    )
    parser_is_patch_release.set_defaults(entry_point=is_patch_release_entry)

    parser_set_version = sub_parsers.add_parser("set-version")
    parser_set_version.add_argument("--version", type=str, required=True, help="The version to set")
    parser_set_version.add_argument(
        "--pyproject-file",
        type=str,
        default="pyproject.toml",
        help="The path to a project's pyproject.toml file, defaults to $pwd/pyproject.toml",
    )
    parser_set_version.add_argument(
        "--file-vars",
        type=str,
        nargs="+",
        help=(
            "A space separated list of file/path.{py, toml}:variable to update with the new version"
        ),
    )
    parser_set_version.set_defaults(entry_point=set_version)

    parser_check_version = sub_parsers.add_parser("check-version")
    parser_check_version.add_argument(
        "--pyproject-file",
        type=str,
        default="pyproject.toml",
        help="The path to a project's pyproject.toml file, defaults to $pwd/pyproject.toml",
    )
    parser_check_version.add_argument(
        "--file-vars",
        type=str,
        nargs="+",
        help=(
            "A space separated list of file/path.{py, toml}:variable to update with the new version"
        ),
    )
    parser_check_version.add_argument(
        "--version", type=str, required=False, help="The version to check"
    )
    parser_check_version.set_defaults(entry_point=check_version)

    cli_args = main_parser.parse_args()

    main(cli_args)
