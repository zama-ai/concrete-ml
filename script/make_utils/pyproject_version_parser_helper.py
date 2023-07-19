"""A helper file to parse the versions from pyproject.toml to install the proper RC releases"""

import argparse
from pathlib import Path
from typing import List

import tomlkit


def convert_pyproject_version_str_to_pip(pyproject_version_str: str) -> str:
    """Convert a pyproject.toml version to a pip installable one.

    Args:
        pyproject_version_str (str): the string to convert.

    Returns:
        str: the converted string.
    """
    # Any version
    if pyproject_version_str == "*":
        return ""
    pyproject_version_str = pyproject_version_str.replace("^", "~=")
    if "<" in pyproject_version_str or ">" in pyproject_version_str or "=" in pyproject_version_str:
        return pyproject_version_str

    return f"=={pyproject_version_str}"


def format_extras(extras_list: List[str]) -> str:
    """Format extras from pyproject.toml for pip.

    Args:
        extras_list (List[str]): list of extras for a package.

    Returns:
        str: the formatted extras string for pip installation.
    """

    extras_spec = f"{','.join(extras_list)}"
    if extras_spec != "":
        extras_spec = f"[{extras_spec}]"

    return extras_spec


def main(args):
    """Entrypoint

    Args:
        args (Namespace): arguments

    Raises:
        RuntimeError: if can't find package in dependencies.

    """

    pkg_name = args.get_pip_install_spec_for_dependency
    pyproject_path = Path(args.pyproject_toml_file)

    pyproject_content = None
    with open(pyproject_path, encoding="utf-8") as f:
        pyproject_content = tomlkit.loads(f.read())

    project_deps = pyproject_content["tool"]["poetry"]["dependencies"]  # type: ignore[index]
    pkg_infos = project_deps.get(pkg_name, None)  # type: ignore[union-attr]
    if pkg_infos is None:
        raise RuntimeError(f"No {pkg_name} in the project dependencies.")

    pip_install_spec = ""
    if isinstance(pkg_infos, dict):
        if "git" in pkg_infos:
            extras_spec = format_extras(pkg_infos.get("extras", []))
            git_rev = pkg_infos.get("rev", "")
            rev_spec = ""
            if git_rev != "":
                rev_spec = f"@{git_rev}"
            pip_install_spec = f"git+{pkg_infos['git']}{rev_spec}#egg={pkg_name}{extras_spec}"
        else:
            # plain version case
            extras_spec = format_extras(pkg_infos.get("extras", []))
            version_spec = pkg_infos.get("version", "")
            pip_version_spec = convert_pyproject_version_str_to_pip(version_spec)
            pip_install_spec = f"{pkg_name}{extras_spec}{pip_version_spec}"
    else:
        pip_version_spec = convert_pyproject_version_str_to_pip(pkg_infos)
        pip_install_spec = f"{pkg_name}{pip_version_spec}"

    print(pip_install_spec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pyproject version parser", allow_abbrev=False)

    parser.add_argument(
        "--pyproject-toml-file",
        type=str,
        help="Path to the pyproject.toml to use",
        required=True,
    )

    parser.add_argument(
        "--get-pip-install-spec-for-dependency",
        type=str,
        help=(
            "Indicate the name of the dependency package for which you want the pip install string"
        ),
        required=True,
    )

    cli_args = parser.parse_args()
    main(cli_args)
