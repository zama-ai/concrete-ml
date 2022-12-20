#!/bin/env python
"""Check links to local files."""

import re
import sys
from pathlib import Path
from typing import List, Union

# A regex that matches [foo (bar)](my_link) and returns the my_link
# used to find all links made in our markdown files.
MARKDOWN_LINK_REGEX = [re.compile(r"\[[^\]]*\]\(([^\)]*)\)"), re.compile(r"href=\".*\"")]

# FIXME: add target check https://github.com/zama-ai/concrete-ml-internal/issues/1435


def check_content_for_dead_links(content: str, file_path: Path) -> List[str]:
    """Check the content of a markdown file for dead links.

    This checks a markdown file for dead-links to local files.

    Inputs:
        content (str): The content of the file.
        file_path (pathlib.Path): The path to the file.
    Outputs:
        List[str]: a list of errors (dead-links) found.

    """
    errors: List[str] = []
    links = []
    for regex in MARKDOWN_LINK_REGEX:
        links_found = regex.findall(content)
        for link in links_found:
            link = link.replace(r"\_", "_")  # for gitbook
            if "href=" in link:  # for html links
                link = link.replace('href="', "")  # remove href=""
                link = link[0:-1]  # remove last "
            links.append(link)

    for link in links:

        if link.startswith("http"):
            # This means this is a reference to a website
            continue
        if link.startswith("#"):
            # This means this is a reference to a header
            continue
        if link.startswith("mailto:"):
            # This means this is a reference to an email
            continue
        if "#" in link:
            # This means this is a reference to a file with header
            link = link.split("#")[0]

        link_path = file_path.parent / link
        ext = link_path.suffix
        link_path_no_ext = link_path.parent / link_path.stem

        if ext == ".html":
            rst_alternative = link_path_no_ext.with_suffix(".rst")
            if not link_path.exists() and not rst_alternative.exists():
                errors.append(
                    f"{file_path} contains a link to {link_path} "
                    f"could not find either files:\n{link_path}\n{rst_alternative}"
                )
            continue

        if not link_path.exists():
            errors.append(f"{file_path} contains a link to {link_path} " "that can't be found")
    return errors


def is_relative_to(path: Path, other_path: Union[str, Path]) -> bool:
    """Implementation of is_relative_to

    is_relative_to is not available until python 3.9
    https://docs.python.org/3.9/library/pathlib.html#pathlib.PurePath.is_relative_to
    """
    try:
        path.relative_to(other_path)
        return True
    except ValueError:
        return False


def main():
    """Main function

    Check all files (except those that match a pattern in .gitignore) for
    dead links to local files.
    """
    root = Path(".")
    errors: List[str] = []

    gitignore_file = root / ".gitignore"
    if gitignore_file.exists():
        with gitignore_file.open(encoding="UTF-8") as file:
            ignores = file.read().split("\n")
            ignores = [elt for elt in map(lambda elt: elt.split("#")[0].strip(), ignores) if elt]

    for path in root.glob("**/*"):
        if (
            path.is_file()
            and path.suffix == ".md"
            and not any(is_relative_to(path, ignore) for ignore in ignores)
        ):
            print(f"checking {path}")
            with path.open() as file:
                file_content = file.read()
            errors += check_content_for_dead_links(file_content, path)

    if errors:
        sys.exit("\n".join(errors))


if __name__ == "__main__":
    main()
