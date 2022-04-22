"""Check that the release commit is part of the main or release branch"""

import argparse

from git.repo import Repo
from semver import VersionInfo


def main(args):
    """Entry point"""

    repo = Repo(args.repo_path)
    to_commit = repo.commit()
    to_version = args.git_tag

    if to_version.startswith("v"):
        to_version = to_version[1:]

    # The tag must be conform with the semver version
    if not VersionInfo.isvalid(to_version):
        raise RuntimeError(f"Invalid version: {to_version}")

    # Transform the parsed version to a branch name
    # e.g. 1.0.0-rc.1 => 1.0.x
    to_version = VersionInfo.parse(to_version)
    to_version = str(to_version.major) + "." + str(to_version.minor) + ".x"

    # Tagged commit for release (to_commit) should be part of the some specific branches.
    branches_to_check = ["main", f"release/{to_version}"]
    for branch in branches_to_check:
        if branch in repo.branches:
            branch_main = repo.branches[branch]
            ancestors_main = repo.merge_base(branch_main, to_commit)
            assert len(ancestors_main) == 1
            ancestor_main = ancestors_main[0]
            if ancestor_main.hexsha == to_commit.hexsha:
                break
    else:
        raise ValueError(
            f"Commit {to_commit.hexsha} is not part of the branches {branches_to_check}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "check release commit in main/release branch parser", allow_abbrev=False
    )

    parser.add_argument("--git-tag", type=str, required=True, help="Tag of the release")
    parser.add_argument("--repo-path", type=str, default=".", help="Path to the repo root")

    cli_args = parser.parse_args()
    main(cli_args)
