"""Check that the release commit is part of the main or release branch"""

import argparse

import git
from git.repo import Repo
from semver import VersionInfo


def main(args):
    """Entry point

    Args:
        args (List[str]): a list of arguments

    Raises:
        RuntimeError: if invalid version
        ValueError: if sha error
    """

    repo = Repo(args.repo_path)
    to_commit = repo.commit()
    to_version = args.git_tag

    if to_version.startswith("v"):
        to_version = to_version[1:]

    # The tag must be conform with the semver version
    if not VersionInfo.isvalid(to_version):
        raise RuntimeError(f"Invalid version: {to_version}")

    # Transform the parsed version to a branch name
    # e.g., 1.0.0-rc.1 => 1.0.x
    to_version = VersionInfo.parse(to_version)
    to_version = str(to_version.major) + "." + str(to_version.minor) + ".x"

    # Tagged commit for release (to_commit) should be part of the some specific branches.
    branches_to_check = ["main", f"release/{to_version}"]

    # Github CI will not have local branches, only remote references
    # Thus we check the presence of the tag commit in the remote references
    remote_refs = {}
    for ref in repo.references:
        # Remote refs may be prefixed by "refs/" in some cases, and always contain "remotes/origin"
        # Strip these path items to check against the target branch names
        stripped_name = ref.name.replace("refs/", "").replace("remotes/", "").replace("origin/", "")
        if isinstance(ref, git.RemoteReference) and stripped_name in branches_to_check:
            remote_refs[stripped_name] = ref

    for branch in branches_to_check:
        print(f"Checking branch: {branch}")
        # We need to check that the branch exists as a remote reference
        if branch in remote_refs:
            print(f"Branch found as {remote_refs[branch].name}")
            branch_main = remote_refs[branch]
            ancestors_main = repo.merge_base(branch_main, to_commit)
            assert len(ancestors_main) == 1
            ancestor_main = ancestors_main[0]

            # Mypy does not seem to be able to see that 'to_commit' has a hexsha attribute, even
            # if we add some assert. THerefore, it is disabled
            print(
                f"Checking current commit {to_commit.hexsha} against "  # type: ignore[union-attr]
                f"commit on branch {ancestor_main.hexsha}"
            )
            if ancestor_main.hexsha == to_commit.hexsha:  # type: ignore[union-attr]
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
