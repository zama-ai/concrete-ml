"""Tool to bypass the insane logic of semantic-release and generate changelogs we want"""

import argparse
import subprocess
import sys
from collections import deque

from git.repo import Repo
from semantic_release.changelog import markdown_changelog
from semantic_release.errors import UnknownCommitMessageStyleError
from semantic_release.settings import config, current_commit_parser
from semantic_release.vcs_helpers import get_repository_owner_and_name
from semver import VersionInfo


def log_msg(*args, file=sys.stderr, **kwargs):
    """Shortcut to print to sys.stderr.

    Args:
        *args: print args
        file: where to log message
        **kwargs: print kwargs
    """
    print(*args, file=file, **kwargs)


def strip_leading_v(version_str: str):
    """Strip leading v of a version which is not SemVer compatible.

    Args:
        version_str: version as string (i.e., either `vX.Y.Z` or `X.Y.Z`)

    Returns:
        str: version without v

    """
    return version_str[1:] if version_str.startswith("v") else version_str


def get_poetry_project_version() -> VersionInfo:
    """Run poetry version and get the project version

    Returns:
        VersionInfo

    """
    command = ["poetry", "version"]
    poetry_version_output = subprocess.check_output(command, text=True)
    return version_string_to_version_info(poetry_version_output.split(" ")[1])


def raise_exception_or_print_warning(is_error: bool, message_body: str):
    """Raise an exception if is_error is true else print a warning to stderr

    Args:
        is_error (bool): if it's an error
        message_body (str): message of error

    Raises:
        RuntimeError: if is_error

    """
    msg_start = "Error" if is_error else "Warning"
    msg = f"{msg_start}: {message_body}"
    if is_error:
        raise RuntimeError(msg)
    log_msg(msg)


def version_string_to_version_info(version_string: str) -> VersionInfo:
    """Convert git tag to VersionInfo.

    Args:
        version_string (str): version as string

    Returns:
        VersionInfo

    """
    return VersionInfo.parse(strip_leading_v(version_string))


def generate_changelog(repo: Repo, from_commit_excluded: str, to_commit_included: str) -> dict:
    """Recreate the functionality from semantic release with the from and to commits.

    Args:
        repo (Repo): the gitpython Repo object representing your git repository
        from_commit_excluded (str): the commit after which we want to collect commit messages for
            the changelog
        to_commit_included (str): the last commit included in the collected commit messages for the
            changelog.

    Returns:
        dict: the same formatted dict as the generate_changelog from semantic-release
    """
    # Additional sections will be added as new types are encountered
    changes: dict = {"breaking": []}

    rev = f"{from_commit_excluded}...{to_commit_included}"

    for commit in repo.iter_commits(rev):
        hash_ = commit.hexsha
        commit_message = (
            commit.message.replace("\r\n", "\n")
            if isinstance(commit.message, str)
            else commit.message.replace(b"\r\n", b"\n")
        )
        try:
            message = current_commit_parser()(commit_message)
            if message.type not in changes:
                log_msg(f"Creating new changelog section for {message.type} ")
                changes[message.type] = []

            # Capitalize the first letter of the message, leaving others as they were
            # (using str.capitalize() would make the other letters lowercase)
            formatted_message = message.descriptions[0][0].upper() + message.descriptions[0][1:]
            if config.get("changelog_capitalize") is False:
                formatted_message = message.descriptions[0]

            # By default, feat(x): description shows up in changelog with the
            # scope in bold, like:
            #
            # * **x**: description
            if config.get("changelog_scope") and message.scope:
                formatted_message = f"**{message.scope}:** {formatted_message}"

            changes[message.type].append((hash_, formatted_message))

            if message.breaking_descriptions:
                # Copy breaking change descriptions into changelog
                for paragraph in message.breaking_descriptions:
                    changes["breaking"].append((hash_, paragraph))
            elif message.bump == 3:
                # Major, but no breaking descriptions, use commit subject instead
                changes["breaking"].append((hash_, message.descriptions[0]))

        except UnknownCommitMessageStyleError as err:
            log_msg(f"Ignoring UnknownCommitMessageStyleError: {err}")

    return changes


def main(args):
    """Entry point

    Args:
        args (List[str]): a list of strings

    """

    repo = Repo(args.repo_root)

    sha1_to_tags = {tag.commit.hexsha: tag for tag in repo.tags}

    # Get last commit to include in the changelog (found from the given sha, marked by
    # the given tag, ...)
    to_commit = repo.commit(args.to_ref)
    log_msg(f"To commit: {to_commit}")

    # Get the tag associated to the commit
    to_tag = sha1_to_tags.get(to_commit.hexsha, None)
    if to_tag is None:
        raise_exception_or_print_warning(
            is_error=args.to_ref_must_have_tag,
            message_body=f"to-ref {args.to_ref} has no tag associated to it",
        )

    # Get the version associated to the tag
    to_version = (
        get_poetry_project_version()
        if to_tag is None
        else version_string_to_version_info(to_tag.name)
    )
    log_msg(f"Project version {to_version} taken from tag: {to_tag}")

    from_commit = None
    from_commit_is_initial_commit = False

    # If no 'from_ref' reference was given, find the first commit to include in the changelog : the
    # commit represented by the latest stable release tag (not release candidate)
    if args.from_ref is None:

        # Get all actual release tag names (not release candidate ones) older than the target
        # version. This is because we want to generate the changelog for all new versions starting
        # from the latest non-release candidate.
        tags_by_name = {strip_leading_v(tag.name): tag for tag in repo.tags}
        versions_before_target_version = {
            VersionInfo.parse(tag_name): tags_by_name[tag_name]
            for tag_name in tags_by_name
            if VersionInfo.isvalid(tag_name)
            and VersionInfo.parse(tag_name)
            and VersionInfo.parse(tag_name).prerelease is None
            and VersionInfo.parse(tag_name) < to_version
        }
        log_msg(
            f"All release versions {versions_before_target_version} before version {to_version}"
        )

        # If at least one previous version has been found, get the commit from the latest stable
        # version
        if len(versions_before_target_version) > 0:
            highest_version_before_current_version = max(versions_before_target_version)
            highest_version_tag = versions_before_target_version[
                highest_version_before_current_version
            ]
            from_commit = highest_version_tag.commit

        # Else, get the initial commit reachable from 'to_commit' (from
        # https://stackoverflow.com/a/48232574)
        else:
            last_element_extractor = deque(repo.iter_commits(to_commit), 1)
            from_commit = last_element_extractor.pop()
            from_commit_is_initial_commit = True

    # Else, include all commits up to 'from_ref' reference
    else:
        from_commit = repo.commit(args.from_ref)

    log_msg(f"From commit: {from_commit}")

    # Get the oldest common commit between the new version and the previous latest one. If the tree
    # is clean, 'ancestor_commit' should be the same as 'from_commit'
    ancestor_commits = repo.merge_base(to_commit, from_commit)
    assert len(ancestor_commits) == 1
    ancestor_commit = ancestor_commits[0]
    assert ancestor_commit is not None
    log_msg(f"Common ancestor: {ancestor_commit}")

    if ancestor_commit != from_commit:
        do_not_change_from_ref = args.do_not_change_from_ref and args.from_ref is not None
        raise_exception_or_print_warning(
            is_error=do_not_change_from_ref,
            message_body=(
                f"the ancestor {ancestor_commit} for {from_commit} and {to_commit} "
                f"is not the same commit as the commit for '--from-ref' {from_commit}."
            ),
        )

    # Mypy does not seem to be able to see that 'to_commit' has a hexsha attribute, even
    # if we add some assert. THerefore, it is disabled
    ancestor_tag = sha1_to_tags.get(ancestor_commit.hexsha, None)  # type: ignore[union-attr]

    # The ancestor commit is allowed to not have a tag when generating changelogs only if no
    # previous version was found (the new version is the first one published in the repository).
    # Otherwise, an error is raised
    if ancestor_tag is None and not from_commit_is_initial_commit:
        raise_exception_or_print_warning(
            is_error=args.ancestor_must_have_tag,
            message_body=(
                f"the ancestor {ancestor_commit} for " f"{from_commit} and {to_commit} has no tag"
            ),
        )

    ancestor_version_str = (
        None if ancestor_tag is None else str(version_string_to_version_info(ancestor_tag.name))
    )

    log_msg(
        f"Collecting commits from \n{ancestor_commit} "
        f"(tag: {ancestor_tag} - parsed version "
        f"{str(ancestor_version_str)}) to \n{to_commit} "
        f"(tag: {to_tag} - parsed version {str(to_version)})"
    )

    # Generate the changelog with commits from the previous latest version to the new one
    # Mypy does not seem to be able to see that 'to_commit' has a hexsha attribute, even
    # if we add some assert. Therefore, it is disabled
    log_dict = generate_changelog(
        repo,
        ancestor_commit.hexsha,
        to_commit.hexsha,
    )  # type: ignore[union-attr]

    owner, name = get_repository_owner_and_name()
    md_changelog = markdown_changelog(
        owner,
        name,
        str(to_version),
        log_dict,
        header=True,
        previous_version=ancestor_version_str,
    )

    print(md_changelog)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Changelog helper", allow_abbrev=False)

    parser.add_argument("--repo-root", type=str, default=".", help="Path to the repo root")
    parser.add_argument(
        "--to-ref",
        type=str,
        help="Specify the git ref-like string (sha1, tag, HEAD~, etc.) that will mark the LAST "
        "included commit of the changelog. If this is not specified, the current project version "
        "will be used to create a changelog with the current commit as last commit.",
    )
    parser.add_argument(
        "--from-ref",
        type=str,
        help="Specify the git ref-like string (sha1, tag, HEAD~, etc.) that will mark the commit "
        "BEFORE the first included commit of the changelog. If this is not specified, the most "
        "recent actual release tag (no pre-releases) before the '--to-ref' argument will be used. "
        "If the tagged commit is not an ancestor of '--to-ref' then the most recent common ancestor"
        "(git merge-base) will be used unless '--do-not-change-from-ref' is specified.",
    )
    parser.add_argument(
        "--ancestor-must-have-tag",
        action="store_true",
        help="Set if the used ancestor must have a tag associated to it. If the ancestor commit is"
        " the initial commit from the repo then no error will be raised.",
    )
    parser.add_argument(
        "--to-ref-must-have-tag",
        action="store_true",
        help="Set if '--to-ref' must have a tag associated to it.",
    )
    parser.add_argument(
        "--do-not-change-from-ref",
        action="store_true",
        help="Specify to prevent selecting a different '--from-ref' than the one specified in cli. "
        "Will raise an exception if '--from-ref' is not a suitable ancestor for '--to-ref' and "
        "would otherwise use the most recent common ancestor (git merge-base) as '--from-ref'.",
    )

    cli_args = parser.parse_args()
    main(cli_args)
