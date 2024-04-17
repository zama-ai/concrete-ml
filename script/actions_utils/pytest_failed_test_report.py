"""Pytest JSON report on failed tests utils."""

import argparse
import json
from pathlib import Path
from typing import Dict, Optional


def write_failed_tests_comment(failed_tests_comment_path: Path, failed_tests_report: Dict):
    """Write a formatted PR comment for failed tests as a text file.

    Args:
        failed_tests_comment_path (Path): Path where to write the formatted comment.
        failed_tests_report (Dict): Formatted report of failed tests.
    """
    with open(failed_tests_comment_path, "w", encoding="utf-8") as f:

        # If at least one test failed, write the comment
        if failed_tests_report["tests_failed"]:

            # Write the comment's title and main header
            if failed_tests_report["all_failed_tests_are_flaky"]:
                f.write("## :warning: Known flaky tests have been rerun :warning:\n\n")
                failed_tests_header = (
                    "One or several tests initially failed but were identified as known flaky. "
                    "tests. Therefore, they have been rerun and passed. See below for more "
                    "details.\n\n"
                )
            else:
                f.write("## ❌ Some tests failed after rerun ❌\n\n")
                failed_tests_header = (
                    "At least one of the following tests initially failed. They have therefore"
                    "been rerun but failed again. See below for more details.\n\n"
                )

            f.writelines(failed_tests_header)

            # Open collapsible section
            f.write("<details><summary>Failed tests details</summary>\n<p>\n\n")

            # Write all flaky tests that initially failed as a list
            f.write("### Known flaky tests that initially failed: \n")
            for flaky_name in failed_tests_report["flaky"]:
                f.write("- " + flaky_name)

            # Write all other tests that initially failed as a list if they were not all known
            # flaky tests
            if not failed_tests_report["all_failed_tests_are_flaky"]:
                f.write("\n\n")

                f.write("### Other tests that initially failed: \n")
                for non_flaky_name in failed_tests_report["non_flaky"]:
                    f.write("- " + non_flaky_name)

            # Close collapsible section
            f.write("\n\n</p>\n</details>\n\n")


def write_failed_tests_report(
    failed_tests_report_path: Path,
    input_report: Dict,
    failed_tests_comment_path: Optional[Path] = None,
):
    """Write a formatted report of failed tests as a JSON file.

    Also optionally write a formatted PR comment for failed tests as a text file.

    Args:
        failed_tests_report_path (Path): Path where to write the formatted report of failed tests.
        input_report (Dict): Pytest overall report.
        failed_tests_comment_path (Optional[Path]): Path where to write the formatted PR comment.
            If None, no file is written. Default to None.
    """
    # Safest default parameters
    failed_tests_report = {
        "tests_failed": True,
        "flaky": [],
        "non_flaky": [],
        "all_failed_tests_are_flaky": False,
    }

    # Pytest exitcode 0 means all tests passed:
    # https://docs.pytest.org/en/7.1.x/reference/exit-codes.html
    # We therefore only need to check for failed tests if the exit code is 1 or above
    if input_report["exitcode"] != 0:

        # Retrieve all test reports
        tests_report = input_report["tests"]

        for test_report in tests_report:

            # If the test failed, check if it was marked as flaky or not
            if test_report["outcome"] == "failed":
                test_name = test_report["nodeid"]

                # Disable mypy as it does not seem to see that the following objects are lists
                if "flaky" in test_report["keywords"]:
                    failed_tests_report["flaky"].append(test_name)  # type: ignore[attr-defined]
                else:
                    failed_tests_report["non_flaky"].append(test_name)  # type: ignore[attr-defined]

        # If there are some flaky tests but no non-flaky tests failed, report that all failed tests
        # were known flaky tests
        # We need to make sure that at least one flaky test has been detected for one specific
        # reason: if, for example, a test file has a syntax error, pytest will "crash" and therefore
        # won't collect any tests in the file. The problem is that this will return an 'exitcode'
        # of 1, making this script unexpectedly return 'all_failed_tests_are_flaky=True' in the
        # case where 'failed_tests_report["non_flaky"]' is empty
        if failed_tests_report["flaky"] and not failed_tests_report["non_flaky"]:
            failed_tests_report["all_failed_tests_are_flaky"] = True

    else:
        failed_tests_report["tests_failed"] = False

    # Write the report
    with open(failed_tests_report_path, "w", encoding="utf-8") as f:
        json.dump(failed_tests_report, f, indent=4)

    # Write the PR comment if a path is given
    if failed_tests_comment_path is not None:
        write_failed_tests_comment(failed_tests_comment_path, failed_tests_report)


def main(args):
    """Entry point

    Args:
        args (List[str]): a list of arguments
    """

    # Read Pytest overall report
    input_report_path = Path(args.pytest_input_report).resolve()
    input_report = None
    with open(input_report_path, "r", encoding="utf-8") as f:
        input_report = json.load(f)

    failed_tests_report_path = Path(args.failed_tests_report).resolve()

    failed_tests_comment_path = args.failed_tests_comment
    if failed_tests_comment_path is not None:
        failed_tests_comment_path = Path(failed_tests_comment_path).resolve()

    write_failed_tests_report(
        failed_tests_report_path, input_report, failed_tests_comment_path=failed_tests_comment_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Pytest failed test report parser", allow_abbrev=False)

    parser.add_argument(
        "--pytest-input-report",
        type=str,
        help="Path to pytest's JSON report file to use",
        required=True,
    )
    parser.add_argument(
        "--failed-tests-report",
        type=str,
        help="Path where to write the failed tests JSON report",
        required=True,
    )
    parser.add_argument(
        "--failed-tests-comment",
        type=str,
        help="Path where to write the warning comment for failed tests as a txt file",
    )

    cli_args = parser.parse_args()

    main(cli_args)
