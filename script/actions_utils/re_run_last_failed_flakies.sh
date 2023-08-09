#!/usr/bin/env bash

set -e

# Check if the last failed tests are flaky tests or not. The 'pytest_check_last_failed_tests_are_flaky'
# command outputs a report message. If 'Last failed are flaky: true' is found in it, this means that
# all failed tests are known flaky tests. Else, if 'Last failed are flaky: false' is found, either 
# no tests failed or at least one of them is not a know flaky test. 
LAST_FAILED_TESTS_ARE_FLAKY=$(make pytest_check_last_failed_tests_are_flaky | sed -n -e 's/^Last failed are flaky: //p')

# If all failed tests are known flaky tests, re-run pytest on these tests only. If at least one test
# fails again, the script exits with status code 2. Else, exit with status code 1.
if [ "$LAST_FAILED_TESTS_ARE_FLAKY" == "true" ]
then
    make pytest_run_last_failed
else
    exit 1
fi
