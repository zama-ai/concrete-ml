#!/usr/bin/env bash

# Disable these errors, whose fix is boring and make the script unreadable
#
#   https://www.shellcheck.net/wiki/SC2086 -- Double quote to prevent globbing ...
#   https://www.shellcheck.net/wiki/SC2129 -- Consider using { cmd1; cmd2; } >>...

# shellcheck disable=SC2086
# shellcheck disable=SC2129

# Before anything, install `gh`, which is GitHub CLI, https://github.com/cli/cli#installation
set -e

# Things you may want to change
FROM_WHEN="2023-01-01"
LIST_OF_REPOSITORIES=(concrete-ml-internal
                      concrete-ml)

# Will not work when we have more than 999 issues/PR, but does gh with search does not accept a
# larger size
NO_LIMIT=999
SUMMARY_FILE="$PWD/summary.txt"
TABLE_FILE="$PWD/table.txt"
CREATED_WHEN_OPTION='--search created:>='${FROM_WHEN}
CLOSED_WHEN_OPTION='--search closed:>='${FROM_WHEN}
NO_LIMIT_OPTION='-L '$NO_LIMIT
TEMP_DIR="temp_bcm"

function measure()
{
    # $1: the name of the repo
    # $2: the temp directory where to git clone
    # $3: the summary file where to dump info

    REPO_NAME="$1"
    TEMP_DIR="$2"
    SUMMARY_FILE="$3"
    TABLE_FILE="$4"

    OLD_PWD="$PWD"
    cd "$TEMP_DIR"
    echo "Getting statistics for $REPO_NAME repository"
    git clone https://github.com/zama-ai/$REPO_NAME 2>&1 | grep Cloning
    cd $REPO_NAME

    # Issues

    # Closed issues
    NB_CLOSED_ISSUES=$(gh  issue list ${NO_LIMIT_OPTION} --state closed ${CLOSED_WHEN_OPTION}  | wc -l)

    # Created issues
    NB_CREATED_ISSUES=$(gh issue list ${NO_LIMIT_OPTION} --state all    ${CREATED_WHEN_OPTION} | wc -l)

    # PR

    # Closed PR
    NB_CLOSED_PR=$(gh      pr    list ${NO_LIMIT_OPTION} --state closed ${CLOSED_WHEN_OPTION}  | wc -l)

    # Created PR
    NB_CREATED_PR=$(gh     pr    list ${NO_LIMIT_OPTION} --state all    ${CREATED_WHEN_OPTION} | wc -l)

    # GitHub stars and forks of public repos
    NB_STARS=$(curl --silent https://api.github.com/repos/zama-ai/$REPO_NAME | grep 'stargazers_count' | cut -f 2 -d ":" | sed -e "s@ @@g" | sed -e "s@,@@g")
    NB_FORKS=$(curl --silent https://api.github.com/repos/zama-ai/$REPO_NAME | grep 'forks_count' | cut -f 2 -d ":" | sed -e "s@ @@g" | sed -e "s@,@@g")

    # Commits
    # NB_COMMITS_PER_USER=$(git shortlog -s -n --since $FROM_WHEN) would be the number of commits
    # per user
    NB_COMMITS_TOTAL=$(git shortlog -s -n --since $FROM_WHEN | cut -b 1-7 | paste -s -d "+" -  | bc)

    # Make summary
    echo "Statistics for $REPO_NAME"                                            >> "$SUMMARY_FILE"
    echo ""                                                                     >> "$SUMMARY_FILE"
    echo "    Closed issues (since $FROM_WHEN):           $NB_CLOSED_ISSUES"    >> "$SUMMARY_FILE"
    echo "    Created issues (since $FROM_WHEN):          $NB_CREATED_ISSUES"   >> "$SUMMARY_FILE"
    echo "    Closed pull requests (since $FROM_WHEN):    $NB_CLOSED_PR"        >> "$SUMMARY_FILE"
    echo "    Created pull requests (since $FROM_WHEN):   $NB_CREATED_PR"       >> "$SUMMARY_FILE"
    echo "" >> "$SUMMARY_FILE"

    echo "Number of commits (since $FROM_WHEN): $NB_COMMITS_TOTAL"              >> "$SUMMARY_FILE"

    echo "Number of stars: $NB_STARS" >> "$SUMMARY_FILE"
    echo "Number of forks: $NB_FORKS" >> "$SUMMARY_FILE"
    echo >> "$SUMMARY_FILE"

    cd "$OLD_PWD"

    # Make summary for a table dump
    #       Closed issues | Created issues | Close PR | Created PR | Commits | Stars | Forks
    echo -e $NB_CLOSED_ISSUES' \t '$NB_CREATED_ISSUES' \t '$NB_CLOSED_PR' \t '$NB_CREATED_PR' \t '$NB_COMMITS_TOTAL' \t '$NB_STARS' \t '$NB_FORKS >> "$TABLE_FILE"
}

# Main
echo "" > "$SUMMARY_FILE"
echo "" > "$TABLE_FILE"

rm -rf $TEMP_DIR
mkdir $TEMP_DIR

for REPOSITORIES in "${LIST_OF_REPOSITORIES[@]}"
do
    measure $REPOSITORIES $TEMP_DIR $SUMMARY_FILE $TABLE_FILE
done

rm -rf $TEMP_DIR

echo >> "$SUMMARY_FILE"

echo "Statistics for Community:"        >> "$SUMMARY_FILE"
python3 script/other/community_stats.py >> "$SUMMARY_FILE"
echo >> "$SUMMARY_FILE"

cat "$SUMMARY_FILE"

echo "Dump in Excel (repositories):"
cat "$TABLE_FILE"
echo

echo "Successful end"



