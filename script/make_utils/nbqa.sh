#!/usr/bin/env bash
set -e

# Remove "SC2086: Double quote to prevent globbing and word splitting." since it doesn't understand
# that sometimes, you want the word splitting
# shellcheck disable=SC2086

function nbqa_ize()
{
    NB="$1"

    OPTIONS=""

    if [ ${CHECK} -eq 1 ]
    then
        OPTIONS="$OPTIONS --check"
    fi

    # TODO: add pylint, flake8, mypy, pydocstyle
    # #141, #142, #143, #144
    poetry run nbqa isort "${NB}" ${OPTIONS} -l 100 --profile black
    poetry run nbqa black "${NB}" ${OPTIONS} -l 100

}

WHAT_TO_DO="all"
CHECK=0

while [ -n "$1" ]
do
   case "$1" in
        "--all_notebooks" )
            WHAT_TO_DO="all"
            ;;

        "--notebook" )
            WHAT_TO_DO="one"
            shift
            NOTEBOOK="$1"
            ;;

        "--check" )
            CHECK=1
            ;;

        *)
            echo "Unknown param : $1"
            exit 1
            ;;
   esac
   shift
done

if [ "$WHAT_TO_DO" == "all" ]
then
    LIST_OF_NOTEBOOKS=$(find docs -name "*.ipynb" | grep -v ".nbconvert" | grep -v "_build" | grep -v "ipynb_subtoolpoints")

    for NOTEBOOK in ${LIST_OF_NOTEBOOKS}
    do
        echo "Running ${NOTEBOOK}"
        nbqa_ize "${NOTEBOOK}"
    done

elif [ "$WHAT_TO_DO" == "one" ]
then
    echo "Running ${NOTEBOOK}"
    nbqa_ize "${NOTEBOOK}"
fi


