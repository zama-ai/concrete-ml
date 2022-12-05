#!/usr/bin/env bash
set -e

# Remove "SC2086: Double quote to prevent globbing and word splitting." since it doesn't understand
# that sometimes, you want the word splitting
# shellcheck disable=SC2086

function nbqa_ize()
{
    NB="$1"
    PYLINT_EXTRA_OPTIONS="$2"

    OPTIONS=""

    if [ ${CHECK} -eq 1 ]
    then
        OPTIONS="$OPTIONS --check"
    fi

    # Tools which may change or check the notebooks
    poetry run nbqa isort "${NB}" ${OPTIONS} -l 100 --profile black
    poetry run nbqa black "${NB}" ${OPTIONS} -l 100

    # Tools which just checks the notebooks
    if [ ${CHECK} -eq 1 ]
    then
        # Ignore E402 module level import not at top of file, because it fails with
        #       %matplotlib inline
        # --extend-ignore=DAR is because we don't want to run darglint
        poetry run nbqa flake8 "${NB}" --max-line-length 100 --per-file-ignores="__init__.py:F401" \
            --ignore=E402 --extend-ignore=DAR

        # With some ignored errors, since we don't care:
        #       that the notebook filename is capitalized (invalid-name)
        #       that there are no docstring for the file since we have '\# comments' in
        #           the noteook (missing-module-docstring)
        #       that imports are done in a certain order to be more readable, or in sync with
        #           '\# comments' (wrong-import-position, ungrouped-imports, wrong-import-order)
        #       that the functions are not docstringed (missing-function-docstring)
        #       that the classes are not docstringed (missing-class-docstring)
        #       that some variable names are reused (redefined-outer-name)
        #
        # Also, --extension-pkg-whitelist=numpy is because of
        #       https://github.com/PyCQA/pylint/issues/1975
        poetry run nbqa pylint "${NB}" --rcfile=pylintrc --disable=invalid-name \
                --disable=missing-module-docstring --disable=missing-class-docstring \
                --disable=missing-function-docstring \
                --disable=wrong-import-position --disable=ungrouped-imports \
                --disable=wrong-import-order\
                --extension-pkg-whitelist=numpy --disable=redefined-outer-name \
                $PYLINT_EXTRA_OPTIONS
    fi
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
    echo "Running nbqa on docs"

    # We disable code-duplication check since we don't care that some tutorials share some code, we
    # want them to be as self-contained as possible
    PYLINT_EXTRA_OPTIONS="--disable duplicate-code"
    nbqa_ize docs "${PYLINT_EXTRA_OPTIONS}"

    # In addition, we disable import-error, because we may have packages which are just for
    # use_case_examples, ie not available in `make sync_env`
    PYLINT_EXTRA_OPTIONS="$PYLINT_EXTRA_OPTIONS --disable import-error"
    nbqa_ize use_case_examples "${PYLINT_EXTRA_OPTIONS}"

elif [ "$WHAT_TO_DO" == "one" ]
then
    echo "Running nbqa on ${NOTEBOOK}"
    PYLINT_EXTRA_OPTIONS=""
    nbqa_ize "${NOTEBOOK}" "${PYLINT_EXTRA_OPTIONS}"
fi


