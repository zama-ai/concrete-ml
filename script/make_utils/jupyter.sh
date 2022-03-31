#!/usr/bin/env bash
set -e

DIR=$(dirname "$0")

# shellcheck disable=SC1090,SC1091
source "${DIR}/detect_docker.sh"

WHAT_TO_DO="open"

while [ -n "$1" ]
do
   case "$1" in
        "--run_all_notebooks" )
            WHAT_TO_DO="run_all_notebooks"
            ;;

        "--run_all_notebooks_parallel" )
            WHAT_TO_DO="run_all_notebooks_parallel"
            ;;

        "--run_notebook" )
            WHAT_TO_DO="run_notebook"
            shift
            NOTEBOOK="$1"
            ;;

        "--open" )
            WHAT_TO_DO="open"
            ;;

        *)
            echo "Unknown param : $1"
            exit 1
            ;;
   esac
   shift
done

if [ "$WHAT_TO_DO" == "open" ]
then
    if isDockerContainer; then
        poetry run jupyter notebook --allow-root --no-browser --ip=0.0.0.0
    else
        poetry run jupyter notebook --allow-root
    fi
elif [ "$WHAT_TO_DO" == "run_all_notebooks" ]
then
    LIST_OF_NOTEBOOKS=$(find docs -name "*.ipynb" | grep -v ".nbconvert" | grep -v "_build" | grep -v "ipynb_checkpoints")

    for NOTEBOOK in ${LIST_OF_NOTEBOOKS}
    do
        echo "Running ${NOTEBOOK}"
        jupyter nbconvert --to notebook --inplace --execute "${NOTEBOOK}"
    done

    # Then, one needs to sanitize the notebooks
elif [ "$WHAT_TO_DO" == "run_all_notebooks_parallel" ]
then
    echo "Refreshing notebooks"

    # shellcheck disable=SC2207
    NOTEBOOKS=($(find ./docs/ -type f -name "*.ipynb" | grep -v ".nbconvert" | grep -v "_build" | grep -v "ipynb_checkpoints"))
    PIDS_TO_WATCH=""

    # Run notebooks in sub processes
    # shellcheck disable=SC2068
    for NOTEBOOK in ${NOTEBOOKS[@]}; do
        ( jupyter nbconvert --to notebook --inplace --execute "${NOTEBOOK}" ) &
        # store PID of process
        NOTEBOOK_PID="$!"
        PIDS_TO_WATCH+=" ${NOTEBOOK_PID}"
        echo "Notebook ${NOTEBOOK} running with PID: ${NOTEBOOK_PID}"
    done

    STATUS=0
    for p in ${PIDS_TO_WATCH}; do
        if wait "${p}"; then
            echo "Process ${p} success"
        else
            echo "Process ${p} fail"
            STATUS=1
        fi
    done

    exit "${STATUS}"
elif [ "$WHAT_TO_DO" == "run_notebook" ]
then
    echo "Running ${NOTEBOOK}"
    jupyter nbconvert --to notebook --inplace --execute "${NOTEBOOK}"

    # Then, one needs to sanitize the notebooks
fi


