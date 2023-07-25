#!/usr/bin/env bash
set -e

DIR=$(dirname "$0")

# shellcheck disable=SC1090,SC1091
source "${DIR}/detect_docker.sh"

if isDocker; then
    echo "Can not run in docker -> this script needs to install new virtualenvs"
    exit 1
fi


CML_DIR=$(pwd)
USE_CASE_REL_DIR="use_case_examples"
USE_CASE_DIR="${CML_DIR}/${USE_CASE_REL_DIR}"

if [ ! -d "$USE_CASE_DIR" ]; then
    echo "This script must be run in the Concrete ML source root where the '$USE_CASE_REL_DIR' directory is present"
    exit 1
fi

echo "Refreshing notebooks with PIP installed Concrete ML"

# shellcheck disable=SC2143
if [[ $(git ls-files --others --exclude-standard | grep ${USE_CASE_REL_DIR}) ]]; then
    echo "This script must be run in a clean clone of the Concrete ML repo"
    echo "This directory has untracked files in ${USE_CASE_REL_DIR}"
    echo "You can LIST all untracked files using: "
    echo
    # shellcheck disable=SC2028
    echo "   git ls-files --others --exclude-standard | grep ${USE_CASE_REL_DIR}"
    echo
    echo "You can REMOVE all untracked files using: "
    echo
    # shellcheck disable=SC2028
    echo "   git ls-files --others --exclude-standard | grep ${USE_CASE_REL_DIR} | xargs -0 -d '\n' --no-run-if-empty rm"
    echo
    exit 1
fi

if [[ -z "${USE_CASE}" ]]; then
  # shellcheck disable=SC2207
  LIST_OF_USE_CASES=($(find "$USE_CASE_DIR/" -mindepth 1 -maxdepth 2 -type d | grep -v checkpoints))
else
  LIST_OF_USE_CASES=("${USE_CASE}")
  if [ ! -d "${USE_CASE}" ]; then 
    echo "The use case specified to be executed, ${USE_CASE}, does not exist"
    exit 1
  fi
fi

if [ ! "$(docker images -q zamafhe/concrete-ml:latest 2> /dev/null)" ]; then
    # BUILD THE DOCKER IMAGE
    echo "Building docker image"
    poetry build && mkdir -p pkg && cp dist/* pkg/ && make release_docker
    docker tag concrete-ml-release:latest zamafhe/concrete-ml:latest
fi

# shellcheck disable=SC2068
for EXAMPLE in ${LIST_OF_USE_CASES[@]}
do
    EXAMPLE_NAME=$(basename "${EXAMPLE}")

    if [ -f "${EXAMPLE}/Makefile" ]; then
        echo "*** Processing example ${EXAMPLE_NAME}"
    else
        continue
    fi

    # Setup a new venv
    VENV_PATH="/tmp/virtualenv_${EXAMPLE_NAME}"
    if [ -d "$VENV_PATH" ]; then
        echo " - VirtualEnv already exists, deleting the old one"
        rm -rf "$VENV_PATH"
    fi
    virtualenv -q "$VENV_PATH"
    echo " - VirtualEnv created at $VENV_PATH"
    # shellcheck disable=SC1090,SC1091
    source "${VENV_PATH}/bin/activate"
    # Install Concrete ML
    set +e
    cd "$CML_DIR"
    pip install -e . &> "/tmp/log_cml_pip_${EXAMPLE_NAME}"
    hresult=$?
    if [ $hresult -ne 0 ]; then
        echo "Could not install Concrete ML in the virtualenv, see /tmp/log_cml_pip_${EXAMPLE_NAME}"
        rm -rf "$VENV_PATH"
        continue
    fi
    set -e
    echo " - Concrete ML installed in $VENV_PATH"

    # Install example requirements
    cd "$EXAMPLE"
    if [ -f "requirements.txt" ]; then
        set +e
        pip install -r requirements.txt &> "/tmp/log_reqs_${EXAMPLE_NAME}"
        hresult=$?
        set -e
        if [ $hresult -ne 0 ]; then
            echo "Could not install Concrete ML in the virtualenv, see /tmp/log_reqs_${EXAMPLE_NAME}"
            rm -rf "$VENV_PATH"
            continue
        fi     
        echo " - Requirements installed in $VENV_PATH"
    fi
    
    set +e
    # Strip colors from the error output before piping to the log files
    # Swap stderr and stdout, all output of jupyter execution is in stderr
    # The information about time spent running the notebook is in stdout
    # The following will pipe the stderr to the regex so that it 
    # ends up in the log file.
    # The timing shows in the terminal
    USE_CASE_DIR=$USE_CASE_DIR make 3>&2 2>&1 1>&3- | perl -pe 's/\e([^\[\]]|\[.*?[a-zA-Z]|\].*?\a)//g' > "/tmp/log_${EXAMPLE_NAME}"

    # Neet to check the result of execution of the make command (ignore the results 
    # of the other commands in the pipe)
    hresult="${PIPESTATUS[0]}"
    if [ "$hresult" -ne 0 ]; then
        echo "Error while running example ${EXAMPLE_NAME} see /tmp/log_${EXAMPLE_NAME}"
    fi
    set -e             

    # Remove the virtualenv
    rm -rf "$VENV_PATH"
done
