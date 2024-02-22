#!/usr/bin/env bash

set -e

METHOD="sync_env"
VERSION_LIST=""

while [ -n "$1" ]
do
   case "$1" in
        "--sync_env" )
            METHOD="sync_env"
            ;;

        "--wheel" )
            METHOD="wheel"
            ;;

        "--pip" )
            METHOD="pip"
            ;;

        "--clone" )
            METHOD="clone"
            ;;

        "--all" )
            VERSION_LIST="3.8 3.9 3.10"
            ;;

        "--version" )
            shift
            VERSION_LIST="${VERSION_LIST} $1"
            ;;

        *)
            echo "Unknown param : $1"
            exit 1
            ;;

   esac
   shift
done

if [ "${VERSION_LIST}" == "" ]
then
    VERSION_LIST="3.8 3.9 3.10"
fi

for VERSION in $VERSION_LIST
do
    VENV=".tmp_venv_${VERSION}"
    rm -rf "${VENV}"

    python"${VERSION}" -m venv "${VENV}"

    # shellcheck disable=SC1090,SC1091
    source "${VENV}"/bin/activate
    python --version

    CHECK_VERSION=$(python --version)

    if [[ "${CHECK_VERSION}" != *"${VERSION}"* ]]; then
        echo "Issue in the version"
        exit 255
    fi

    if [ "$METHOD" == "wheel" ]
    then
        # Delete the directory where the pypi wheel file will be created (if it already exists)
        rm -rf dist

        # Build the wheel file
        poetry build -f wheel

        # Install the dependencies as PyPI would do using the wheel file
        PYPI_WHEEL=$(find dist -type f -name "*.whl")
        python -m pip install "${PYPI_WHEEL}"

    elif [ "$METHOD" == "pip" ]
    then
        pip install concrete-ml

    elif [ "$METHOD" == "sync_env" ]
    then
        make sync_env
    elif [ "$METHOD" == "clone" ]
    then
        deactivate
        rm -rf "${VENV}"
        TMP_DIR=".tmp_dir_clone_${VERSION}"
        rm -rf "${TMP_DIR}"
        mkdir "${TMP_DIR}"
        cd "${TMP_DIR}"
        git clone https://github.com/zama-ai/concrete-ml
        cd concrete-ml
        make sync_env
        cd ../..
        rm -rf "${TMP_DIR}"
    else
        echo "Unsupported method: $METHOD"
        exit 255
    fi

    rm -rf "${VENV}"
done
