#!/bin/bash

set -e

BASENAME="licenses"
LICENSE_DIRECTORY="deps_licenses"
CHECK=0
DIFF_TOOL="diff --ignore-all-space --ignore-tab-expansion --ignore-space-change --ignore-all-space --ignore-blank-lines --strip-trailing-cr"
TMP_VENV_PATH="/tmp/tmp_venv"
DO_USER_LICENSES=1

# Dev licences are not done, but could be re-enabled
DO_DEV_LICENSES=0

OUTPUT_DIRECTORY="${LICENSE_DIRECTORY}"
DO_FORCE_UPDATE=0

while [ -n "$1" ]
do
   case "$1" in
        "--check" )
            CHECK=1
            OUTPUT_DIRECTORY=$(mktemp -d)
            ;;

        "--force_update" )
            DO_FORCE_UPDATE=1
            ;;

        *)
            echo "Unknown param : $1"
            exit 1
            ;;
   esac
   shift
done

UNAME=$(uname)
if [ "$UNAME" == "Darwin" ]
then
    OS=mac
elif [ "$UNAME" == "Linux" ]
then
    OS=linux
else
    echo "Problem with OS"
    exit 255
fi

mkdir -p "${LICENSE_DIRECTORY}"

if [ $DO_USER_LICENSES -eq 1 ]
then
    #Licenses for user (install in a temporary venv)
    echo "Doing licenses for user"

    FILENAME="${BASENAME}_${OS}_user.txt"
    LICENSES_FILENAME="${LICENSE_DIRECTORY}/${FILENAME}"
    NEW_LICENSES_FILENAME="${OUTPUT_DIRECTORY}/${FILENAME}"

    # If the dependencies have not changed, don't do anything
    MD5_OLD_DEPENDENCIES=$(cat ${LICENSES_FILENAME}.md5)
    MD5_NEW_DEPENDENCIES=$(openssl md5 poetry.lock)

    echo "MD5 of the poetry.lock for which dependencies have been listed: ${MD5_OLD_DEPENDENCIES}"
    echo "MD5 of the current poetry.lock:                                 ${MD5_NEW_DEPENDENCIES}"

    if [ "${MD5_OLD_DEPENDENCIES}" == "${MD5_NEW_DEPENDENCIES}" ] && [ ${DO_FORCE_UPDATE} -ne 1 ]
    then
        echo "The lock file hasn't changed, early exit"
        exit 0
    fi

    rm -rf $TMP_VENV_PATH/tmp_venv
    python3 -m venv $TMP_VENV_PATH/tmp_venv

    # SC1090: Can't follow non-constant source. Use a directive to specify location.
    # shellcheck disable=SC1090,SC1091
    source $TMP_VENV_PATH/tmp_venv/bin/activate

    python -m pip install -U pip wheel
    python -m pip install -U --force-reinstall setuptools
    poetry install --no-dev
    python -m pip install pip-licenses

    # In --format=csv such that the column length (and so, the diff) do not change with longer
    # names
    pip-licenses --format=csv | tr -d "\"" | grep -v "pkg\-resources\|concrete-ml" | \
        tee "${NEW_LICENSES_FILENAME}"

    # Remove trailing whitespaces
    if [ "$UNAME" == "Darwin" ]
    then
        sed -i "" 's/[t ]*$//g' "${NEW_LICENSES_FILENAME}"
    else
        sed -i 's/[t ]*$//g' "${NEW_LICENSES_FILENAME}"
    fi

    # Replace "," by ", "
    sed -i "" 's/,/, /g' "${NEW_LICENSES_FILENAME}"

    deactivate

    if [ $CHECK -eq 1 ]
    then
        echo "$DIFF_TOOL $LICENSES_FILENAME ${NEW_LICENSES_FILENAME}"
        $DIFF_TOOL "$LICENSES_FILENAME" "${NEW_LICENSES_FILENAME}"
        echo "Success: no update in $LICENSES_FILENAME"
    else
        # Update the .md5 files
        openssl md5 poetry.lock > ${LICENSES_FILENAME}.md5
    fi
fi

if [ $DO_DEV_LICENSES -eq 1 ]
then
    # Licenses for developer (install in a temporary venv)
    echo "Doing licenses for developper"
    echo "Warning: contrarily to the user licences, there is currently no early abort if "
    echo "poetry.lock hasn't changed (in the dev licences). Please update"

    FILENAME="${BASENAME}_${OS}_dev.txt"
    LICENSES_FILENAME="${LICENSE_DIRECTORY}/${FILENAME}"
    NEW_LICENSES_FILENAME="${OUTPUT_DIRECTORY}/${FILENAME}"

    rm -rf $TMP_VENV_PATH/tmp_venv
    python3 -m venv $TMP_VENV_PATH/tmp_venv

    # SC1090: Can't follow non-constant source. Use a directive to specify location.
    # shellcheck disable=SC1090,SC1091
    source $TMP_VENV_PATH/tmp_venv/bin/activate

    make setup_env
    pip-licenses | grep -v "pkg\-resources\|concrete-ml" | tee "${NEW_LICENSES_FILENAME}"

    # Remove trailing whitespaces
    if [ "$UNAME" == "Darwin" ]
    then
        sed -i "" 's/[t ]*$//g' "${NEW_LICENSES_FILENAME}"
    else
        sed -i 's/[t ]*$//g' "${NEW_LICENSES_FILENAME}"
    fi

    deactivate

    if [ $CHECK -eq 1 ]
    then

        echo "$DIFF_TOOL $LICENSES_FILENAME ${NEW_LICENSES_FILENAME}"
        $DIFF_TOOL "$LICENSES_FILENAME" "${NEW_LICENSES_FILENAME}"
        echo "Success: no update in $LICENSES_FILENAME"
    fi
fi

rm -f ${LICENSE_DIRECTORY}/licenses_*.txt.tmp
rm -rf $TMP_VENV_PATH/tmp_venv

echo "End of license script"
