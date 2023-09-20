#!/usr/bin/env bash

set -e

ENV_VARS_FILE="${1}"

PIP_CONFIG="$(pip config list)"

SECRET_INDEX_URL="$(echo "${PIP_CONFIG}" | grep 'global.index-url' | cut -f 2 -d '=')"
SECRET_INDEX_URL="${SECRET_INDEX_URL//\'/}"
# Left as a comment for debugging, don't print in the terminal otherwise
# echo "pip index-url: ${SECRET_INDEX_URL}"

echo "PIP_INDEX_URL='${SECRET_INDEX_URL}'" >> "${ENV_VARS_FILE}"

# We could do something much more generic but for now this will be good enough
SECRET_EXTRA_INDEX_URL="$(echo "${PIP_CONFIG}" | grep 'global.extra-index-url' | cut -f 2 -d '=')"
SECRET_EXTRA_INDEX_URL="${SECRET_EXTRA_INDEX_URL//\'/}"
# Left as a comment for debugging, don't print in the terminal otherwise
# echo "pip extra-index-url: ${SECRET_EXTRA_INDEX_URL}"

if [[ "${SECRET_EXTRA_INDEX_URL}" != "" ]]; then
    # Sometimes, for no obvious reason, keyring is not installed, so let's reinstall it for more
    # reliance
    poetry run python -m pip install keyring

    CRED_JSON="$(python script/make_utils/pip_auth_util.py \
    --get-credentials-for "${SECRET_EXTRA_INDEX_URL}" \
    --check-netrc-first \
    --return-url-encoded-credentials)"
    #shellcheck disable=SC2002
    USER_ID="$(echo "${CRED_JSON}" | jq -rc '.user_id')"
    #shellcheck disable=SC2002
    PASSWORD="$(echo "${CRED_JSON}" | jq -rc '.password')"
    if [[ "${USER_ID}" != "" ]] && [[ "${PASSWORD}" != "" ]]; then
        AUTHENTICATED_URL="${SECRET_EXTRA_INDEX_URL/https:\/\//}"
        AUTHENTICATED_URL="https://${USER_ID}:${PASSWORD}@${AUTHENTICATED_URL}"
        echo "PIP_EXTRA_INDEX_URL='${AUTHENTICATED_URL}'" >> "${ENV_VARS_FILE}"
    fi
fi
