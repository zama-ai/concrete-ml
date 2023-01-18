"""Script to manage credentials for pip for docker and CI"""

import argparse
import json
import netrc
import sys
from typing import List, Optional, Tuple
from urllib.parse import quote, urlparse

import keyring

NETRC_METHOD = "netrc"
KEYRING_METHOD = "keyring"


def get_credentials_source_order(check_netrc_first: bool) -> List[str]:
    """Get the order of sources to check.

    Args:
        check_netrc_first (bool): Should netrc file be checked first.

    Returns:
        List[str]: The method names to use returned in the order they need to be checked.
    """

    return [NETRC_METHOD, KEYRING_METHOD] if check_netrc_first else [KEYRING_METHOD, NETRC_METHOD]


def get_credentials_from_netrc(get_credentials_for: str) -> Optional[Tuple[str, str]]:
    """Get credentials from a netrc file.

    Args:
        get_credentials_for (str): The name/url/machine for which to get the credentials.

    Returns:
        Optional[Tuple[str, str]]: The user_id to use and corresponding password if they were found.
    """
    try:
        netrc_file = netrc.netrc()
    except FileNotFoundError:
        print("No .netrc file found.", file=sys.stderr)
        return None
    credentials = netrc_file.authenticators(get_credentials_for)
    if credentials is None:
        return None
    login, account, password = credentials
    user_id = login if login is not None else account
    # For mypy
    assert password is not None
    return user_id, password


def get_credentials_from_keyring(get_credentials_for: str) -> Optional[Tuple[str, str]]:
    """Get credentials from a keyring.

    Args:
        get_credentials_for (str): The name/url/machine for which to get the credentials.

    Returns:
        Optional[Tuple[str, str]]: The user_id to use and corresponding password if they were found.
    """
    credentials = keyring.get_credential(get_credentials_for, None)
    if credentials is None:
        return None
    return credentials.username, credentials.password


METHOD_TO_FUNC = {
    KEYRING_METHOD: get_credentials_from_keyring,
    NETRC_METHOD: get_credentials_from_netrc,
}


def main(args):
    """Entry point.

    Args:
        args (List[str]): a list of arguments
    """

    parsed_url = urlparse(args.get_credentials_for)
    netloc = parsed_url.netloc
    get_credentials_for = netloc if netloc != "" else parsed_url.path
    source_order = get_credentials_source_order(args.check_netrc_first)
    for source in source_order:
        get_credentials_func = METHOD_TO_FUNC[source]
        credentials = get_credentials_func(get_credentials_for)
        if credentials is not None:
            break

    if credentials is None:
        print("No credentials found!", file=sys.stderr)
        cred_json = {"user_id": "", "password": ""}
        print(json.dumps(cred_json))
        return

    user_id, password = credentials
    if args.return_url_encoded_credentials:
        user_id = quote(user_id, safe="")
        password = quote(password, safe="")
    cred_json = {"user_id": user_id, "password": password}
    print(json.dumps(cred_json))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("pip auth utils", allow_abbrev=False)

    parser.add_argument(
        "--get-credentials-for",
        type=str,
        help="Specify the name/machine/url for which to get the user credentials if available. "
        "It will first check the user's keyring (if any) and then the .netrc file.",
    )

    parser.add_argument(
        "--check-netrc-first",
        action="store_true",
        help="Specify to first query the .netrc for credentials.",
    )

    parser.add_argument(
        "--return-url-encoded-credentials",
        action="store_true",
        help="Specify to get the string 'username:password' encoded to be used in a url like "
        "https://username:password@myrepo.com/",
    )

    cli_args = parser.parse_args()

    main(cli_args)
