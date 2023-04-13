"""Utils.

- Check if connection possible
- Wait for connection to be available (with timeout)
"""
import subprocess
import time
from pathlib import Path

from tqdm import tqdm


def filter_logs(previous_logs: str, current_logs: str) -> str:
    """Filter logs based on previous logs.

    Arguments:
        previous_logs (str): previous logs
        current_logs (str): current logs

    Returns:
        str: filtered logs
    """
    current_splitted = current_logs.split("\n")
    previous_splitted = set(previous_logs.split("\n"))

    for current_index, current_line in enumerate(current_splitted):
        if current_line not in previous_splitted:
            return "\n".join(current_splitted[current_index:])
    return ""


def wait_for_connection_to_be_available(
    hostname: str,
    ip_address: str,
    path_to_private_key: Path,
    timeout: int = 1,
    wait_time: int = 1,
    max_retries: int = 20,
    wait_bar: bool = False,
):
    """Wait for connection to be available.

    Arguments:
        hostname (str): host name
        ip_address (str): ip address
        path_to_private_key (Path): path to private key
        timeout (int): ssh timeout option
        wait_time (int): time to wait between retries
        max_retries (int): number of retries, if < 0 unlimited retries
        wait_bar (bool): tqdm progress bar of retries

    Raises:
        TimeoutError: if it wasn't able connect to ssh with the given constraints
    """
    with tqdm(disable=not wait_bar) as pbar:
        # We can't cover infinite retry without risking an infinite loop
        if max_retries < 0:  # pragma: no cover
            while True:
                if is_connection_available(
                    hostname=hostname,
                    ip_address=ip_address,
                    timeout=timeout,
                    path_to_private_key=path_to_private_key,
                ):
                    return
                time.sleep(wait_time)
                pbar.update(1)
        else:
            for _ in range(max_retries):
                if is_connection_available(
                    hostname=hostname,
                    ip_address=ip_address,
                    timeout=timeout,
                    path_to_private_key=path_to_private_key,
                ):
                    return
                time.sleep(wait_time)
                pbar.update(1)

    raise TimeoutError(
        "Timeout reached while trying to check for connection "
        f"availability on {hostname}@{ip_address}"
    )


def is_connection_available(
    hostname: str, ip_address: str, path_to_private_key: Path, timeout: int = 1
):
    """Check if ssh connection is available.

    Arguments:
        hostname (str): host name
        ip_address (str): ip address
        path_to_private_key (Path): path to private key
        timeout: ssh timeout option

    Returns:
        bool: True if connection succeeded
    """

    command = (
        f"ssh -i {path_to_private_key.resolve()} "
        + f"-q -o ConnectTimeout={timeout} -o BatchMode=yes -o "
        + f"\"StrictHostKeyChecking=no\" {hostname}@{ip_address} 'exit 0'"
    )
    try:
        subprocess.check_output(command, shell=True)
        return True
    except subprocess.CalledProcessError:
        return False
