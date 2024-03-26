import itertools
from functools import partial
from pathlib import Path

from concrete import fhe
from concrete.ml.pandas._utils import deserialize_evaluation_keys, serialize_evaluation_keys

_script_dir = Path(__file__).parent

_CLIENT_SERVER_DIR = _script_dir / "_client_server_files"
_CLIENT_PATH = _CLIENT_SERVER_DIR / "client.zip"
_SERVER_PATH = _CLIENT_SERVER_DIR / "server.zip"

_N_BITS_PANDAS = 4


def _identity_pbs(a):
    """Define an identity TLU."""
    return fhe.univariate(lambda x: x)(a)


@fhe.compiler(
    {"val_1": "encrypted", "val_2": "encrypted", "left_key": "encrypted", "right_key": "encrypted"}
)
def _left_right_join_to_compile(val_1, val_2, left_key, right_key):
    """Atomic function to compose during the left join.

    Args:
        val_1 (int): Value 1 to sum
        val_2 (int): Value 2 to sum
        left_key (int): Left key to match
        right_key (int): Right key to match

    Returns:
        int: Summed value
    """
    condition = left_key == right_key

    sum_on_condition = val_1 + (val_2 * condition)

    sum_with_tlu = _identity_pbs(sum_on_condition)

    return sum_with_tlu


def _get_left_right_join_max_value(n_bits: int):
    return 2**n_bits - 1


def _get_left_right_join_inputset(n_bits: int):
    """Define the inputset to use for the merge operator."""
    # Build the circuit using at most `n_bits` bits, which defines :
    # - the input's unsigned integer dtype allowed (at most)
    # - the maximum number of rows allowed in an input (assuming the merge is done on a column of
    # unsigned integers starting at 1)
    # Note that a 0 is used to represent a NaN (Not a Number) value
    max_row_allowed = high = _get_left_right_join_max_value(n_bits)

    inputset = list(
        itertools.product(
            [0, high],
            [0, high],
            [1, max_row_allowed],
            [1, max_row_allowed],
        )
    )

    return inputset


# Link the circuits and inputset generators to their operator's name
_PANDAS_OPS_TO_CIRCUIT_CONFIG = {
    "left_right_join": {
        "get_inputset": partial(_get_left_right_join_inputset, n_bits=_N_BITS_PANDAS),
        "to_compile": _left_right_join_to_compile,
        "encrypt_config": {
            "n": 4,
            "pos": 1,
        },
    }
}


def _get_encrypt_config():
    return _PANDAS_OPS_TO_CIRCUIT_CONFIG["left_right_join"]["encrypt_config"]


def _get_min_max_allowed():
    return (1, _get_left_right_join_max_value(_N_BITS_PANDAS))


def _save_client_server(config, force_save=True):
    if force_save or not _CLIENT_PATH.is_file() or not _SERVER_PATH.is_file():
        _CLIENT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SERVER_PATH.parent.mkdir(parents=True, exist_ok=True)

        inputset = config["get_inputset"]()
        cp_func = config["to_compile"]

        # Compile with composability
        merge_circuit = cp_func.compile(inputset, composable=True)

        # Save the client and server files
        merge_circuit.server.save(_SERVER_PATH, via_mlir=True)
        merge_circuit.client.save(_CLIENT_PATH)


def load_client(keygen=True, keys_path=None):
    client = fhe.Client.load(_CLIENT_PATH)

    if keygen:
        if keys_path is not None:
            client.keys.load_if_exists_generate_and_save_otherwise(keys_path)
        else:
            client.keygen(True)

    return client
