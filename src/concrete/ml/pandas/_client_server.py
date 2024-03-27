import itertools
from functools import partial
from pathlib import Path

from concrete import fhe
from concrete.ml.pandas._utils import deserialize_evaluation_keys, serialize_evaluation_keys

script_dir = Path(__file__).parent

CLIENT_SERVER_DIR = script_dir / "_client_server_files"
CLIENT_PATH = CLIENT_SERVER_DIR / "client.zip"
SERVER_PATH = CLIENT_SERVER_DIR / "server.zip"

N_BITS_PANDAS = 4


def identity_pbs(a):
    """Define an identity TLU."""
    return fhe.univariate(lambda x: x)(a)


@fhe.compiler(
    {"val_1": "encrypted", "val_2": "encrypted", "left_key": "encrypted", "right_key": "encrypted"}
)
def left_right_join_to_compile(val_1, val_2, left_key, right_key):
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

    sum_with_tlu = identity_pbs(sum_on_condition)

    return sum_with_tlu


def get_left_right_join_max_value(n_bits: int):
    return 2**n_bits - 1


def get_left_right_join_inputset(n_bits: int):
    """Define the inputset to use for the merge operator."""
    # Build the circuit using at most `n_bits` bits, which defines :
    # - the input's unsigned integer dtype allowed (at most)
    # - the maximum number of rows allowed in an input (assuming the merge is done on a column of
    # unsigned integers starting at 1)
    # Note that a 0 is used to represent a NaN (Not a Number) value
    max_row_allowed = high = get_left_right_join_max_value(n_bits)

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
PANDAS_OPS_TO_CIRCUIT_CONFIG = {
    "left_right_join": {
        "get_inputset": partial(get_left_right_join_inputset, n_bits=N_BITS_PANDAS),
        "to_compile": left_right_join_to_compile,
        "encrypt_config": {
            "n": 4,
            "pos": 1,
        },
    }
}


def get_encrypt_config():
    return PANDAS_OPS_TO_CIRCUIT_CONFIG["left_right_join"]["encrypt_config"]


def get_min_max_allowed():
    return (1, get_left_right_join_max_value(N_BITS_PANDAS))


def save_client_server(client_path=CLIENT_PATH, server_path=SERVER_PATH):
    client_path, server_path = Path(client_path), Path(server_path)

    if not client_path.is_file() or not server_path.is_file():
        client_path.parent.mkdir(parents=True, exist_ok=True)
        server_path.parent.mkdir(parents=True, exist_ok=True)

        config = PANDAS_OPS_TO_CIRCUIT_CONFIG["left_right_join"]

        inputset = config["get_inputset"]()
        cp_func = config["to_compile"]

        # Compile with composability
        merge_circuit = cp_func.compile(inputset, composable=True)

        # Save the client and server files
        merge_circuit.client.save(client_path)
        merge_circuit.server.save(server_path, via_mlir=True)


def load_server():
    return fhe.Server.load(SERVER_PATH)


def load_client(keygen=True, keys_path=None):
    client = fhe.Client.load(CLIENT_PATH)

    if keygen:
        if keys_path is not None:
            client.keys.load_if_exists_generate_and_save_otherwise(keys_path)
        else:
            client.keygen(True)

    return client
