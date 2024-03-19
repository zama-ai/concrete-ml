import itertools
from functools import partial
from pathlib import Path

from concrete import fhe

from ._utils import deserialize_evaluation_keys, serialize_evaluation_keys

_script_dir = Path(__file__).parent

_CLIENTS_DIR = _script_dir / "client_files"
_SERVERS_DIR = _script_dir / "server_files"

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
        "min": 1,
        "max": _get_left_right_join_max_value(_N_BITS_PANDAS),
    }
}

SUPPORTED_PANDAS_OPS_AND_KWARGS = {"merge": {"how": ["left", "right"]}}


def _get_encrypt_config(operator):
    return _PANDAS_OPS_TO_CIRCUIT_CONFIG[operator]["encrypt_config"]


def _get_min_max_allowed(operator):
    return (
        _PANDAS_OPS_TO_CIRCUIT_CONFIG[operator]["min"],
        _PANDAS_OPS_TO_CIRCUIT_CONFIG[operator]["max"],
    )


def _save_client_server(operator, config, clients_dir, servers_dir, force_save):
    client_path = clients_dir / operator / "client.zip"
    server_path = servers_dir / operator / "server.zip"

    if force_save or not client_path.is_file() or not server_path.is_file():
        client_path.parent.mkdir(parents=True, exist_ok=True)
        server_path.parent.mkdir(parents=True, exist_ok=True)

        inputset = config["get_inputset"]()
        cp_func = config["to_compile"]

        # Compile with composability
        merge_circuit = cp_func.compile(inputset, composable=True)

        # Save the client and server files
        merge_circuit.server.save(server_path, via_mlir=True)
        merge_circuit.client.save(client_path)


def save_clients_servers(clients_dir=_CLIENTS_DIR, servers_dir=_SERVERS_DIR, force_save=True):
    for operator, config in _PANDAS_OPS_TO_CIRCUIT_CONFIG.items():
        _save_client_server(operator, config, clients_dir, servers_dir, force_save)


def load_client(operator, clients_dir=_CLIENTS_DIR, force_keygen=True):
    client_path = clients_dir / operator / "client.zip"
    client = fhe.Client.load(client_path)

    client.keygen(force_keygen)

    return client


def get_client_and_eval_keys(operator, clients_dir=_CLIENTS_DIR, force_keygen=True):
    client = load_client(operator, clients_dir=clients_dir, force_keygen=force_keygen)

    return client, client.evaluation_keys


def load_server(operator, servers_dir=_SERVERS_DIR):
    server_path = servers_dir / operator / "server.zip"
    server = fhe.Server.load(server_path)
    return server


def load_clients(clients_dir=_CLIENTS_DIR, force_keygen=True):
    clients = {}

    for operator in _PANDAS_OPS_TO_CIRCUIT_CONFIG:
        client = load_client(operator, clients_dir, keygen=force_keygen)
        clients[operator] = client

    return clients


def get_clients_and_eval_keys(clients_dir=_CLIENTS_DIR, force_keygen=True):
    clients = {}

    for operator in _PANDAS_OPS_TO_CIRCUIT_CONFIG:
        client, eval_keys = get_client_and_eval_keys(
            operator, clients_dir=clients_dir, force_keygen=force_keygen
        )
        clients[operator]["client"] = client
        clients[operator]["evaluation_keys"] = eval_keys

    return clients


def load_servers(servers_dir=_SERVERS_DIR, keygen=True):
    servers = {}

    for operator in _PANDAS_OPS_TO_CIRCUIT_CONFIG:
        server = load_server(operator, servers_dir, keygen=keygen)
        servers[operator] = server

    return servers


def load_eval_keys(file_path):
    file_path = Path(file_path)
    with file_path.open("rb") as file:
        serialized_evaluation_keys = file.read()

    evaluation_keys = deserialize_evaluation_keys(serialized_evaluation_keys)
    return evaluation_keys


def dump_eval_keys(evaluation_keys, file_path):
    serialized_evaluation_keys = serialize_evaluation_keys(evaluation_keys)

    file_path = Path(file_path)
    with file_path.open("wb") as file:
        file.write(serialized_evaluation_keys)
