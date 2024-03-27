import itertools
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from concrete.fhe.tracing import Tracer

from concrete import fhe

script_dir = Path(__file__).parent

# The paths where to find or save the client and server files
CLIENT_SERVER_DIR = script_dir / "_client_server_files"
CLIENT_PATH = CLIENT_SERVER_DIR / "client.zip"
SERVER_PATH = CLIENT_SERVER_DIR / "server.zip"

N_BITS_PANDAS = 4


def identity_pbs(value: Union[Tracer, int]) -> Union[Tracer, int]:
    """Define an identity TLU.

    Args:
        value (Union[Tracer, int]): The value on which to apply the identity.

    Returns:
        Union[Tracer, int]: The input value.
    """
    return fhe.univariate(lambda x: x)(value)


@fhe.compiler(
    {"val_1": "encrypted", "val_2": "encrypted", "left_key": "encrypted", "right_key": "encrypted"}
)
def left_right_join_to_compile(
    val_1: Union[Tracer, int],
    val_2: Union[Tracer, int],
    left_key: Union[Tracer, int],
    right_key: Union[Tracer, int],
) -> Union[Tracer, int]:
    """Define the atomic function to consider for running a left/right join in FHE.

    TODO: explain algo

    Args:
        val_1 (Union[Tracer, int]): The value used for accumulating the sum.
        val_2 (Union[Tracer, int]): The value to add if the keys match.
        left_key (Union[Tracer, int]): The left data-frame's encrypted key to consider.
        right_key (Union[Tracer, int]): The right data-frame's encrypted key to consider.

    Returns:
        Union[Tracer, int]): The new accumulated sum.
    """
    condition = left_key == right_key

    sum_on_condition = val_1 + (val_2 * condition)

    sum_with_tlu = identity_pbs(sum_on_condition)

    return sum_with_tlu


def get_left_right_join_max_value(n_bits: int) -> int:
    """Get the maximum value allowed in the data-frames for the left/right join operator.

    Args:
        n_bits (int): The maximum number of bits allowed.

    Returns:
        int: The maximum value allowed.
    """
    return 2**n_bits - 1


def get_left_right_join_inputset(n_bits: int) -> List:
    """Generate the input-set to use for compiling the left/right join operator.

    Args:
        n_bits (int): The maximum number of bits allowed for generating the input-set's values.

    Returns:
        List: The input-set.
    """
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


def get_encrypt_config() -> Dict:
    """Get the configuration parameters to use when encrypting the input values.

    Configuration parameters for encryption include the total number of inputs used in the FHE
    circuit as well as the input position to consider when encrypting.

    Returns:
        Dict: The configuration parameters for encryption.
    """
    return PANDAS_OPS_TO_CIRCUIT_CONFIG["left_right_join"]["encrypt_config"]


def get_min_max_allowed() -> Tuple[int, int]:
    """Get the minimum and maximum value allowed in the data-frames.

    Returns:
        Tuple[int, int]: The minimum and maximum value allowed.
    """
    return (1, get_left_right_join_max_value(N_BITS_PANDAS))


def save_client_server(client_path: Path = CLIENT_PATH, server_path: Path = SERVER_PATH):
    """Build the FHE circuit for all supported operators and save the client/server files.

    Note that this function is not made public as the files are built and saved only once directly
    in the source.

    Args:
        client_path (Path): The path where to save the client file. Default to CLIENT_PATH.
        server_path (Path): The path where to save the client file. Default to SERVER_PATH.
    """
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


def load_server() -> fhe.Server:
    """Load the server to use for executing operators on encrypted data-frames.

    Returns:
        fhe.Server: The loaded server.
    """
    return fhe.Server.load(SERVER_PATH)
