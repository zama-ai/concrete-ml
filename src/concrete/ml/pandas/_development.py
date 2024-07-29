"""Define development methods for generating client/server files."""

import itertools
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple, Union

from concrete.fhe import Configuration
from concrete.fhe.tracing import Tracer

from concrete import fhe

script_dir = Path(__file__).parent

# The paths where to find and save the client/server files
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

    This function is going to be composed with itself as part of the encrypted merge algorithm,
    which is explained in the '_operators.py' file. Here, the function takes two keys and two
    values, all encrypted:
        * left_key and right_key, one for each data-frame
        * val_1, which will ultimately become the value to insert in the output data-frame once the
            composition loop is done. It is therefore either representing a 0 (most of the time) or
            a value from one of the input data-frame (only once during the composition loop).
        * val_2, the value to add to val_1 if both keys match. As said just above, this value
            should actually only be added once during the composition loop, as the keys are expected
            to be unique in both data-frames

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

    # Adding an identity TLU is necessary here, else the function won't compile in FHE
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
    # Build the circuit using at most 'n_bits' bits. This value defines :
    # - the maximum integer value allowed in the all data-frames
    # - the maximum number of rows allowed in all data-frames, assuming that the column on which to
    # merge contains unique integers that start at value 1
    high = get_left_right_join_max_value(n_bits)

    # Note that any column can include NaN values, which are currently represented by 0. This means
    # the input-set needs to consider 0 although pre-processing requires data-frame to provide
    # integers values greater or equal to 1
    inputset = list(itertools.product([0, high], [0, high], [0, high], [0, high]))

    return inputset


# Store the configuration functions and parameters to their associated operator
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


# Allow 0 values once NaN values are not represented by it anymore
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4342
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

    client_path.parent.mkdir(parents=True, exist_ok=True)
    server_path.parent.mkdir(parents=True, exist_ok=True)

    config = PANDAS_OPS_TO_CIRCUIT_CONFIG["left_right_join"]

    # Get the input-set and circuit generating functions
    inputset = config["get_inputset"]()
    cp_func = config["to_compile"]
    compilation_configuration = Configuration(compress_evaluation_keys=True)

    # Compile the circuit and allow it to be composable with itself
    merge_circuit = cp_func.compile(
        inputset, composable=True, configuration=compilation_configuration
    )

    # Save the client and server files using the MLIR
    merge_circuit.client.save(client_path)
    merge_circuit.server.save(server_path, via_mlir=True)


def load_server() -> fhe.Server:
    """Load the server to use for executing operators on encrypted data-frames.

    Returns:
        fhe.Server: The loaded server.
    """
    return fhe.Server.load(SERVER_PATH)


# This part is used for updating the files when needed (for example, when Concrete Python is updated
# and some backward compatibility issues arise)
if __name__ == "__main__":
    save_client_server()  # pragma: no cover
