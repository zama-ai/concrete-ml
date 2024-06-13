"""Define development methods for generating client/server files."""

import itertools
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy
from concrete.fhe import Configuration
from concrete.fhe.tracing import Tracer

from concrete import fhe

from ..quantization.quantized_module import _get_inputset_generator

script_dir = Path(__file__).parent


CURRENT_API_VERSION = 2

API_VERSION_SPECS = {
    1: {"configuration": Configuration()},
    2: {
        "configuration": Configuration(
            compress_input_ciphertexts=True, compress_evaluation_keys=True
        )
    },
}

# The paths where to find and save the client/server files
CLIENT_SERVER_DIR = script_dir / "_client_server_files" / f"api_{CURRENT_API_VERSION}"
CLIENT_PATH = CLIENT_SERVER_DIR / "client.zip"
SERVER_PATH = CLIENT_SERVER_DIR / "server.zip"

N_BITS_PANDAS = 4

from ..common.utils import generate_proxy_function
from ..sklearn._fhe_training_utils import LogisticRegressionTraining, make_training_inputset
from ..torch.compile import build_quantized_module


class DFApiV2Helper:
    _N_DIMS_TRAINING = 1

    # Expect data and weight values to be between -1 and 1
    _training_calibration_set = make_training_inputset(
        x_min=numpy.ones((_N_DIMS_TRAINING,), dtype=numpy.float32) * (-1),
        x_max=numpy.ones((_N_DIMS_TRAINING,), dtype=numpy.float32),
        param_min=-1,
        param_max=1,
        batch_size=8,
        fit_intercept=True,
    )

    # Build the quantized module
    _training_module = build_quantized_module(
        model=LogisticRegressionTraining(
            learning_rate=1,
            iterations=1,
            fit_bias=False,
        ),
        torch_inputset=_training_calibration_set,
        import_qat=False,
        n_bits=N_BITS_PANDAS,
        rounding_threshold_bits=6,
    )

    _forward_proxy, _orig_args_to_proxy_func_args = generate_proxy_function(
        _training_module._clear_forward, _training_module.ordered_module_input_names
    )

    _training_q_input_set = _training_module.quantize_input(*_training_calibration_set)


@fhe.module()
class DFApiV2:

    @fhe.function(
        {
            "val_1": "encrypted",
            "val_2": "encrypted",
            "left_key": "encrypted",
            "right_key": "encrypted",
        }
    )
    def left_right_join_to_compile(
        val_1: Union[Tracer, int],
        val_2: Union[Tracer, int],
        left_key: Union[Tracer, int],
        right_key: Union[Tracer, int],
    ) -> Union[Tracer, int]:
        return _left_right_join_to_compile_internal(val_1, val_2, left_key, right_key)

    @fhe.function(
        {
            "a": "encrypted",
            "b": "encrypted",
            "c": "encrypted",
            "d": "encrypted",
            "e": "encrypted",
            "f": "encrypted",
            "g": "encrypted",
            "h": "encrypted",
        }
    )
    def to_encrypted_tensor(a, b, c, d, e, f, g, h):
        output = fhe.array([a, b, c, d, e, f, g, h])
        return output

    @fhe.function({"x": "encrypted", "y": "encrypted", "weights": "encrypted", "bias": "encrypted"})
    def train_log_reg(
        x: Union[Tracer, int],
        y: Union[Tracer, int],
        weights: Union[Tracer, int],
        bias: Union[Tracer, int],
    ):
        return DFApiV2Helper._forward_proxy(x, y, weights, bias)

    composition = fhe.Wired(
        [
            fhe.Wire(
                fhe.AllOutputs(left_right_join_to_compile), fhe.AllInputs(to_encrypted_tensor)
            ),
            fhe.Wire(fhe.AllOutputs(to_encrypted_tensor), fhe.Input(train_log_reg, 0)),
            fhe.Wire(fhe.AllOutputs(to_encrypted_tensor), fhe.Input(train_log_reg, 1)),
        ]
    )


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
    """Runs the atomic left/right join in FHE.
    Args:
        val_1 (Union[Tracer, int]): The value used for accumulating the sum.
        val_2 (Union[Tracer, int]): The value to add if the keys match.
        left_key (Union[Tracer, int]): The left data-frame's encrypted key to consider.
        right_key (Union[Tracer, int]): The right data-frame's encrypted key to consider.

    Returns:
        Union[Tracer, int]): The new accumulated sum.
    """
    return _left_right_join_to_compile_internal(val_1, val_2, left_key, right_key)


def _left_right_join_to_compile_internal(
    val_1: Union[Tracer, int],
    val_2: Union[Tracer, int],
    left_key: Union[Tracer, int],
    right_key: Union[Tracer, int],
) -> Union[Tracer, int]:
    """Runs the atomic left/right join in FHE.

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


def get_to_encrypted_tensor_inputset(n_bits, n_inputs):
    high = get_left_right_join_max_value(n_bits)

    min_max_values = tuple([[0, high]] * n_inputs)

    inputset = list(itertools.product(*min_max_values))
    return inputset


def get_training_q_inputset():
    return _get_inputset_generator(DFApiV2Helper._training_q_input_set)


# Store the configuration functions and parameters to their associated operator
PANDAS_OPS_TO_CIRCUIT_CONFIG = {
    1: {
        "get_inputset": partial(get_left_right_join_inputset, n_bits=N_BITS_PANDAS),
        "to_compile": left_right_join_to_compile,
        "encrypt_config": {
            "n": 4,
            "pos": 1,
        },
    },
    2: {
        "get_inputset": {
            "left_right_join_to_compile": partial(
                get_left_right_join_inputset, n_bits=N_BITS_PANDAS
            ),
            "to_encrypted_tensor": partial(
                get_to_encrypted_tensor_inputset, n_bits=N_BITS_PANDAS, n_inputs=8
            ),
            "train_log_reg": get_training_q_inputset,
        },
        "to_compile": DFApiV2,
        "encrypt_config": {
            "n": 4,
            "pos": 1,
        },
    },
}


def get_encrypt_config() -> Dict:
    """Get the configuration parameters to use when encrypting the input values.

    Configuration parameters for encryption include the total number of inputs used in the FHE
    circuit as well as the input position to consider when encrypting.

    Returns:
        Dict: The configuration parameters for encryption.
    """
    return PANDAS_OPS_TO_CIRCUIT_CONFIG[CURRENT_API_VERSION]["encrypt_config"]


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
        server_path (Path): The path where to save the server file. Default to SERVER_PATH.
    """
    client_path, server_path = Path(client_path), Path(server_path)

    client_path.parent.mkdir(parents=True, exist_ok=True)
    server_path.parent.mkdir(parents=True, exist_ok=True)

    config = PANDAS_OPS_TO_CIRCUIT_CONFIG[CURRENT_API_VERSION]

    # Get the input-set and circuit generating functions
    if isinstance(config["get_inputset"], dict):
        inputset = {func: config["get_inputset"][func]() for func in config["get_inputset"].keys()}
    else:
        inputset = config["get_inputset"]()

    cp_func = config["to_compile"]
    cp_func.dump_artifacts_on_unexpected_failures = False

    # Configuration used for this API version
    configuration = API_VERSION_SPECS[CURRENT_API_VERSION]["configuration"]

    # Compile the circuit and allow it to be composable with itself
    merge_circuit = cp_func.compile(inputset, configuration=configuration)

    # Save the client and server files using the MLIR
    merge_circuit.client.save(client_path)
    merge_circuit.server.save(server_path, via_mlir=True)


def load_server() -> fhe.Server:
    """Load the server to use for executing operators on encrypted data-frames.

    Returns:
        fhe.Server: The loaded server.
    """
    return fhe.Server.load(SERVER_PATH)
