"""Define development methods for generating client/server files."""

import itertools
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy
from concrete.fhe import Configuration
from concrete.fhe.compilation.module import FheModule
from concrete.fhe.tracing import Tracer

from concrete import fhe

from  ..quantization.quantized_module import _get_inputset_generator

script_dir = Path(__file__).parent

CURRENT_API_VERSION = 2

API_VERSION_SPECS = {
    1: {"configuration": Configuration(), "join_function": "main"},
    2: {
        "configuration": Configuration(
            compress_evaluation_keys=True, compress_input_ciphertexts=True,
        ),
        "join_function": "left_right_join_to_compile",
        "batch_1d_function": "build_batch_1d",
        "batch_2d_function": "build_batch_2d",
        "train_log_reg_function": "train_log_reg",
        "create_batch_2d": "create_batch_2d",
        "create_batch_1d": "create_batch_1d"
    },
}

# The paths where to find and save the client/server files
CLIENT_SERVER_DIR = script_dir / "_client_server_files" / f"api_{CURRENT_API_VERSION}"
CLIENT_PATH = CLIENT_SERVER_DIR / "client.zip"
SERVER_PATH = CLIENT_SERVER_DIR / "server.zip"

N_BITS_PANDAS = 4

from ..common.utils import generate_proxy_function
from ..common._fhe_training_utils import LogisticRegressionTraining, make_training_inputset
from ..torch.compile import build_quantized_module
from concrete.fhe import Wired, Wire, Output, Input, AllInputs, AllOutputs

class DFApiV2StaticHelper:
    N_DIMS_TRAINING = 16
    BATCH_SIZE = 8

    _training_input_set = make_training_inputset(
        numpy.ones((N_DIMS_TRAINING,), dtype=numpy.float64) * -1.0,
        numpy.ones((N_DIMS_TRAINING,), dtype=numpy.float64) * 1.0,
        0,
        2**N_BITS_PANDAS - 1,
        BATCH_SIZE,
        True,
    )


def create_api_v2():
    class DFApiV2Helper:
        # Build the quantized module
        _training_module = build_quantized_module(
            model=LogisticRegressionTraining(
                learning_rate=1,
                iterations=1,
                fit_bias=False,
            ),
            torch_inputset=DFApiV2StaticHelper._training_input_set,
            import_qat=False,
            n_bits=N_BITS_PANDAS,
            rounding_threshold_bits={"n_bits": 6, "method": fhe.Exactness.EXACT},
        )

        _forward_proxy, _orig_args_to_proxy_func_args = generate_proxy_function(
            _training_module._clear_forward, _training_module.ordered_module_input_names
        )

    @fhe.module()
    class DFApiV2:
        @fhe.function(
            {
                "features": "encrypted",
                "targets": "encrypted",
                "weights": "encrypted",
                "bias": "encrypted",
            }
        )
        def train_log_reg(
            features: Union[Tracer, int],
            targets: Union[Tracer, int],
            weights: Union[Tracer, int],
            bias: Union[Tracer, int],
        ):
            return DFApiV2Helper._forward_proxy(features, targets, weights, bias)

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

        @fhe.function({"value": "encrypted"})
        def create_batch_2d(value):
            batch = fhe.zeros((DFApiV2StaticHelper.BATCH_SIZE, DFApiV2StaticHelper.N_DIMS_TRAINING))
            batch[0,0] = fhe.refresh(value)
            return batch

        @fhe.function({"value": "encrypted"})
        def create_batch_1d(value):
            batch = fhe.zeros((DFApiV2StaticHelper.BATCH_SIZE, ))
            batch[0] = fhe.refresh(value)
            return batch

        @fhe.function(
            {
                "batch": "encrypted",
                "value": "encrypted",
                "index1": "clear",
                "index2": "clear",
            }
        )
        def build_batch_2d(
            batch: Union[Tracer, int],
            value: Union[Tracer, int],
            index1: Union[Tracer, int],
            index2: Union[Tracer, int],
        ):
            batch[index1, index2] = fhe.refresh(value)
            return batch

        @fhe.function(
            {
                "batch": "encrypted",
                "value": "encrypted",
                "index1": "clear",
            }
        )
        def build_batch_1d(
            batch: Union[Tracer, int],
            value: Union[Tracer, int],
            index1: Union[Tracer, int],
        ):
            batch[index1] = fhe.refresh(value)
            return batch

        composition = Wired(
            [
                # Compose every input -> output of the join function
                Wire(AllOutputs(left_right_join_to_compile), AllInputs(left_right_join_to_compile)),

                # The output of the join function is used to build the training batch or the labels batch
                Wire(Output(left_right_join_to_compile, 0), Input(build_batch_2d, 1)),
                Wire(Output(left_right_join_to_compile, 0), Input(build_batch_1d, 1)),

                # Batch creation
                Wire(Output(create_batch_2d, 0), Input(build_batch_2d, 0)),
                Wire(Output(create_batch_1d, 0), Input(build_batch_1d, 0)),

                # Batch building is composable
                Wire(Output(build_batch_2d, 0), Input(build_batch_2d, 0)),
                Wire(Output(build_batch_1d, 0), Input(build_batch_1d, 0)),

                # Batches of training data and labels are inputs to log reg training
                Wire(Output(build_batch_2d, 0), Input(train_log_reg, 0)),
                Wire(Output(build_batch_1d, 0), Input(train_log_reg, 1)),

                Wire(Output(train_log_reg, 0), Input(train_log_reg, 2)),
                Wire(Output(train_log_reg, 1), Input(train_log_reg, 3)),
            ]
        )

    return DFApiV2


def identity_pbs(value: Union[Tracer, int]) -> Union[Tracer, int]:
    """Define an identity TLU.

    Args:
        value (Union[Tracer, int]): The value on which to apply the identity.

    Returns:
        Union[Tracer, int]: The input value.
    """
    return fhe.univariate(lambda x: x)(value)


def create_api_v1():
    @fhe.compiler(
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

    return left_right_join_to_compile


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


def get_training_inputset():
    return list(
        _get_inputset_generator(
            tuple(map(lambda x: x.astype(numpy.int64), DFApiV2StaticHelper._training_input_set))
        )
    )

def get_batch_build_dataset_2d():
    batch_min = numpy.zeros((DFApiV2StaticHelper.BATCH_SIZE, DFApiV2StaticHelper.N_DIMS_TRAINING), dtype=numpy.uint64)
    batch_max = numpy.ones((DFApiV2StaticHelper.BATCH_SIZE, DFApiV2StaticHelper.N_DIMS_TRAINING), dtype=numpy.uint64) * (2 ** N_BITS_PANDAS - 1)
    value_min = 0
    value_max = (2 ** N_BITS_PANDAS - 1)
    index_min = 0
    index_max = DFApiV2StaticHelper.BATCH_SIZE - 1
    return [
        (batch_min, value_max, index_min, index_max), 
        (batch_max, value_min, index_max, index_min), 
        (batch_min, value_min, index_min, index_max), 
        (batch_max, value_max, index_max, index_min)
    ]

def get_batch_build_dataset_1d():
    batch_min = numpy.zeros((DFApiV2StaticHelper.BATCH_SIZE, ), dtype=numpy.uint64)
    batch_max = numpy.ones((DFApiV2StaticHelper.BATCH_SIZE, ), dtype=numpy.uint64) * (2 ** N_BITS_PANDAS - 1)
    value_min = 0
    value_max = (2 ** N_BITS_PANDAS - 1)
    index_min = 0
    index_max = DFApiV2StaticHelper.BATCH_SIZE - 1
    return [
        (batch_min, value_max, index_min), 
        (batch_max, value_min, index_max), 
        (batch_min, value_min, index_min), 
        (batch_max, value_max, index_max)
    ]

def get_batch_create_dataset():
    value_min = 0
    value_max = (2 ** N_BITS_PANDAS - 1)
    return [
        (value_max,), 
        (value_min,), 
    ]

# Store the configuration functions and parameters to their associated operator
PANDAS_OPS_TO_CIRCUIT_CONFIG = {
    1: {
        "get_inputset": partial(get_left_right_join_inputset, n_bits=N_BITS_PANDAS),
        "to_compile": create_api_v1,
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
            "train_log_reg": get_training_inputset,
            "build_batch_2d": get_batch_build_dataset_2d,
            "build_batch_1d": get_batch_build_dataset_1d,
            "create_batch_1d": get_batch_create_dataset,
            "create_batch_2d": get_batch_create_dataset,
        },
        "to_compile": create_api_v2,
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
        server_path (Path): The path where to save the client file. Default to SERVER_PATH.
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

    cp_func = config["to_compile"]()

    # Configuration used for this API version
    cfg = API_VERSION_SPECS[CURRENT_API_VERSION]["configuration"]

    # Compile the circuit and allow it to be composable with itself
    merge_circuit = cp_func.compile(inputset, composable=True, configuration=cfg)

    # Save the client and server files using the MLIR
    if isinstance(merge_circuit, FheModule):
        merge_circuit.runtime.server.save(server_path, via_mlir=True)
        merge_circuit.runtime.client.save(client_path)
    else:
        merge_circuit.server.save(server_path, via_mlir=True)
        merge_circuit.client.save(client_path)


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
