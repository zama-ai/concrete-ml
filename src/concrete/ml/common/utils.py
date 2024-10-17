"""Utils that can be re-used by other pieces of code in the module."""

import enum
import os
import string
from functools import partial
from types import FunctionType
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy
import onnx
import torch
from concrete.compiler import check_gpu_available, check_gpu_enabled
from concrete.fhe import Exactness
from concrete.fhe.dtypes import Integer
from sklearn.base import is_classifier, is_regressor

from ..common.check_inputs import check_array_and_assert
from ..common.debugging import assert_true

_VALID_ARG_CHARS = set(string.ascii_letters).union(str(i) for i in range(10)).union(("_",))


SUPPORTED_FLOAT_TYPES = {
    "float64": torch.float64,
    "float32": torch.float32,
}

SUPPORTED_INT_TYPES = {
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
}

SUPPORTED_TYPES = {**SUPPORTED_FLOAT_TYPES, **SUPPORTED_INT_TYPES}

SUPPORTED_DEVICES = ["cuda", "cpu"]

MAX_BITWIDTH_BACKWARD_COMPATIBLE = 8

# Indicate if the old virtual library method should be used instead of the compiler simulation
# when simulating FHE executions
USE_OLD_VL = False

# Debug option for testing round PBS optimization
# Setting this option to true will make quantizers "round half up"
# For example: 0.5 -> 1, 1.5 -> 2 instead of "round half to even"
# When the option is set to false, Concrete ML uses numpy.rint
# which has the same behavior as torch.round -> Brevitas nets
# should be exact compared to their Concrete ML QuantizedModule
QUANT_ROUND_LIKE_ROUND_PBS = False

# Enable input ciphertext compression
# Note: This setting is fixed and cannot be altered by users
# However, for internal testing purposes, we retain the capability to disable this feature
os.environ["USE_INPUT_COMPRESSION"] = os.environ.get("USE_INPUT_COMPRESSION", "1")

# Enable PBS evaluation key compression (~4x size reduction)
# Note: This setting is fixed and cannot be altered by users
# However, for internal testing purposes, we retain the capability to disable this feature
os.environ["USE_KEY_COMPRESSION"] = os.environ.get("USE_KEY_COMPRESSION", "1")


class FheMode(str, enum.Enum):
    """Enum representing the execution mode.

    This enum inherits from str in order to be able to easily compare a string parameter to its
    equivalent Enum attribute.

    Examples:
        fhe_disable = FheMode.DISABLE

        >>> fhe_disable == "disable"
        True

        >>> fhe_disable == "execute"
        False

        >>> FheMode.is_valid("simulate")
        True

        >>> FheMode.is_valid(FheMode.EXECUTE)
        True

        >>> FheMode.is_valid("predict_in_fhe")
        False

    """

    DISABLE = "disable"
    SIMULATE = "simulate"
    EXECUTE = "execute"

    @staticmethod
    def is_valid(fhe: Union["FheMode", str]) -> bool:
        """Indicate if the given name is a supported FHE mode.

        Args:
            fhe (Union[FheMode, str]): The FHE mode to check.

        Returns:
            bool: Whether the FHE mode is supported or not.
        """
        return fhe in FheMode.__members__.values()


class HybridFHEMode(enum.Enum):
    """Simple enum for different modes of execution of HybridModel."""

    DISABLE = "disable"  # Use torch weights
    REMOTE = "remote"  # Use remote FHE server
    SIMULATE = "simulate"  # Use FHE simulation
    CALIBRATE = "calibrate"  # Use calibration (to run before FHE compilation)
    EXECUTE = "execute"  # Use FHE execution
    TORCH = "torch"  # Use torch layers


def replace_invalid_arg_name_chars(arg_name: str) -> str:
    """Sanitize arg_name, replacing invalid chars by _.

    This does not check that the starting character of arg_name is valid.

    Args:
        arg_name (str): the arg name to sanitize.

    Returns:
        str: the sanitized arg name, with only chars in _VALID_ARG_CHARS.
    """
    arg_name_as_chars = list(arg_name)
    for idx, char in enumerate(arg_name_as_chars):
        if char not in _VALID_ARG_CHARS:
            arg_name_as_chars[idx] = "_"

    return "".join(arg_name_as_chars)


def generate_proxy_function(
    function_to_proxy: Callable,
    desired_functions_arg_names: Iterable[str],
) -> Tuple[Callable, Dict[str, str]]:
    r"""Generate a proxy function for a function accepting only \*args type arguments.

    This returns a runtime compiled function with the sanitized argument names passed in
    desired_functions_arg_names as the arguments to the function.

    Args:
        function_to_proxy (Callable): the function defined like def f(\*args) for which to return a
            function like f_proxy(arg_1, arg_2) for any number of arguments.
        desired_functions_arg_names (Iterable[str]): the argument names to use, these names are
            sanitized and the mapping between the original argument name to the sanitized one is
            returned in a dictionary. Only the sanitized names will work for a call to the proxy
            function.

    Returns:
        Tuple[Callable, Dict[str, str]]: the proxy function and the mapping of the original arg name
            to the new and sanitized arg names.
    """
    # Some input names can be invalid arg names (e.g., coming from torch input.0) so sanitize them
    # to be valid Python arg names.
    orig_args_to_proxy_func_args = {
        arg_name: f"_{replace_invalid_arg_name_chars(arg_name)}"
        for arg_name in desired_functions_arg_names
    }
    proxy_func_arg_string = ", ".join(orig_args_to_proxy_func_args.values())
    proxy_func_name = replace_invalid_arg_name_chars(f"{function_to_proxy.__name__}_proxy")
    # compile is the built-in Python compile to generate code at runtime.
    function_proxy_code = compile(
        f"def {proxy_func_name}({proxy_func_arg_string}): "
        f"return function_to_proxy({proxy_func_arg_string})",
        __file__,
        mode="exec",
    )
    function_proxy = FunctionType(function_proxy_code.co_consts[0], locals(), proxy_func_name)

    return function_proxy, orig_args_to_proxy_func_args


def get_onnx_opset_version(onnx_model: onnx.ModelProto) -> int:
    """Return the ONNX opset_version.

    Args:
        onnx_model (onnx.ModelProto): the model.

    Returns:
        int: the version of the model
    """

    info = onnx_model.opset_import[0]
    assert_true(info.domain == "", "onnx version information is not as expected")
    return info.version


def manage_parameters_for_pbs_errors(
    p_error: Optional[float] = None,
    global_p_error: Optional[float] = None,
):
    """Return (p_error, global_p_error) that we want to give to Concrete.

    The returned (p_error, global_p_error) depends on user's parameters and the way we want to
    manage defaults in Concrete ML, which may be different from the way defaults are managed in
    Concrete.

    Principle:
        - if none are set, we set global_p_error to a default value of our choice
        - if both are set, we raise an error
        - if one is set, we use it and forward it to Concrete

    Note that global_p_error is currently set to 0 in the FHE simulation mode.

    Args:
        p_error (Optional[float]): probability of error of a single PBS.
        global_p_error (Optional[float]): probability of error of the full circuit.

    Returns:
        (p_error, global_p_error): parameters to give to the compiler

    Raises:
        ValueError: if the two parameters are set (this is _not_ as in Concrete-Python)

    """
    # Default probability of error of a circuit. Only used if p_error is set to None
    default_p_error_pbs = 2**-40

    if (p_error, global_p_error) == (None, None):
        p_error, global_p_error = (default_p_error_pbs, None)
    elif p_error is None:
        # Nothing to do, use user's parameters
        pass
    elif global_p_error is None:
        # Nothing to do, use user's parameters
        pass
    else:
        raise ValueError("Please only set one of (p_error, global_p_error) values")

    return p_error, global_p_error


def check_there_is_no_p_error_options_in_configuration(configuration):
    """Check the user did not set p_error or global_p_error in configuration.

    It would be dangerous, since we set them in direct arguments in our calls to Concrete-Python.

    Args:
        configuration: Configuration object to use
            during compilation

    """
    if configuration is not None:
        assert_true(
            configuration.p_error is None,
            "Don't set p_error in configuration, use kwargs",
        )
        assert_true(
            configuration.global_p_error is None,
            "Don't set global_p_error in configuration, use kwargs",
        )


def get_model_class(model_class):
    """Return the class of the model (instantiated or not), which can be a partial() instance.

    Args:
        model_class: The model, which can be a partial() instance.

    Returns:
        The model's class.

    """
    # If model_class is a functool.partial instance
    if isinstance(model_class, partial):
        return model_class.func

    # If model_class is a instantiated model
    if not isinstance(model_class, type) and hasattr(model_class, "__class__"):
        return model_class.__class__

    # Else, it is already a (not instantiated) model class
    return model_class


def is_model_class_in_a_list(model_class, a_list):
    """Indicate if a model class, which can be a partial() instance, is an element of a_list.

    Args:
        model_class: The model, which can be a partial() instance.
        a_list: The list in which to look into.

    Returns:
        If the model's class is in the list or not.

    """
    return get_model_class(model_class) in a_list


def get_model_name(model_class):
    """Return the name of the model, which can be a partial() instance.

    Args:
        model_class: The model, which can be a partial() instance.

    Returns:
        the model's name.

    """
    return get_model_class(model_class).__name__


def is_classifier_or_partial_classifier(model_class):
    """Indicate if the model class represents a classifier.

    Args:
        model_class: The model class, which can be a functool's `partial` class.

    Returns:
        bool: If the model class represents a classifier.
    """
    return is_classifier(get_model_class(model_class))


def is_regressor_or_partial_regressor(model_class):
    """Indicate if the model class represents a regressor.

    Args:
        model_class: The model class, which can be a functool's `partial` class.

    Returns:
        bool: If the model class represents a regressor.
    """
    return is_regressor(get_model_class(model_class))


def is_pandas_dataframe(input_container: Any) -> bool:
    """Indicate if the input container is a Pandas DataFrame.

    This function is inspired from Scikit-Learn's test validation tools and avoids the need to add
    and import Pandas as an additional dependency to the project.
    See https://github.com/scikit-learn/scikit-learn/blob/98cf537f5/sklearn/utils/validation.py#L629

    Args:
        input_container (Any): The input container to consider

    Returns:
        bool: If the input container is a DataFrame
    """
    return hasattr(input_container, "dtypes") and hasattr(input_container.dtypes, "__array__")


def is_pandas_series(input_container: Any) -> bool:
    """Indicate if the input container is a Pandas Series.

    This function is inspired from Scikit-Learn's test validation tools and avoids the need to add
    and import Pandas as an additional dependency to the project.
    See https://github.com/scikit-learn/scikit-learn/blob/98cf537f5/sklearn/utils/validation.py#L629

    Args:
        input_container (Any): The input container to consider

    Returns:
        bool: If the input container is a Series
    """
    return hasattr(input_container, "iloc") and hasattr(input_container, "dtype")


def is_pandas_type(input_container: Any) -> bool:
    """Indicate if the input container is a Pandas DataFrame or Series.

    Args:
        input_container (Any): The input container to consider

    Returns:
        bool: If the input container is a DataFrame orSeries
    """
    return is_pandas_dataframe(input_container) or is_pandas_series(input_container)


def _get_dtype(values: Any):
    """Get a set of values' dtype in a string format to facilitate operations between sets.

    Args:
        values (Any): The values to consider

    Returns:
        set[str]: The values' dtype(s) in string format.

    Examples:
        X_pandas = pandas.DataFrame([[1, 0.5, '1']])
        X_tensor = torch.tensor([[12, 45, 747.00]])

        >>> X_pandas.dtypes
        0      int64
        1    float64
        2     object
        dtype: object

        >>> _get_dtype(X_pandas)
        {'float64', 'int64', 'object'}

        >>> X_tensor.dtype
        torch.float32

        >>> _get_dtype(X_tensor)
        {'torch.float32'}
    """
    # Specific case: if `values` is a Dict, return all items in a string format and in a single set
    if isinstance(values, Dict):
        return set(map(str, sum(values.items(), ())))

    # If the `values` is a pandas.DataFrame, retrieve all the different dtypes found in it
    if is_pandas_dataframe(values):
        return set(map(str, values.dtypes))

    # Note that numpy.ndarray, pandas.Series and torch.Tensor objects support the dtype attribute
    return set(map(str, (values.dtype,)))


def _is_of_dtype(values: Any, valid_dtypes: Dict) -> bool:
    """Indicate if the values' dtype(s) matches the given one(s).

    Args:
        values (Any): The values to consider.
        valid_dtypes (Dict): The only dtype(s) to consider.

    Returns:
        bool: If the values' dtype(s) matches the given ones.

    Examples:
        values_types = _get_dtype(pandas.DataFrame([[1, 0.5, '1']]))
        valid_dtypes = _get_dtype(SUPPORTED_INT_TYPE)

        >>> values_types
        {'float64', 'int64', 'object'}

        >>> valid_dtypes
        {'int16', 'int32', 'int64', 'int8', 'torch.int16',
         'torch.int32', 'torch.int64', 'torch.int8'}

        >>> values_types - valid_dtypes
        {'float64', 'object'}
        # The only supported types are integers, {'float64', 'object'} are not valid in this case.

    """
    assert_true(isinstance(valid_dtypes, Dict))

    # Convert the list to numpy
    if isinstance(values, list):
        values = numpy.array(values)

    # Get the values' types in string format + set type
    values_types = _get_dtype(values)
    valid_dtypes = _get_dtype(valid_dtypes)

    # All the elements in first set that are not present in the second set
    uncommon_types = values_types - valid_dtypes

    return len(uncommon_types) == 0


# pylint: disable=unexpected-keyword-arg
def check_dtype_and_cast(values: Any, expected_dtype: str, error_information: Optional[str] = ""):
    """Convert any allowed type into an array and cast it if required.

    If values types don't match with any supported type or the expected dtype, raise a ValueError.

    Args:
        values (Any): The values to consider
        expected_dtype (str): The expected dtype, either "float32" or "int64"
        error_information (str): Additional information to put in front of the error message when
            raising a ValueError. Default to None.

    Returns:
        (Union[numpy.ndarray, torch.utils.data.dataset.Subset]): The values with proper dtype.

    Raises:
        ValueError: If the values' dtype don't match the expected one or casting is not possible.
    """
    assert_true(
        expected_dtype in ("float32", "int64"),
        f'Expected dtype parameter should either be "float32" or "int64". Got {expected_dtype}',
    )

    assert_true(
        isinstance(values, (numpy.ndarray, torch.Tensor, List)) or is_pandas_type(values),
        "Unsupported type. Expected numpy.ndarray, pandas.DataFrame, pandas.Series, list "
        f"or torch.Tensor but got {type(values)}.",
    )

    if (expected_dtype == "int64" and _is_of_dtype(values, SUPPORTED_INT_TYPES)) or (
        expected_dtype == "float32" and _is_of_dtype(values, SUPPORTED_FLOAT_TYPES)
    ):
        # If the expected dtype is an int64 and the values are integers of lower precision, we can
        # safely cast them to int64
        # If the expected dtype is a float32 and the values are float64, we chose to cast them to
        # float32 in order to give the user more flexibility. This should not have a great impact on
        # the models' performances
        values = check_array_and_assert(values, ensure_2d=False)
        values = values.astype(expected_dtype)

    else:
        # Else, if the values' dtype doesn't match the expected one, raise an error
        error_message = (
            f"{error_information} dtype does not match the expected dtype and values cannot be "
            f"properly casted using that latter. Expected dtype {expected_dtype} and got "
            f"{_get_dtype(values)} with type {type(values)}."
        )

        raise ValueError(error_message)

    return values


def compute_bits_precision(x: numpy.ndarray) -> int:
    """Compute the number of bits required to represent x.

    Args:
        x (numpy.ndarray): Integer data

    Returns:
        int: the number of bits required to represent x
    """
    return Integer.that_can_represent([x.min(), x.max()]).bit_width


def is_brevitas_model(model: torch.nn.Module) -> bool:
    """Check if a model is a Brevitas type.

    Args:
        model: PyTorch model.

    Returns:
        bool: True if `model` is a Brevitas network.

    """
    return isinstance(model, torch.nn.Module) and any(
        hasattr(module, "__class__")
        and module.__class__.__name__
        in ["QuantConv1d", "QuantConv2d", "QuantIdentity", "QuantLinear", "QuantReLU"]
        for module in model.modules()
    )


def to_tuple(x: Any) -> tuple:
    """Make the input a tuple if it is not already the case.

    Args:
        x (Any): The input to consider. It can already be an input.

    Returns:
        tuple: The input as a tuple.
    """
    # If the input is not a tuple, return a tuple of a single element
    if not isinstance(x, tuple):
        return (x,)

    return x


def all_values_are_integers(*values: Any) -> bool:
    """Indicate if all unpacked values are of a supported integer dtype.

    Args:
        *values (Any): The values to consider.

    Returns:
        bool: Whether all values are supported integers or not.

    """
    return all(_is_of_dtype(value, SUPPORTED_INT_TYPES) for value in values)


def all_values_are_floats(*values: Any) -> bool:
    """Indicate if all unpacked values are of a supported float dtype.

    Args:
        *values (Any): The values to consider.

    Returns:
        bool: Whether all values are supported floating points or not.

    """
    return all(_is_of_dtype(value, SUPPORTED_FLOAT_TYPES) for value in values)


def all_values_are_of_dtype(
    *values: Any, dtypes: Union[str, List[str]], allow_none: bool = False
) -> bool:
    """Indicate if all unpacked values are of the specified dtype(s).

    Args:
        *values (Any): The values to consider.
        dtypes (Union[str, List[str]]): The dtype(s) to consider.
        allow_none (bool): Indicate if the values can be None.

    Returns:
        bool: Whether all values are of the specified dtype(s) or not.

    """
    if isinstance(dtypes, str):
        dtypes = [dtypes]

    supported_dtypes = {}
    for dtype in dtypes:
        supported_dtype = SUPPORTED_TYPES.get(dtype, None)

        assert supported_dtype is not None, (
            f"The given dtype is not supported. Expected one of {SUPPORTED_TYPES.keys()}, "
            f"got {dtype}."
        )

        supported_dtypes[dtype] = supported_dtype

    # If the values can be None, only check the other values
    if allow_none:
        return all(
            _is_of_dtype(value, supported_dtypes) if value is not None else True for value in values
        )

    return all(_is_of_dtype(value, supported_dtypes) for value in values)


def array_allclose_and_same_shape(
    a, b, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False
) -> bool:
    """Check if two numpy arrays are equal within a tolerances and have the same shape.

    Args:
        a (numpy.ndarray): The first input array
        b (numpy.ndarray): The second input array
        rtol (float): The relative tolerance parameter
        atol (float): The absolute tolerance parameter
        equal_nan (bool): Whether to compare NaN’s as equal. If True, NaN’s in a will be considered
            equal to NaN’s in b in the output array

    Returns:
        bool: True if the arrays have the same shape and all elements are equal within the specified
            tolerances, False otherwise.
    """

    assert isinstance(a, numpy.ndarray)
    assert isinstance(b, numpy.ndarray)

    return a.shape == b.shape and numpy.allclose(a, b, rtol, atol, equal_nan)


def process_rounding_threshold_bits(rounding_threshold_bits):
    """Check and process the rounding_threshold_bits parameter.

    Args:
        rounding_threshold_bits (Union[None, int, Dict[str, Union[str, int]]]): Defines precision
            rounding for model accumulators. Accepts None, an int, or a dict.
            The dict can specify 'method' (fhe.Exactness.EXACT or fhe.Exactness.APPROXIMATE)
            and 'n_bits' ('auto' or int)

    Returns:
        Dict[str, Union[str, int]]: Processed rounding_threshold_bits dictionary.

    Raises:
        NotImplementedError: If 'auto' rounding is specified but not implemented.
        ValueError: If an invalid type or value is provided for rounding_threshold_bits.
        KeyError: If the dict contains keys other than 'n_bits' and 'method'.
    """
    n_bits_rounding: Union[None, str, int] = None
    method: Exactness = Exactness.EXACT

    # Only process if rounding_threshold_bits is not None
    if rounding_threshold_bits is not None:
        if isinstance(rounding_threshold_bits, int):
            n_bits_rounding = rounding_threshold_bits
        elif isinstance(rounding_threshold_bits, dict):
            valid_keys = {"n_bits", "method"}
            if not valid_keys.issuperset(rounding_threshold_bits.keys()):
                raise KeyError(
                    f"Invalid keys in rounding_threshold_bits. "
                    f"Allowed keys are {sorted(valid_keys)}."
                )
            n_bits_rounding = rounding_threshold_bits.get("n_bits")
            if n_bits_rounding == "auto":
                raise NotImplementedError("Automatic rounding is not implemented yet.")
            if not isinstance(n_bits_rounding, int):
                raise ValueError("n_bits must be an integer.")
            method = rounding_threshold_bits.get("method", method)
            if not isinstance(method, Exactness):
                method_str = method.upper()
                if method_str in ["EXACT", "APPROXIMATE"]:
                    method = Exactness[method_str]
                else:
                    raise ValueError(
                        f"{method_str} is not a valid method. Must be one of EXACT, APPROXIMATE."
                    )
        else:
            raise ValueError("Invalid type for rounding_threshold_bits. Must be int or dict.")

        if n_bits_rounding is not None and not 2 <= n_bits_rounding <= 8:
            raise ValueError("n_bits_rounding must be between 2 and 8 inclusive.")

        rounding_threshold_bits = {"n_bits": n_bits_rounding, "method": method}

    return rounding_threshold_bits


def check_device_is_valid(device: str) -> str:
    """Check whether the device string is valid or raise an exception.

    Args:
        device (str): the device string. Valid values are 'cpu', 'cuda'

    Returns:
        str: the valid device string

    Raises:
        ValueError: if the device string is incorrect
    """

    str_devices = "[" + ",".join(map(lambda s: "'" + s + "'", SUPPORTED_DEVICES)) + "]"

    if device not in SUPPORTED_DEVICES:
        raise ValueError(
            f"Model compilation targets given through the `device` "
            f"argument can be one of {str_devices}"
        )

    return device


def check_compilation_device_is_valid_and_is_cuda(device: str) -> bool:
    """Check whether the device string for compilation or FHE execution is CUDA or CPU.

    Args:
        device (str): the device string. Valid values are 'cpu', 'cuda'

    Returns:
        bool: whether GPU should be enabled for compilation

    Raises:
        ValueError: if the device string is incorrect or if CUDA is not supported
    """

    # Only parse device ids for FHE execution
    device = check_device_is_valid(device)

    # Allow forcing device to GPU for tests
    if os.environ.get("CML_USE_GPU", False) == "1" and not device == "cuda":  # pragma: no cover
        if not check_gpu_enabled():
            raise ValueError(
                "CUDA FHE execution was requested with CML_USE_GPU but the Concrete runtime "
                "that is installed on this system does not support CUDA. Please"
                "install a GPU-enabled Concrete-Python package."
            )

        print(f"Compilation device override, was '{device}' -> change to 'cuda'")
        device = "cuda"

    # All other devices are considered cpu for now
    is_cuda = device == "cuda"

    if is_cuda:
        if not check_gpu_enabled():
            raise ValueError(
                "CUDA FHE execution was requested but the Concrete runtime "
                "that is installed on this system does not support CUDA. Please"
                "install a GPU-enabled Concrete-Python package."
            )

        return True  # pragma: no cover

    return False


def check_execution_device_is_valid_and_is_cuda(
    is_compiled_for_cuda: bool,
    fhe: Union[FheMode, str],
) -> None:
    """Check whether the circuit can be executed on the required device.

    Args:
        is_compiled_for_cuda (bool): whether the circuit is compiled for CUDA
        fhe (Union[FheMode, str]): the execution mode of the circuit

    Raises:
        ValueError: if the requested device is not available
    """

    if fhe == FheMode.EXECUTE and is_compiled_for_cuda:
        if not check_gpu_available():
            raise ValueError(
                "CUDA FHE execution was requested but no compatible CUDA "
                "enabled device could be found"
            )
