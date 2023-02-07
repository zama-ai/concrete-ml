"""Utils that can be re-used by other pieces of code in the module."""

import string
from functools import partial
from types import FunctionType
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy
import onnx
import torch
from concrete.numpy.dtypes import Integer

from ..common.debugging import assert_true

_VALID_ARG_CHARS = set(string.ascii_letters).union(str(i) for i in range(10)).union(("_",))

SUPPORTED_TORCH_DTYPES = {
    "float64": torch.float64,
    "float32": torch.float32,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
}

MAX_BITWIDTH_BACKWARD_COMPATIBLE = 8


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
    # Some input names can be invalid arg names (e.g. coming from torch input.0) so sanitize them
    # to be valid python arg names.
    orig_args_to_proxy_func_args = {
        arg_name: f"_{replace_invalid_arg_name_chars(arg_name)}"
        for arg_name in desired_functions_arg_names
    }
    proxy_func_arg_string = ", ".join(orig_args_to_proxy_func_args.values())
    proxy_func_name = replace_invalid_arg_name_chars(f"{function_to_proxy.__name__}_proxy")
    # compile is the built-in python compile to generate code at runtime.
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
    """Return (p_error, global_p_error) that we want to give to Concrete-Numpy and the compiler.

    The returned (p_error, global_p_error) depends on user's parameters and the way we want to
    manage defaults in Concrete-ML, which may be different from the way defaults are managed in
    Concrete-Numpy

    Principle:
        - if none are set, we set global_p_error to a default value of our choice
        - if both are set, we raise an error
        - if one is set, we use it and forward it to Concrete-Numpy and the compiler

    Note that global_p_error is currently not simulated by the VL, i.e., taken as 0.

    Args:
        p_error (Optional[float]): probability of error of a single PBS.
        global_p_error (Optional[float]): probability of error of the full circuit.

    Returns:
        (p_error, global_p_error): parameters to give to the compiler

    Raises:
        ValueError: if the two parameters are set (this is _not_ as in Concrete-Numpy)

    """
    # Default probability of error of a circuit. Only used if p_error is set to None
    # We also need to find the most appropriate value for default_global_p_error_pbs
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2223
    default_global_p_error_pbs = 0.01

    if (p_error, global_p_error) == (None, None):
        p_error, global_p_error = (None, default_global_p_error_pbs)
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

    It would be dangerous, since we set them in direct arguments in our calls to Concrete-Numpy.

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


def is_model_class_in_a_list(model_class, a_list):
    """Say if model_class (which may be a partial()) is an element of a_list.

    Args:
        model_class: the model
        a_list: the list in which to look

    Returns:
        whether the class is in the list

    """
    if isinstance(model_class, partial):
        return model_class.func in a_list

    return model_class in a_list


def get_model_name(model_class):
    """Return a model (which may be a partial()) name.

    Args:
        model_class: the model

    Returns:
        the class name

    """
    if isinstance(model_class, partial):
        return model_class.func.__name__

    return model_class.__name__


def is_pandas_dataframe(input_container):
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


def is_pandas_series(input_container):
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


def _get_dtype(values):
    """Get the values' dtype.

    Args:
        values (Union[numpy.ndarray, pandas.DataFrame, pandas.Series, torch.Tensor]): The values
            to consider

    Returns:
        List[Union[torch.dtype, numpy.dtype]]: The values' dtype(s)
    """
    # If the container is a pandas.DataFrame, retrieve all the different dtypes found in it
    if is_pandas_dataframe(values):
        return list(set(values.dtypes))

    # Note that numpy.ndarray, pandas.Series and torch.Tensor objects support the dtype attribute
    return [values.dtype]


def _is_of_dtype(values, dtypes):
    """Indicate if the values' dtype(s) matches the given one(s).

    Args:
        values (Union[numpy.ndarray, pandas.DataFrame, pandas.Series, torch.Tensor]): The values
            to consider
        dtypes (Union[str, List[str]]): The dtype(s) to consider.

    Returns:
        bool: If the values' dtype matches the given one. If several dtypes are given, indicates if
            the values' dtype matches at least one of them.
    """
    # Convert the dtype string to a list if only one is given
    dtypes = [dtypes] if isinstance(dtypes, str) else dtypes

    matches = []
    for dtype in dtypes:
        # If the container is a pandas.DataFrame, check that is contains only one dtype in total
        # and compare it to the given one
        if is_pandas_dataframe(values):
            matches.append(list(set(values.dtypes)) == [dtype])

        # If the container is a torch.Tensor, check that the given dtype is expected and compare it
        # to the given one using a dictionary mapping string dtypes to torch dtypes
        elif isinstance(values, torch.Tensor):
            matches.append(
                dtype in SUPPORTED_TORCH_DTYPES and values.dtype == SUPPORTED_TORCH_DTYPES[dtype]
            )

        # Else, if the container is a numpy.ndarray or pandas.Series, compare the value's dtype to
        # the given one, knowing that pandas dtypes are subinstances of numpy.dtype objects, which
        # can be compared to string dtypes (i.e. numpy.float64 == 'float64' is True)
        else:
            matches.append(values.dtype == dtype)

    return any(matches)


def _cast_to_dtype(values, dtype):
    """Cast the values to the given dtype.

    Args:
        values (Union[numpy.ndarray, pandas.DataFrame, pandas.Series, torch.Tensor]): The values
            to consider
        dtype (str): The dtype to consider.

    Returns:
        Union[numpy.ndarray, pandas.DataFrame, pandas.Series, torch.Tensor]: The values casted to
            the given dtype
    """
    # If the container is a torch.Tensor and if the given dtype is expected, cast the values to this
    # dtype using a dictionary mapping string dtypes to torch dtypes
    if isinstance(values, torch.Tensor) and dtype in SUPPORTED_TORCH_DTYPES:
        return values.to(SUPPORTED_TORCH_DTYPES[dtype])

    # Note that numpy.ndarray, pandas.Series and pandas.DataFrame objects support the astype method
    return values.astype(dtype)


def check_dtype_and_cast(values, expected_dtype, error_information=None):
    """Check that the values' dtype(s) match(es) the given expected dtype.

    If they don't match, cast the values to the expected dtype if possible, else raise a ValueError.

    Args:
        values (Union[numpy.ndarray, pandas.DataFrame, pandas.Series, torch.Tensor]): The values
            to consider
        expected_dtype (str): The expected dtype, either "float32" or "int64"
        error_information (str): Additional information to put in front of the error message when
            raising a ValueError. Default to None.

    Returns:
        Union[numpy.ndarray, pandas.DataFrame, pandas.Series, torch.Tensor]: The values with
            proper dtype.

    Raises:
        ValueError: If the values' dtype don't match the expected one and casting is not possible.
    """
    # This case should not be handled here as this type appears when fitting a model (when calling
    # the predict_proba method), where inputs and targets already have the right dtypes
    if isinstance(values, torch.utils.data.dataset.Subset):
        return values

    assert_true(
        expected_dtype in ("float32", "int64"),
        f'Expected dtype parameter should either be "float32" or "int64". Got {expected_dtype}',
    )

    assert_true(
        isinstance(values, (numpy.ndarray, torch.Tensor, list))
        or is_pandas_dataframe(values)
        or is_pandas_series(values),
        "Unsupported type. Expected numpy.ndarray, pandas.DataFrame, pandas.Series, list "
        f"or torch.Tensor but got {type(values)}.",
    )

    # Convert the list to a float32 torch tensor if it is X or an int64 tensor if it is y
    if isinstance(values, list):
        return _cast_to_dtype(torch.tensor(values), expected_dtype)

    # If the expected dtype is an int64 and the values are integers of lower precision, we can
    # safely cast them to int64
    if expected_dtype == "int64" and _is_of_dtype(values, ["int32", "int16", "int8"]):
        return _cast_to_dtype(values, expected_dtype)

    # If the expected dtype is a float32 and the values are float64, we chose to cast them to
    # float32 in order to give the user more flexibility. This should not have a great impact on
    # the models' performances
    if expected_dtype == "float32" and _is_of_dtype(values, "float64"):
        return _cast_to_dtype(values, expected_dtype)

    # Else, if the values' dtype doesn't match the expected one, raise an error
    if not _is_of_dtype(values, expected_dtype):
        error_message = (
            "dtype does not match the expected dtype and values cannot be properly casted "
            f"using that latter. Expected dtype {expected_dtype} and got {_get_dtype(values)} "
            f"with type {type(values)}."
        )

        if error_information is not None:
            error_message = error_information + " " + error_message

        raise ValueError(error_message)

    # Return the values if their dtype matches the expected one
    return values


def compute_bits_precision(x: numpy.ndarray) -> int:
    """Compute the number of bits required to represent x.

    Args:
        x (numpy.ndarray): Integer data

    Returns:
        int: the number of bits required to represent x
    """
    return Integer.that_can_represent([x.min(), x.max()]).bit_width
