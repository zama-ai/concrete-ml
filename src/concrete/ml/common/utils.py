"""Utils that can be re-used by other pieces of code in the module."""

import string
from types import FunctionType
from typing import Callable, Dict, Iterable, Optional, Tuple

import onnx

from ..common.debugging import assert_true

_VALID_ARG_CHARS = set(string.ascii_letters).union(str(i) for i in range(10)).union(("_",))

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
    # FIXME #2223: we'll find the most appropriate value for default_global_p_error_pbs
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
        assert_true(  # Defaults for Configuration object
            not (configuration.p_error is None and configuration.global_p_error == 1 / 100_000),
            "Setting p_error or global_p_error in the configuration is not supported, "
            "use kwargs instead.\n"
            "Please consider setting p_error and global_p_error to None in"
            " the Configuration object.\n"
            "Default value for Configuration.global_p_error is not None.",
        )
        assert_true(configuration.p_error is None, "Don't set p_error in configuration, use kwargs")
        assert_true(
            configuration.global_p_error is None,
            "Don't set global_p_error in configuration, use kwargs",
        )
