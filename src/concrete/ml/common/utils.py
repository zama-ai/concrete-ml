"""Utils that can be re-used by other pieces of code in the module."""

import string
from types import FunctionType
from typing import Callable, Dict, Iterable, Tuple

import onnx

from ..common.debugging import assert_true

_VALID_ARG_CHARS = set(string.ascii_letters).union(str(i) for i in range(10)).union(("_",))

# Default probability of success of PBS
DEFAULT_P_ERROR_PBS = 6.3342483999973e-05


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
