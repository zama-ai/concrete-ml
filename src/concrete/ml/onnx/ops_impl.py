"""ONNX ops implementation in python + numpy."""

# pylint: disable=too-many-lines
from inspect import signature
from typing import Optional, Sequence, Set, SupportsIndex, Tuple, Union

import numpy
import onnx
import torch
import torch.nn
from brevitas.function import max_int, min_int
from concrete.numpy import univariate
from scipy import special

from ..common.debugging import assert_true


# FIXME: to remove once https://github.com/zama-ai/concrete-ml-internal/issues/1117 is done.
# This function is only used for comparison operators that return boolean values by default.
def cast_to_float(inputs):
    """Cast values to floating points.

    Args:
        inputs (Tuple[numpy.ndarray]): The values to consider.

    Returns:
        Tuple[numpy.ndarray]: The float values.
    """
    return tuple(map(lambda x: x.astype(numpy.float64), inputs))


class ONNXMixedFunction:
    """A mixed quantized-raw valued onnx function.

    ONNX functions will take inputs which can be either quantized or float. Some functions
    only take quantized inputs, but some functions take both types. For mixed functions
    we need to tag the parameters that do not need quantization. Thus quantized ops
    can know which inputs are not QuantizedArray and we avoid unnecessary wrapping of float
    values as QuantizedArrays.
    """

    def __init__(self, function, non_quant_params: Set[str]):
        """Create the mixed function and raw parameter list.

        Args:
            function (Any): function to be decorated
            non_quant_params: Set[str]: set of parameters that will not be quantized (stored
                as numpy.ndarray)
        """

        self.non_quant_params: Set[str] = non_quant_params
        bad_non_quant_params = set(non_quant_params).difference(set(signature(function).parameters))
        assert_true(
            len(bad_non_quant_params) == 0,
            f"ONNX function {function.__name__} tagged with invalid integer parameters: "
            ",".join(bad_non_quant_params),
        )
        self.function = function  # type: ignore

    def __call__(self, *args, **kwargs):
        """Call the wrapped numpy function.

        Args:
            args (tuple[Any]): function arguments
            kwargs (dict[str, Any]): function key value arguments

        Returns:
            result (Any): result of calling the wrapped function on the input arguments
        """
        return self.function(*args, **kwargs)

    @property
    def __name__(self):
        """Return the wrapped function name.

        Returns:
            result (str): name of the wrapped function
        """
        return self.function.__name__


def onnx_func_raw_args(*args):
    """Decorate a numpy onnx function to flag the raw/non quantized inputs.

    Args:
        *args (tuple[Any]): function argument names

    Returns:
        result (ONNXMixedFunction): wrapped numpy function with a list of mixed arguments
    """

    def decoration(function):
        """Construct the mixed function class.

        Args:
            function (Any): function to be decorated

        Returns:
            result (ONNXMixedFunction): wrapped numpy function with a list of mixed arguments
        """
        return ONNXMixedFunction(function, set(args))

    return decoration


def numpy_where_body(
    c: numpy.ndarray, t: numpy.ndarray, f: Union[numpy.ndarray, int], /
) -> numpy.ndarray:
    """Compute the equivalent of numpy.where.

    This function is not mapped to any ONNX operator (as opposed to numpy_where). It is usable by
    functions which are mapped to ONNX operators, e.g. numpy_div or numpy_where.

    Args:
        c (numpy.ndarray): Condition operand.
        t (numpy.ndarray): True operand.
        f (numpy.ndarray): False operand.

    Returns:
        numpy.ndarray: numpy.where(c, t, f)

    """

    # FIXME: can it be improved with a native numpy.where in Concrete Numpy?
    # https://github.com/zama-ai/concrete-numpy-internal/issues/1429

    return c * t + (1.0 - c) * f


def numpy_where(c: numpy.ndarray, t: numpy.ndarray, f: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute the equivalent of numpy.where.

    Args:
        c (numpy.ndarray): Condition operand.
        t (numpy.ndarray): True operand.
        f (numpy.ndarray): False operand.

    Returns:
        numpy.ndarray: numpy.where(c, t, f)

    """
    return (numpy_where_body(c, t, f),)


def numpy_add(a: numpy.ndarray, b: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute add in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Add-13

    Args:
        a (numpy.ndarray): First operand.
        b (numpy.ndarray): Second operand.

    Returns:
        Tuple[numpy.ndarray]: Result, has same element type as two inputs
    """
    return (a + b,)


# input, min and max are python built-in but we need to match the ONNX naming, ignore the lint
# pylint: disable=redefined-builtin
@onnx_func_raw_args("min", "max")
def numpy_clip(a: numpy.ndarray, /, min=None, max=None) -> Tuple[numpy.ndarray]:
    """Compute clip in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Clip-13

    Args:
        a (numpy.ndarray): Input tensor whose elements to be clipped.
        min ([type], optional): Minimum value, under which element is replaced by min.
            It must be a scalar(tensor of empty shape).
            Defaults to None.
        max ([type], optional): Maximum value, above which element is replaced by max.
            It must be a scalar(tensor of empty shape).
            Defaults to None.

    Returns:
        Tuple[numpy.ndarray]: Output tensor with clipped input elements.
    """

    assert_true(
        min is not None and max is not None,
        f"{numpy_clip.__name__} currently does not support passing `None` "
        "for the min or max inputs.",
    )

    return (numpy.clip(a, min, max),)


# pylint: enable=redefined-builtin


def numpy_constant(**kwargs):
    """Return the constant passed as a kwarg.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Constant-13

    Args:
        **kwargs: keyword arguments

    Returns:
        Any: The stored constant.
    """

    # Given the variety of possible kwargs (see spec), we just check there is only one kwargs and
    # return the corresponding value
    assert len(kwargs) == 1
    single_key = next(iter(kwargs.keys()))

    return (kwargs[single_key],)


# transA and transB are not snake case but need to match ONNX attribute naming, ignore the lint
# pylint: disable=invalid-name
# 1 is technically an int but is accepted by mypy as a float (and it simplifies our life for
# compilation) so instead of passing 1.0 by default 1 is passed
@onnx_func_raw_args("c")
def numpy_gemm(
    a: numpy.ndarray,
    b: numpy.ndarray,
    /,
    c: Optional[numpy.ndarray] = None,
    *,
    alpha: float = 1,
    beta: float = 1,
    transA: int = 0,
    transB: int = 0,
) -> Tuple[numpy.ndarray]:
    """Compute Gemm in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Gemm-13

    Args:
        a (numpy.ndarray): Input tensor A. The shape of A should be (M, K) if transA is 0, or (K, M)
            if transA is non-zero.
        b (numpy.ndarray): Input tensor B. The shape of B should be (K, N) if transB is 0, or (N, K)
            if transB is non-zero.
        c (Optional[numpy.ndarray]): Optional input tensor C. If not specified, the
            computation is done as if C is a scalar 0. The shape of C should be unidirectional
            broadcastable to (M, N).
            Defaults to None.
        alpha (float): Scalar multiplier for the product of input tensors A * B.
            Defaults to 1.
        beta (float): Scalar multiplier for input tensor C.
            Defaults to 1.
        transA (int): Whether A should be transposed. The type is kept as int as it's the
            type used by ONNX and it can easily be interpreted by python as a boolean.
            Defaults to 0.
        transB (int): Whether B should be transposed. The type is kept as int as it's the
            type used by ONNX and it can easily be interpreted by python as a boolean.
            Defaults to 0.

    Returns:
        Tuple[numpy.ndarray]: The tuple containing the result tensor
    """
    # If alpha and beta are integer, apply the int type for concrete-numpy
    # to see they are integers (see issue #277)
    processed_alpha = int(alpha) if round(alpha) == alpha else alpha
    processed_beta = int(beta) if round(beta) == beta else beta

    a_prime = numpy.transpose(a) if transA else a
    b_prime = numpy.transpose(b) if transB else b
    c_prime: Union[numpy.ndarray, float] = c if c is not None else 0

    # Do
    #
    #       y = processed_alpha * numpy.matmul(a_prime, b_prime) + processed_beta * c_prime
    #
    # in an efficient way, i.e. to make tracing directly optimized, without expecting any opt from
    # the compiler here

    y = numpy.matmul(a_prime, b_prime)

    if processed_alpha != 1:
        y = y * processed_alpha

    if numpy.any(c_prime != 0):
        if processed_beta == 1:
            y = y + c_prime
        else:
            y = y + processed_beta * c_prime

    return (y,)


def numpy_matmul(a: numpy.ndarray, b: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute matmul in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#MatMul-13

    Args:
        a (numpy.ndarray): N-dimensional matrix A
        b (numpy.ndarray): N-dimensional matrix B

    Returns:
        Tuple[numpy.ndarray]: Matrix multiply results from A * B
    """
    return (numpy.matmul(a, b),)


def numpy_relu(x: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute relu in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Relu-14

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.maximum(x, 0),)


def numpy_sigmoid(x: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute sigmoid in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sigmoid-13

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (1.0 / (1.0 + numpy.exp(-x)),)


def numpy_softmax(x, axis=1, keepdims=True):
    """Compute softmax in numpy according to ONNX spec.

    Softmax is currently not supported in FHE.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#softmax-13

    Args:
        x (numpy.ndarray): Input tensor
        axis (None, int, tuple of ints): Axis or axes along which a softmax's sum is performed. If
            None, it will sum all of the elements of the input array.  If axis is negative it counts
            from the last to the first axis. Default to 1.
        keepdims (bool): If True, the axes which are reduced along the sum are left in the result as
            dimensions with size one. Default to True.

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    x = numpy.exp(x)
    x /= numpy.sum(x, axis=axis, keepdims=keepdims)
    return (x,)


def numpy_cos(x: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute cos in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Cos-7

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.cos(x),)  # pragma: no cover


def numpy_cosh(x: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute cosh in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Cosh-9

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.cosh(x),)  # pragma: no cover


def numpy_sin(x: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute sin in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sin-7

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.sin(x),)  # pragma: no cover


def numpy_sinh(x: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute sinh in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sinh-9

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.sinh(x),)  # pragma: no cover


def numpy_tan(x: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute tan in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Tan-7

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.tan(x),)  # pragma: no cover


def numpy_tanh(x: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute tanh in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Tanh-13

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.tanh(x),)


def numpy_acos(x: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute acos in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Acos-7

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.arccos(x),)  # pragma: no cover


def numpy_acosh(x: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute acosh in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Acosh-9

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.arccosh(x),)  # pragma: no cover


def numpy_asin(x: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute asin in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Asin-7

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.arcsin(x),)  # pragma: no cover


def numpy_asinh(x: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute sinh in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Asinh-9

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.arcsinh(x),)  # pragma: no cover


def numpy_atan(x: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute atan in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Atan-7

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.arctan(x),)  # pragma: no cover


def numpy_atanh(x: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute atanh in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Atanh-9

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.arctanh(x),)  # pragma: no cover


def numpy_elu(x: numpy.ndarray, /, *, alpha: float = 1) -> Tuple[numpy.ndarray]:
    """Compute elu in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Elu-6

    Args:
        x (numpy.ndarray): Input tensor
        alpha (float): Coefficient

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return numpy_where(x > 0, x, alpha * (numpy.exp(x) - 1))


def numpy_selu(
    x: numpy.ndarray,
    /,
    *,
    alpha: float = 1.6732632423543772848170429916717,
    gamma: float = 1.0507009873554804934193349852946,
) -> Tuple[numpy.ndarray]:
    """Compute selu in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Selu-6

    Args:
        x (numpy.ndarray): Input tensor
        alpha (float): Coefficient
        gamma (float): Coefficient

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return numpy_where(x > 0, gamma * x, (gamma * alpha) * (numpy.exp(x) - 1))


def numpy_celu(x: numpy.ndarray, /, *, alpha: float = 1) -> Tuple[numpy.ndarray]:
    """Compute celu in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Celu-12

    Args:
        x (numpy.ndarray): Input tensor
        alpha (float): Coefficient

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (numpy.maximum(0, x) + numpy.minimum(0, alpha * (numpy.exp(x / alpha) - 1)),)


def numpy_leakyrelu(x: numpy.ndarray, /, *, alpha: float = 0.01) -> Tuple[numpy.ndarray]:
    """Compute leakyrelu in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#LeakyRelu-6

    Args:
        x (numpy.ndarray): Input tensor
        alpha (float): Coefficient

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return numpy_where(x > 0, x, alpha * x)


def numpy_thresholdedrelu(x: numpy.ndarray, /, *, alpha: float = 1) -> Tuple[numpy.ndarray]:
    """Compute thresholdedrelu in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#ThresholdedRelu-10

    Args:
        x (numpy.ndarray): Input tensor
        alpha (float): Coefficient

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    if x > alpha:  # pragma: no cover
        return (x,)  # pragma: no cover

    return (numpy.zeros_like(x),)  # pragma: no cover


def numpy_hardsigmoid(
    x: numpy.ndarray, /, *, alpha: float = 0.2, beta: float = 0.5
) -> Tuple[numpy.ndarray]:
    """Compute hardsigmoid in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#HardSigmoid-6

    Args:
        x (numpy.ndarray): Input tensor
        alpha (float): Coefficient
        beta (float): Coefficient

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (numpy.maximum(0, numpy.minimum(1, alpha * x + beta)),)


def numpy_softplus(x: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute softplus in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Softplus-1

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (numpy.log(numpy.exp(x) + 1),)


def numpy_abs(x: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute abs in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Abs-13

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (numpy.abs(x),)


def numpy_div(a: numpy.ndarray, b: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute div in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Div-14

    Args:
        a (numpy.ndarray): Input tensor
        b (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    # FIXME: remove this once https://github.com/zama-ai/concrete-ml-internal/issues/857 is
    # explained
    bp = numpy_where_body(b != 0, b, 1)
    ans = numpy.divide(a, bp)

    return (ans,)


def numpy_mul(a: numpy.ndarray, b: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute mul in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Mul-14

    Args:
        a (numpy.ndarray): Input tensor
        b (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (a * b,)


def numpy_sub(a: numpy.ndarray, b: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute sub in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sub-14

    Args:
        a (numpy.ndarray): Input tensor
        b (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (a - b,)


def numpy_log(x: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute log in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Log-13

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    # Epsilon is here to avoid problems with 0 or negative values, which may happen when Concrete
    # Numpy creates the table (even if these problematic values would normally never be used)
    epsilon = 10**-8

    return (numpy.log(numpy.maximum(x, epsilon)),)


@onnx_func_raw_args("slope")
def numpy_prelu(x: numpy.ndarray, slope: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute prelu in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#prelu-16

    Args:
        x (numpy.ndarray): Input tensor
        slope (numpy.ndarray): Slope of PRelu

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    a = numpy.minimum(0, slope * x)
    b = numpy.maximum(0, x)
    return (a + b,)


def numpy_erf(x: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute erf in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Erf-13

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (univariate(special.erf)(x),)  # pylint: disable=no-member


def numpy_hardswish(x: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute hardswish in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#hardswish-14

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    alpha = 1.0 / 6
    beta = 0.5
    r = x * numpy.maximum(0, numpy.minimum(1, alpha * x + beta))

    return (r,)


def numpy_exp(x: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute exponential in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Exp-13

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: The exponential of the input tensor computed element-wise
    """

    return (numpy.exp(x),)


def numpy_equal(x: numpy.ndarray, y: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute equal in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Equal-11

    Args:
        x (numpy.ndarray): Input tensor
        y (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (numpy.equal(x, y),)


def numpy_not(x: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute not in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Not-1

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (numpy.logical_not(x),)


# FIXME: to remove once https://github.com/zama-ai/concrete-ml-internal/issues/1117 is done.
def numpy_not_float(x: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute not in numpy according to ONNX spec and cast outputs to floats.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Not-1

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return cast_to_float(numpy_not(x))


def numpy_greater(x: numpy.ndarray, y: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute greater in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Greater-13

    Args:
        x (numpy.ndarray): Input tensor
        y (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (numpy.greater(x, y),)


# FIXME: to remove once https://github.com/zama-ai/concrete-ml-internal/issues/1117 is done.
def numpy_greater_float(x: numpy.ndarray, y: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute greater in numpy according to ONNX spec and cast outputs to floats.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Greater-13

    Args:
        x (numpy.ndarray): Input tensor
        y (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return cast_to_float(numpy_greater(x, y))


def numpy_greater_or_equal(x: numpy.ndarray, y: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute greater or equal in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#GreaterOrEqual-12

    Args:
        x (numpy.ndarray): Input tensor
        y (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (numpy.greater_equal(x, y),)


# FIXME: to remove once https://github.com/zama-ai/concrete-ml-internal/issues/1117 is done.
def numpy_greater_or_equal_float(x: numpy.ndarray, y: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute greater or equal in numpy according to ONNX specs and cast outputs to floats.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#GreaterOrEqual-12

    Args:
        x (numpy.ndarray): Input tensor
        y (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return cast_to_float(numpy_greater_or_equal(x, y))


def numpy_less(x: numpy.ndarray, y: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute less in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Less-13

    Args:
        x (numpy.ndarray): Input tensor
        y (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (numpy.less(x, y),)


# FIXME: to remove once https://github.com/zama-ai/concrete-ml-internal/issues/1117 is done.
def numpy_less_float(x: numpy.ndarray, y: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute less in numpy according to ONNX spec and cast outputs to floats.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Less-13

    Args:
        x (numpy.ndarray): Input tensor
        y (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return cast_to_float(numpy_less(x, y))


def numpy_less_or_equal(x: numpy.ndarray, y: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute less or equal in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#LessOrEqual-12

    Args:
        x (numpy.ndarray): Input tensor
        y (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (numpy.less_equal(x, y),)


# FIXME: to remove once https://github.com/zama-ai/concrete-ml-internal/issues/1117 is done.
def numpy_less_or_equal_float(x: numpy.ndarray, y: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute less or equal in numpy according to ONNX spec and cast outputs to floats.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#LessOrEqual-12

    Args:
        x (numpy.ndarray): Input tensor
        y (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return cast_to_float(numpy_less_or_equal(x, y))


def numpy_identity(x: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute identity in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Identity-14

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (x,)


@onnx_func_raw_args("newshape")
def numpy_reshape(
    x: numpy.ndarray, newshape: numpy.ndarray, /, *, allowzero=0
) -> Tuple[numpy.ndarray]:
    """Compute reshape in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Reshape-13

    Args:
        x (numpy.ndarray): Input tensor
        newshape (numpy.ndarray): New shape
        allowzero (int): ONNX legacy parameter, by default 0 -> behave like numpy reshape

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    assert_true(allowzero == 0, "Concrete ML currently only accepts numpy style reshape in ONNX")

    return (numpy.reshape(x, newshape),)


def numpy_transpose(x: numpy.ndarray, /, *, perm=None) -> Tuple[numpy.ndarray]:
    """Transpose in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Transpose-13

    Args:
        x (numpy.ndarray): Input tensor
        perm (numpy.ndarray): Permutation of the axes

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    # FIXME: #931, remove the no-cover once #931 is done
    return (numpy.transpose(x, axes=perm),)  # pragma: no cover


@onnx_func_raw_args("b")
def torch_conv(
    x: numpy.ndarray,
    w: numpy.ndarray,
    b: numpy.ndarray,
    /,
    *,
    dilations: Tuple[int, ...],
    group: int = 1,
    kernel_shape: Tuple[int, ...],
    pads: Tuple[int, ...],
    strides: Tuple[int, ...],
) -> Tuple[numpy.ndarray]:
    """Compute N-D convolution using Torch.

    Currently supports 2d convolution with torch semantics. This function is also ONNX compatible.

    See: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv

    Args:
        x (numpy.ndarray): input data (many dtypes are supported). Shape is N x C x H x W for 2d
        w (numpy.ndarray): weights tensor. Shape is (O x I x Kh x Kw) for 2d
        b (numpy.ndarray, Optional): bias tensor, Shape is (O,)
        dilations (Tuple[int]): dilation of the kernel, default 1 on all dimensions.
        group (int): number of convolution groups, default 1
        kernel_shape (Tuple[int]): shape of the kernel. Should have 2 elements for 2d conv
        pads (Tuple[int]): padding in ONNX format (begin, end) on each axis
        strides (Tuple[int]): stride of the convolution on each axis

    Returns:
        res (numpy.ndarray): a tensor of size (N x OutChannels x OutHeight x OutWidth).
           See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    Raises:
        AssertionError: if the convolution arguments are wrong
    """

    # Convert the inputs to tensors to compute conv using torch
    tx = torch.Tensor(x.copy())
    tw = torch.Tensor(w.copy())
    tb = torch.Tensor(b.squeeze().copy()) if b is not None else None

    assert_true(len(kernel_shape) == 2, "The convolution operator currently supports only 2-d")
    assert_true(
        bool(numpy.all(numpy.asarray(dilations) == 1)),
        "The convolution operator in Concrete Numpy does not support dilation",
    )

    # For mypy
    assert len(pads) == 4

    assert_true(group == 1, "The convolution operator in Concrete Numpy does not support groups")
    assert_true(
        pads[0] == pads[1] and pads[2] == pads[3],
        "The convolution operator in Concrete ML only supports symmetric padding",
    )

    # Extract the 'begin' pads for all dimensions. Begin padding should be the same as the end pad
    torch_pads = (pads[0], pads[1])

    # Compute the torch convolution
    res = torch.conv2d(tx, tw, tb, strides, torch_pads, dilations, group).numpy()

    return (res,)


def torch_avgpool(
    x: numpy.ndarray,
    /,
    *,
    ceil_mode: int,
    kernel_shape: Tuple[int, ...],
    pads: Tuple[int, ...],
    strides: Tuple[int, ...],
) -> Tuple[numpy.ndarray]:
    """Compute Average Pooling using Torch.

    Currently supports 2d average pooling with torch semantics. This function is ONNX compatible.

    See: https://github.com/onnx/onnx/blob/main/docs/Operators.md#AveragePool

    Args:
        x (numpy.ndarray): input data (many dtypes are supported). Shape is N x C x H x W for 2d
        ceil_mode (int): ONNX rounding parameter, expected 0 (torch style dimension computation)
        kernel_shape (Tuple[int]): shape of the kernel. Should have 2 elements for 2d conv
        pads (Tuple[int]): padding in ONNX format (begin, end) on each axis
        strides (Tuple[int]): stride of the convolution on each axis

    Returns:
        res (numpy.ndarray): a tensor of size (N x InChannels x OutHeight x OutWidth).
           See https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html

    Raises:
        AssertionError: if the pooling arguments are wrong
    """

    # Convert the inputs to tensors to compute conv using torch
    tx = torch.Tensor(x.copy())

    assert_true(len(kernel_shape) == 2, "The convolution operator currently supports only 2-d")
    assert_true(ceil_mode == 0, "Average Pooling only supports torch style dimension computation")
    # For mypy
    assert len(pads) == 4

    # For mypy
    assert len(kernel_shape) == 2

    assert len(strides) == 2

    assert_true(
        pads[0] == pads[1] and pads[2] == pads[3],
        "The convolution operator in Concrete ML only supports symmetric padding",
    )

    # Extract the 'begin' pads for all dimensions. Begin padding should be the same as the end pad
    torch_pads = (pads[0], pads[1])

    # Compute the torch convolution
    res = torch.nn.functional.avg_pool2d(
        tx, kernel_shape, strides, torch_pads, ceil_mode=False, count_include_pad=True
    ).numpy()
    return (res,)


@onnx_func_raw_args("pads")
def numpy_pad(
    data: numpy.ndarray,
    pads: numpy.ndarray,
    constant_value: Union[numpy.ndarray, None] = None,
    /,
    *,
    mode: str,
) -> Tuple[numpy.ndarray]:
    """Apply padding in numpy according to ONNX spec.

    See: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Pad

    Args:
        data (numpy.ndarray): Input variable/tensor to pad
        pads (numpy.ndarray): List of pads (size 8) to apply, two per N,C,H,W dimension
        constant_value (float): Constant value to use for padding
        mode (str): padding mode: constant/edge/reflect

    Returns:
        res (numpy.ndarray): Padded tensor
    """

    assert_true(bool(numpy.all(pads == 0)), "Padding operator supported only with all pads at zero")
    assert_true(mode == "constant", "Padding only supported with a constant pad value")
    assert_true(
        constant_value is None or constant_value == 0, "Pad only accepts a constant padding with 0s"
    )

    return (data,)


def numpy_cast(data: numpy.ndarray, /, *, to: int) -> Tuple[numpy.ndarray]:
    """Execute ONNX cast in Numpy.

    Supports only booleans for now, which are converted to integers.

    See: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cast

    Args:
        data (numpy.ndarray): Input encrypted tensor
        to (int): integer value of the onnx.TensorProto DataType enum

    Returns:
        result (numpy.ndarray): a tensor with the required data type
    """
    assert_true(to == onnx.TensorProto.BOOL)
    return (data.astype(numpy.float64),)


def numpy_batchnorm(
    x: numpy.ndarray,
    scale: numpy.ndarray,
    bias: numpy.ndarray,
    input_mean: numpy.ndarray,
    input_var: numpy.ndarray,
    /,
    *,
    epsilon=1e-05,
    momentum=0.9,  # pylint: disable=unused-argument
    training_mode=0,
) -> Tuple[numpy.ndarray]:
    """Compute the batch normalization of the input tensor.

    This can be expressed as:

    Y = (X - input_mean) / sqrt(input_var + epsilon) * scale + B

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#BatchNormalization-14

    Args:
        x (numpy.ndarray): tensor to normalize, dimensions are in the form of (N,C,D1,D2,...,Dn),
                           where N is the batch size, C is the number of channels.
        scale (numpy.ndarray): scale tensor of shape (C,)
        bias (numpy.ndarray): bias tensor of shape (C,)
        input_mean (numpy.ndarray): mean values to use for each input channel, shape (C,)
        input_var (numpy.ndarray): variance values to use for each input channel, shape (C,)
        epsilon (float): avoids division by zero
        momentum (float): momentum used during training of the mean/variance, not used in inference
        training_mode (int): if the model was exported in training mode this is set to 1, else 0

    Returns:
        numpy.ndarray: Normalized tensor
    """

    assert_true(
        training_mode == 0,
        "Model was exported with BatchNorm in training mode, this is not supported",
    )

    assert_true(
        x.shape[1] == input_mean.shape[0],
        "Number of channels in BatchNorm mean does not match input",
    )

    assert_true(
        x.shape[1] == scale.shape[0],
        "Number of channels in BatchNorm scale does not match input",
    )

    assert_true(
        x.shape[1] == input_var.shape[0],
        "Number of channels in BatchNorm variance does not match input",
    )

    assert_true(
        x.shape[1] == bias.shape[0],
        "Number of channels in BatchNorm bias does not match input",
    )

    shape_input = numpy.ones_like(x.shape)
    shape_input[1] = x.shape[1]

    input_mean = input_mean.reshape(shape_input)
    input_var = input_var.reshape(shape_input)
    scale = scale.reshape(shape_input)
    bias = bias.reshape(shape_input)

    y = (x - input_mean) / numpy.sqrt(input_var + epsilon) * scale + bias
    return (y,)


def numpy_flatten(x: numpy.ndarray, /, *, axis: int = 1) -> Tuple[numpy.ndarray]:
    """Flatten a tensor into a 2d array.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Flatten-13.

    Args:
        x (numpy.ndarray): tensor to flatten
        axis (int): axis after which all dimensions will be flattened (axis=0 gives a 1D output)

    Returns:
        result: flattened tensor
    """
    output_shape: Sequence[SupportsIndex]
    output_shape = (*x.shape[0:axis], numpy.prod(x.shape[axis:]))

    return (numpy.reshape(x, output_shape),)


def numpy_or(a: numpy.ndarray, b: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute or in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Or-7

    Args:
        a (numpy.ndarray): Input tensor
        b (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.logical_or(a, b),)


# FIXME: to remove once https://github.com/zama-ai/concrete-ml-internal/issues/1117 is done.
def numpy_or_float(a: numpy.ndarray, b: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute or in numpy according to ONNX spec and cast outputs to floats.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Or-7

    Args:
        a (numpy.ndarray): Input tensor
        b (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return cast_to_float(numpy_or(a, b))


def numpy_round(a: numpy.ndarray) -> Tuple[numpy.ndarray]:
    """Compute round in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Round-11
    Remark that ONNX Round operator is actually a rint, since the number of decimals is forced to
    be 0

    Args:
        a (numpy.ndarray): Input tensor whose elements to be rounded.

    Returns:
        Tuple[numpy.ndarray]: Output tensor with rounded input elements.
    """

    return (numpy.rint(a),)


def numpy_pow(a: numpy.ndarray, b: numpy.ndarray) -> Tuple[numpy.ndarray]:
    """Compute pow in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Pow-13

    Args:
        a (numpy.ndarray): Input tensor whose elements to be raised.
        b (numpy.ndarray): The power to which we want to raise.

    Returns:
        Tuple[numpy.ndarray]: Output tensor.
    """

    return (numpy.power(a, b),)


# pylint: disable=unused-argument
@onnx_func_raw_args("axes")
def numpy_reduce_sum(
    a: numpy.ndarray,
    /,
    axes: Optional[numpy.ndarray] = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
) -> Tuple[numpy.ndarray]:
    """Compute ReduceSum in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceSum

    Args:
        a (numpy.ndarray): Input tensor whose elements to sum.
        axes (Optional[numpy.ndarray]): Array of integers along which to reduce. The default is to
            reduce over all the dimensions of the input tensor if 'noop_with_empty_axes' is false,
            else act as an Identity op when 'noop_with_empty_axes' is true. Accepted range is
            [-r, r-1] where r = rank(data). Default to None.
        keepdims (int): Keep the reduced dimension or not, 1 means keeping the
            input dimension, 0 will reduce it along the given axis. Default to 1.
        noop_with_empty_axes (int): Defines behavior if 'axes' is empty or set to None.
            Default behavior with 0 is to reduce all axes. When axes is empty and this
            attribute is set to true 1, input tensor will not be reduced, and the output
            tensor would be equivalent to input tensor. Default to 0.

    Returns:
        numpy.ndarray: Output reduced tensor.
    """

    assert_true(
        axes is not None and len(axes.shape) == 1 and axes[0] == 1,
        "ReduceSum currently only handles summing over axis 1.",
    )

    assert_true(len(a.shape) == 2, "ReduceSum currently only handles arrays of 2 dimensions")

    assert_true(
        keepdims == 1, "ReduceSum currently only keeps the inputs' dimensions for its outputs."
    )

    n_values = a.shape[1]
    assert_true(
        (n_values != 0) and (n_values & (n_values - 1) == 0),
        "ReduceSum currently only handles N values with N a power of 2.",
    )

    return (numpy.sum(a, axis=1, keepdims=True),)


@onnx_func_raw_args("scale", "zero_point", "bit_width")
def numpy_brevitas_quant(
    x: numpy.ndarray,
    /,
    scale: float,
    zero_point: float,
    bit_width: int,
    *,
    rounding_mode: str = "ROUND",
    signed: int = 1,
    narrow: int = 0,
):
    """Quantize according to Brevitas uniform quantization.

    Args:
        x (numpy.ndarray): Tensor to be quantized
        scale (float): Quantizer scale
        zero_point (float): Quantizer zero-point
        bit_width (int): Number of bits of the integer representation
        rounding_mode (str): Rounding mode (default and only accepted option is "ROUND")
        signed (int): Whether this op quantizes to signed integers (default 1),
        narrow (int): Whether this op quantizes to a narrow range of integers
            e.g. [-2**n_bits-1 .. 2**n_bits-1] (default 0),

    Returns:
        result (numpy.ndarray): Tensor with float quantized values
    """

    assert_true(rounding_mode == "ROUND", "Only rounding quantization is supported for Brevitas")
    assert_true(signed in (1, 0), "Signed flag in Brevitas quantizer must be 0/1")
    assert_true(narrow in (1, 0), "Narrow range flag in Brevitas quantizer must be 0/1")

    # Compute the re-scaled values
    y = x / scale
    y = y + zero_point

    # Clip the values to the correct range
    min_int_val = min_int(signed, narrow, bit_width)
    max_int_val = max_int(signed, narrow, bit_width)
    y = numpy.clip(y, min_int_val, max_int_val)

    # Quantize to produce integers representing the float quantized values
    y = numpy.rint(y)

    # Compute quantized floating point values
    y = (y - zero_point) * scale

    return (y,)
