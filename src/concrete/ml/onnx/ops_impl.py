"""ONNX ops implementation in Python + NumPy."""

# pylint: disable=too-many-lines
from inspect import signature
from typing import Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy
import onnx
import onnx.helper
from brevitas.function import max_int, min_int
from concrete.fhe import conv as cnp_conv
from concrete.fhe import maxpool as cnp_maxpool
from concrete.fhe import univariate
from scipy import special
from typing_extensions import SupportsIndex

from ..common.debugging import assert_false, assert_true
from .onnx_impl_utils import (
    compute_onnx_pool_padding,
    numpy_onnx_pad,
    onnx_avgpool_compute_norm_const,
)


class RawOpOutput(numpy.ndarray):
    """Type construct that marks an ndarray as a raw output of a quantized op."""


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

    def __init__(self, function, non_quant_params: Set[str], output_is_raw: bool = False):
        """Create the mixed function and raw parameter list.

        Args:
            function (Any): function to be decorated
            non_quant_params: Set[str]: set of parameters that will not be quantized (stored
                as numpy.ndarray)
            output_is_raw (bool): indicates whether the op outputs a value that should
                not be quantized
        """

        self.non_quant_params: Set[str] = non_quant_params
        bad_non_quant_params = set(non_quant_params).difference(set(signature(function).parameters))
        assert_true(
            len(bad_non_quant_params) == 0,
            f"ONNX function {function.__name__} tagged with invalid integer parameters: "
            ",".join(bad_non_quant_params),
        )
        self.function = function  # type: ignore
        self.output_is_raw = output_is_raw

    def __call__(self, *args, **kwargs):
        """Call the wrapped numpy function.

        Args:
            args (tuple[Any]): function arguments
            kwargs (dict[str, Any]): function key value arguments

        Returns:
            result (Any): result of calling the wrapped function on the input arguments
        """
        result = self.function(*args, **kwargs)
        if self.output_is_raw:
            result = tuple(r.view(RawOpOutput) for r in result)
        return result

    @property
    def __name__(self):
        """Return the wrapped function name.

        Returns:
            result (str): name of the wrapped function
        """
        return self.function.__name__


def onnx_func_raw_args(*args, output_is_raw: bool = False):
    """Decorate a numpy onnx function to flag the raw/non quantized inputs.

    Args:
        *args (tuple[Any]): function argument names
        output_is_raw (bool): marks the function as returning raw
            values that should not be quantized

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
        return ONNXMixedFunction(function, set(args), output_is_raw)

    return decoration


def numpy_where_body(
    c: numpy.ndarray,
    t: numpy.ndarray,
    f: Union[numpy.ndarray, int],
) -> numpy.ndarray:
    """Compute the equivalent of numpy.where.

    This function is not mapped to any ONNX operator (as opposed to numpy_where). It is usable by
    functions which are mapped to ONNX operators, e.g., numpy_div or numpy_where.

    Args:
        c (numpy.ndarray): Condition operand.
        t (numpy.ndarray): True operand.
        f (numpy.ndarray): False operand.

    Returns:
        numpy.ndarray: numpy.where(c, t, f)

    """
    # Use numpy.where (it is currently supported by Concrete) once we investigate why it outputs a
    # a different dtype then the following workaround
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2738
    return c * t + (1.0 - c) * f


def numpy_where(
    c: numpy.ndarray,
    t: numpy.ndarray,
    f: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute the equivalent of numpy.where.

    Args:
        c (numpy.ndarray): Condition operand.
        t (numpy.ndarray): True operand.
        f (numpy.ndarray): False operand.

    Returns:
        numpy.ndarray: numpy.where(c, t, f)

    """
    return (numpy_where_body(c, t, f),)


def numpy_add(
    a: numpy.ndarray,
    b: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute add in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Add-13

    Args:
        a (numpy.ndarray): First operand.
        b (numpy.ndarray): Second operand.

    Returns:
        Tuple[numpy.ndarray]: Result, has same element type as two inputs
    """
    return (a + b,)


# input, min and max are Python built-in but we need to match the ONNX naming, ignore the lint
# pylint: disable=redefined-builtin
@onnx_func_raw_args("min", "max")
def numpy_clip(a: numpy.ndarray, min=None, max=None) -> Tuple[numpy.ndarray]:
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
        transA (int): Whether A should be transposed. The type is kept as int as it is the
            type used by ONNX and it can easily be interpreted by Python as a boolean.
            Defaults to 0.
        transB (int): Whether B should be transposed. The type is kept as int as it is the
            type used by ONNX and it can easily be interpreted by Python as a boolean.
            Defaults to 0.

    Returns:
        Tuple[numpy.ndarray]: The tuple containing the result tensor
    """
    # If alpha and beta are integer, apply the int type for Concrete to see they are integers
    processed_alpha = int(alpha) if round(alpha) == alpha else alpha
    processed_beta = int(beta) if round(beta) == beta else beta

    a_prime = numpy.transpose(a) if transA else a
    b_prime = numpy.transpose(b) if transB else b
    c_prime: Union[numpy.ndarray, float] = c if c is not None else 0

    # Do
    #
    #       y = processed_alpha * numpy.matmul(a_prime, b_prime) + processed_beta * c_prime
    #
    # in an efficient way, i.e., to make tracing directly optimized, without expecting any opt from
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


def numpy_matmul(
    a: numpy.ndarray,
    b: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute matmul in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#MatMul-13

    Args:
        a (numpy.ndarray): N-dimensional matrix A
        b (numpy.ndarray): N-dimensional matrix B

    Returns:
        Tuple[numpy.ndarray]: Matrix multiply results from A * B
    """
    return (numpy.matmul(a, b),)


def numpy_relu(
    x: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute relu in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Relu-14

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.maximum(x, 0),)


def numpy_sigmoid(
    x: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
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
        axis (None, int, tuple of int): Axis or axes along which a softmax's sum is performed. If
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


def numpy_cos(
    x: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute cos in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Cos-7

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.cos(x),)  # pragma: no cover


def numpy_cosh(
    x: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute cosh in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Cosh-9

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.cosh(x),)  # pragma: no cover


def numpy_sin(
    x: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute sin in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sin-7

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.sin(x),)  # pragma: no cover


def numpy_sinh(
    x: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute sinh in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sinh-9

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.sinh(x),)  # pragma: no cover


def numpy_tan(
    x: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute tan in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Tan-7

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.tan(x),)  # pragma: no cover


def numpy_tanh(
    x: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute tanh in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Tanh-13

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.tanh(x),)


def numpy_acos(
    x: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute acos in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Acos-7

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.arccos(x),)  # pragma: no cover


def numpy_acosh(
    x: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute acosh in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Acosh-9

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.arccosh(x),)  # pragma: no cover


def numpy_asin(
    x: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute asin in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Asin-7

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.arcsin(x),)  # pragma: no cover


def numpy_asinh(
    x: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute sinh in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Asinh-9

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.arcsinh(x),)  # pragma: no cover


def numpy_atan(
    x: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute atan in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Atan-7

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.arctan(x),)  # pragma: no cover


def numpy_atanh(
    x: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute atanh in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Atanh-9

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.arctanh(x),)  # pragma: no cover


def numpy_elu(x: numpy.ndarray, *, alpha: float = 1) -> Tuple[numpy.ndarray]:
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


def numpy_celu(x: numpy.ndarray, *, alpha: float = 1) -> Tuple[numpy.ndarray]:
    """Compute celu in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Celu-12

    Args:
        x (numpy.ndarray): Input tensor
        alpha (float): Coefficient

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (numpy.maximum(0, x) + numpy.minimum(0, alpha * (numpy.exp(x / alpha) - 1)),)


def numpy_leakyrelu(x: numpy.ndarray, *, alpha: float = 0.01) -> Tuple[numpy.ndarray]:
    """Compute leakyrelu in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#LeakyRelu-6

    Args:
        x (numpy.ndarray): Input tensor
        alpha (float): Coefficient

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return numpy_where(x > 0, x, alpha * x)


def numpy_thresholdedrelu(x: numpy.ndarray, *, alpha: float = 1) -> Tuple[numpy.ndarray]:
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
    x: numpy.ndarray, *, alpha: float = 0.2, beta: float = 0.5
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


def numpy_softplus(
    x: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute softplus in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Softplus-1

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (numpy.log(numpy.exp(x) + 1),)


def numpy_abs(
    x: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute abs in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Abs-13

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (numpy.abs(x),)


def numpy_div(
    a: numpy.ndarray,
    b: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute div in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Div-14

    Args:
        a (numpy.ndarray): Input tensor
        b (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    # Remove the where op once the following issue is explained
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/857
    bp = numpy_where_body(b != 0, b, 1)

    # Check if processing non-encrypted constants.
    # We handle non-encrypted constants differently because integer constants
    # must use `floor_divide`
    if isinstance(a, RawOpOutput) and numpy.issubdtype(a.dtype, numpy.integer):
        return (numpy.floor_divide(a, bp),)

    # This branch may be processing encrypted data or float clear constants that are initializers
    # In FHE for integer values we want floating point behavior that produces TLUs without
    # loss of precision.
    return (numpy.divide(a, bp),)


def numpy_mul(
    a: numpy.ndarray,
    b: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute mul in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Mul-14

    Args:
        a (numpy.ndarray): Input tensor
        b (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (a * b,)


def numpy_sub(
    a: numpy.ndarray,
    b: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute sub in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sub-14

    Args:
        a (numpy.ndarray): Input tensor
        b (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (a - b,)


def numpy_log(
    x: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute log in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Log-13

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    # Epsilon is here to avoid problems with 0 or negative values, which may happen when Concrete
    # creates the table (even if these problematic values would normally never be used)
    epsilon = 10**-8

    return (numpy.log(numpy.maximum(x, epsilon)),)


@onnx_func_raw_args("slope")
def numpy_prelu(
    x: numpy.ndarray,
    slope: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
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


def numpy_erf(
    x: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute erf in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Erf-13

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (univariate(special.erf)(x),)  # pylint: disable=no-member


def numpy_hardswish(
    x: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
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


def numpy_exp(
    x: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute exponential in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Exp-13

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: The exponential of the input tensor computed element-wise
    """

    return (numpy.exp(x),)


def numpy_equal(
    x: numpy.ndarray,
    y: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute equal in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Equal-11

    Args:
        x (numpy.ndarray): Input tensor
        y (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (numpy.equal(x, y),)


def numpy_not(
    x: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute not in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Not-1

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (numpy.logical_not(x),)


def numpy_not_float(
    x: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute not in numpy according to ONNX spec and cast outputs to floats.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Not-1

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return cast_to_float(numpy_not(x))


def numpy_greater(
    x: numpy.ndarray,
    y: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute greater in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Greater-13

    Args:
        x (numpy.ndarray): Input tensor
        y (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (numpy.greater(x, y),)


def numpy_greater_float(
    x: numpy.ndarray,
    y: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute greater in numpy according to ONNX spec and cast outputs to floats.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Greater-13

    Args:
        x (numpy.ndarray): Input tensor
        y (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return cast_to_float(numpy_greater(x, y))


def numpy_greater_or_equal(
    x: numpy.ndarray,
    y: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute greater or equal in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#GreaterOrEqual-12

    Args:
        x (numpy.ndarray): Input tensor
        y (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (numpy.greater_equal(x, y),)


def numpy_greater_or_equal_float(
    x: numpy.ndarray,
    y: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute greater or equal in numpy according to ONNX specs and cast outputs to floats.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#GreaterOrEqual-12

    Args:
        x (numpy.ndarray): Input tensor
        y (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return cast_to_float(numpy_greater_or_equal(x, y))


def numpy_less(
    x: numpy.ndarray,
    y: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute less in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Less-13

    Args:
        x (numpy.ndarray): Input tensor
        y (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (numpy.less(x, y),)


def numpy_less_float(
    x: numpy.ndarray,
    y: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute less in numpy according to ONNX spec and cast outputs to floats.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Less-13

    Args:
        x (numpy.ndarray): Input tensor
        y (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return cast_to_float(numpy_less(x, y))


def numpy_less_or_equal(
    x: numpy.ndarray,
    y: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute less or equal in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#LessOrEqual-12

    Args:
        x (numpy.ndarray): Input tensor
        y (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (numpy.less_equal(x, y),)


def numpy_less_or_equal_float(
    x: numpy.ndarray,
    y: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute less or equal in numpy according to ONNX spec and cast outputs to floats.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#LessOrEqual-12

    Args:
        x (numpy.ndarray): Input tensor
        y (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return cast_to_float(numpy_less_or_equal(x, y))


def numpy_identity(
    x: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute identity in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Identity-14

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (x,)


@onnx_func_raw_args("newshape", "allowzero")
def numpy_reshape(
    x: numpy.ndarray, newshape: numpy.ndarray, *, allowzero=0
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


def numpy_transpose(x: numpy.ndarray, *, perm=None) -> Tuple[numpy.ndarray]:
    """Transpose in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Transpose-13

    Args:
        x (numpy.ndarray): Input tensor
        perm (numpy.ndarray): Permutation of the axes

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.transpose(x, axes=perm),)


@onnx_func_raw_args("b")
def numpy_conv(
    x: numpy.ndarray,
    w: numpy.ndarray,
    b: Optional[numpy.ndarray] = None,
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
        b (Optional[numpy.ndarray]): bias tensor, Shape is (O,). Default to None.
        dilations (Tuple[int, ...]): dilation of the kernel, default 1 on all dimensions.
        group (int): number of convolution groups, can be 1 or a multiple of both (C,) and (O,), so
            that I = C / group. Default to 1.
        kernel_shape (Tuple[int, ...]): shape of the kernel. Should have 2 elements for 2d conv
        pads (Tuple[int, ...]): padding in ONNX format (begin, end) on each axis
        strides (Tuple[int, ...]): stride of the convolution on each axis

    Returns:
        res (numpy.ndarray): a tensor of size (N x OutChannels x OutHeight x OutWidth).
           See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html

    """

    # Convert the inputs to tensors to compute conv using torch
    assert_true(len(kernel_shape) == 2, "The convolution operator currently supports only 2-d")
    assert_true(
        bool(numpy.all(numpy.asarray(dilations) == 1)),
        "The convolution operator in Concrete does not support dilation",
    )

    weight_channels = x.shape[1]
    assert_true(
        w.shape[1] == weight_channels / group,
        f"Expected number of channels in weight to be {weight_channels / group} (C / group). Got "
        f"{w.shape[1]}.",
    )

    assert_true(
        w.shape[0] % group == 0,
        f"Expected number of output O ({w.shape[0]}) to be a multiple of group " f"({group}).",
    )

    # Pad the input if needed
    x_pad = numpy_onnx_pad(x, pads)

    # Compute the torch convolution
    res = cnp_conv(x_pad, w, b, None, strides, dilations, None, group)

    return (res,)


def numpy_avgpool(
    x: numpy.ndarray,
    *,
    ceil_mode: int,
    kernel_shape: Tuple[int, ...],
    pads: Tuple[int, ...] = None,
    strides: Tuple[int, ...] = None,
) -> Tuple[numpy.ndarray]:
    """Compute Average Pooling using Torch.

    Currently supports 2d average pooling with torch semantics. This function is ONNX compatible.

    See: https://github.com/onnx/onnx/blob/main/docs/Operators.md#AveragePool

    Args:
        x (numpy.ndarray): input data (many dtypes are supported). Shape is N x C x H x W for 2d
        ceil_mode (int): ONNX rounding parameter, expected 0 (torch style dimension computation)
        kernel_shape (Tuple[int, ...]): shape of the kernel. Should have 2 elements for 2d conv
        pads (Tuple[int, ...]): padding in ONNX format (begin, end) on each axis
        strides (Tuple[int, ...]): stride of the convolution on each axis

    Returns:
        res (numpy.ndarray): a tensor of size (N x InChannels x OutHeight x OutWidth).
           See https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html

    Raises:
        AssertionError: if the pooling arguments are wrong
    """

    assert_true(len(kernel_shape) == 2, "The average pool operator currently supports only 2-d")

    # For mypy
    assert pads is None or len(pads) == 4

    # For mypy
    assert len(kernel_shape) == 2

    assert strides is None or len(strides) == 2

    # Use default values if the ONNX did not set these parameters
    pads = (0, 0, 0, 0) if pads is None else pads
    strides = (1, 1) if strides is None else strides

    # Compute the average pooling using a grouped convolution (groups = input channels)
    # This means that each slice of the kernel is applied on each input channel respectively
    # We create a kernel full of ones, that sums the values in its support
    n_in_channels = x.shape[1]
    kernel = numpy.ones(
        (n_in_channels, 1, kernel_shape[0], kernel_shape[1]),
        dtype=numpy.int64,
    )

    norm_const = onnx_avgpool_compute_norm_const(x.shape, kernel_shape, pads, strides, ceil_mode)

    # Pad the input tensor
    pool_pads = compute_onnx_pool_padding(x.shape, kernel_shape, pads, strides, ceil_mode)
    q_input_pad = numpy_onnx_pad(x, pool_pads)

    # Compute the sums of input values for each kernel position
    res = cnp_conv(q_input_pad, kernel, None, [0, 0, 0, 0], strides, None, None, n_in_channels)

    # Compute the average of the input values for each kernel position
    res /= norm_const

    return (res,)


def numpy_maxpool(
    x: numpy.ndarray,
    *,
    kernel_shape: Tuple[int, ...],
    strides: Tuple[int, ...] = None,
    auto_pad: str = "NOTSET",
    pads: Tuple[int, ...] = None,
    dilations: Optional[Union[Tuple[int, ...], List[int]]] = None,
    ceil_mode: int = 0,
    storage_order: int = 0,
) -> Tuple[numpy.ndarray]:
    """Compute Max Pooling using Torch.

    Currently supports 2d max pooling with torch semantics. This function is ONNX compatible.

    See: https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool

    Args:
        x (numpy.ndarray): the input
        kernel_shape (Union[Tuple[int, ...], List[int]]): shape of the kernel
        strides (Optional[Union[Tuple[int, ...], List[int]]]): stride along each spatial axis
            set to 1 along each spatial axis if not set
        auto_pad (str): padding strategy, default = "NOTSET"
        pads (Optional[Union[Tuple[int, ...], List[int]]]): padding for the beginning and ending
            along each spatial axis (D1_begin, D2_begin, ..., D1_end, D2_end, ...)
            set to 0 along each spatial axis if not set
        dilations (Optional[Union[Tuple[int, ...], List[int]]]): dilation along each spatial axis
            set to 1 along each spatial axis if not set
        ceil_mode (int): ceiling mode, default = 1
        storage_order (int): storage order, 0 for row major, 1 for column major, default = 0

    Returns:
        res (numpy.ndarray): a tensor of size (N x InChannels x OutHeight x OutWidth).
           See https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html

    """

    assert_true(len(kernel_shape) == 2, "The max pool operator currently supports only 2-d")

    # For mypy
    assert pads is None or len(pads) == 4

    # For mypy
    assert len(kernel_shape) == 2

    assert strides is None or len(strides) == 2

    # Use default values if the ONNX did not set these parameters
    pads = (0, 0, 0, 0) if pads is None else pads
    strides = (1, 1) if strides is None else strides

    # Pad the input tensor
    pool_pads = compute_onnx_pool_padding(x.shape, kernel_shape, pads, strides, ceil_mode)
    q_input_pad = numpy_onnx_pad(x, pool_pads)

    fake_pads = [0] * len(pads)
    res = cnp_maxpool(
        q_input_pad,
        kernel_shape=kernel_shape,
        strides=strides,
        auto_pad=auto_pad,
        pads=fake_pads,
        dilations=dilations,
        ceil_mode=ceil_mode,
        storage_order=storage_order,
    )

    return (res,)


@onnx_func_raw_args("pads", "constant_value")
def numpy_pad(
    data: numpy.ndarray,
    pads: numpy.ndarray,
    constant_value: Union[numpy.ndarray, None] = None,
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

    assert_true(mode == "constant", "Padding only supported with a constant pad value")
    assert_true(
        constant_value is None or constant_value == 0, "Pad only accepts a constant padding with 0s"
    )

    # Pad the input if needed
    x_pad = numpy_onnx_pad(data, tuple(pads))

    return (x_pad,)


def numpy_cast(data: numpy.ndarray, *, to: int) -> Tuple[numpy.ndarray]:
    """Execute ONNX cast in Numpy.

    For traced values during compilation, it supports only booleans, which are converted to float.
    For raw values (used in constant folding or shape computations), any cast is allowed.

    See: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cast

    Args:
        data (numpy.ndarray): Input encrypted tensor
        to (int): integer value of the onnx.TensorProto DataType enum

    Returns:
        result (numpy.ndarray): a tensor with the required data type
    """
    # For raw values, any cast is fine
    if isinstance(data, RawOpOutput):
        return (data.astype(onnx.helper.tensor_dtype_to_np_dtype(to)).view(RawOpOutput),)

    assert_true(to == onnx.TensorProto.BOOL)

    # Will be used for traced values
    return (data.astype(numpy.float64),)


def numpy_batchnorm(
    x: numpy.ndarray,
    scale: numpy.ndarray,
    bias: numpy.ndarray,
    input_mean: numpy.ndarray,
    input_var: numpy.ndarray,
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


def numpy_flatten(x: numpy.ndarray, *, axis: int = 1) -> Tuple[numpy.ndarray]:
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


def numpy_or(
    a: numpy.ndarray,
    b: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
    """Compute or in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Or-7

    Args:
        a (numpy.ndarray): Input tensor
        b (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.logical_or(a, b),)


def numpy_or_float(
    a: numpy.ndarray,
    b: numpy.ndarray,
) -> Tuple[numpy.ndarray]:
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


# noop_with_empty_axes has a strange definition as its description does not seem to exactly match
# the one given for the axes parameter. Moreover, neither Torch or Numpy handles such a feature.
# However, it looks like it is used to indicate that a sum with an empty axes parameter should be
# handled as an identity operator, which is consistent with Numpy's inherent behavior. For this
# particular reason, we decided not to consider it in the following definition.
# pylint: disable=unused-argument
@onnx_func_raw_args("axes")
def numpy_reduce_sum(
    a: numpy.ndarray,
    axes: Optional[numpy.ndarray] = None,
    *,
    keepdims: int = 1,
    noop_with_empty_axes: int = 0,
) -> Tuple[numpy.ndarray]:
    """Compute ReduceSum in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Operators.md#ReduceSum

    Args:
        a (numpy.ndarray): Input tensor whose elements to sum.
        axes (Optional[Union[numpy.ndarray, tuple]]): Array or tuple of integers along which to
            reduce. The default is to reduce over all the dimensions of the input tensor if
            'noop_with_empty_axes' is false, else act as an Identity op when 'noop_with_empty_axes'
            is true. Accepted range is [-r, r-1] where r = rank(data). Default to None.
        keepdims (int): Keep the reduced dimension or not, 1 means keeping the
            input dimension, 0 will reduce it along the given axis. Default to 1.
        noop_with_empty_axes (int): Defines behavior if 'axes' is empty or set to None.
            Default behavior with 0 is to reduce all axes. When axes is empty and this
            attribute is set to true 1, input tensor will not be reduced, and the output
            tensor would be equivalent to input tensor. Default to 0.

    Returns:
        numpy.ndarray: Output reduced tensor.
    """

    assert_true(keepdims in [0, 1], f"keepdims parameter should either be 0 or 1. Got {keepdims}")

    assert_true(
        axes is None or isinstance(axes, (numpy.ndarray, tuple)),
        f"axes parameter should either be None, a Numpy array or a tuple. Got {type(axes)}",
    )

    # Numpy's axis parameter only handles tuple of integers (or None) as input while ONNX's axes
    # parameter is an array (or None)
    axis = tuple(axes) if axes is not None else None

    # Numpy's keepdims parameter is a boolean while ONNX's one is an int (0 or 1). Even though
    # Python handles them equivalently, we need to manually convert it as mypy doesn't accept this
    # type difference
    # Find a way to make axis of type Union[SupportsIndex, Sequence[SupportsIndex]
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2050
    return (numpy.sum(a, axis=axis, keepdims=bool(keepdims)),)  # type: ignore


@onnx_func_raw_args("scale", "zero_point", "bit_width")
def numpy_brevitas_quant(
    x: numpy.ndarray,
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
            e.g., [-2**n_bits-1 .. 2**n_bits-1] (default 0),

    Returns:
        result (numpy.ndarray): Tensor with float quantized values
    """

    assert_true(rounding_mode == "ROUND", "Only rounding quantization is supported for Brevitas")
    assert_true(signed in (1, 0), "Signed flag in Brevitas quantizer must be 0/1")
    assert_true(narrow in (1, 0), "Narrow range flag in Brevitas quantizer must be 0/1")

    assert_false(
        signed == 0 and narrow == 1,
        "Can not use narrow range for non-signed Brevitas quantizers",
    )
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


def numpy_floor(x: numpy.ndarray) -> Tuple[numpy.ndarray]:
    """Compute Floor in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Floor-1

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.floor(x),)


def numpy_max(a: numpy.ndarray, b: numpy.ndarray) -> Tuple[numpy.ndarray]:
    """Compute Max in numpy according to ONNX spec.

    Computes the max between the first input and a float constant.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Max-1

    Args:
        a (numpy.ndarray): Input tensor
        b (numpy.ndarray): Constant tensor to compare to the first input

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.maximum(a, b),)


def numpy_min(a: numpy.ndarray, b: numpy.ndarray) -> Tuple[numpy.ndarray]:
    """Compute Min in numpy according to ONNX spec.

    Computes the minimum between the first input and a float constant.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Max-1

    Args:
        a (numpy.ndarray): Input tensor
        b (numpy.ndarray): Constant tensor to compare to the first input

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.minimum(a, b),)


def numpy_sign(x: numpy.ndarray) -> Tuple[numpy.ndarray]:
    """Compute Sign in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sign-9

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.sign(x),)


def numpy_neg(x: numpy.ndarray) -> Tuple[numpy.ndarray]:
    """Compute Negative in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sign-9

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.negative(x),)


def numpy_concatenate(*x: numpy.ndarray, axis: int) -> Tuple[numpy.ndarray]:
    """Apply concatenate in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#concat-13

    Args:
        *x (numpy.ndarray): Input tensors to be concatenated.
        axis (int): Which axis to concat on.

    Returns:
        Tuple[numpy.ndarray]: Output tensor.
    """
    return (numpy.concatenate(x, axis=axis),)


@onnx_func_raw_args("axis")
def numpy_unsqueeze(x: numpy.ndarray, axis: Iterable) -> Tuple[numpy.ndarray]:
    """Apply the unsqueeze operator in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#unsqueeze-13

    Args:
        x (numpy.ndarray): Input tensor.
        axis (Iterable): Tuple of the axis to unsqueeze.

    Returns:
        Tuple[numpy.ndarray]: Output tensor.
    """
    for i, ax in enumerate(sorted(axis)):
        # Add a dimension to x following the axis.
        # The axis must be shifted by the number of dimensions already
        # added to x (thus the ax + i).
        x = numpy.expand_dims(x, axis=ax + i)
    return (x,)


@onnx_func_raw_args("axis")
def numpy_squeeze(x: numpy.ndarray, axis=None) -> Tuple[numpy.ndarray]:
    """Apply the squeeze operator in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#squeeze-13

    Args:
        x (numpy.ndarray): Input tensor.
        axis: Tuple of the axis to squeeze.

    Returns:
        Tuple[numpy.ndarray]: Output tensor.
    """
    return (numpy.squeeze(x, axis=tuple(axis) if axis is not None else axis),)


@onnx_func_raw_args(output_is_raw=True)
def numpy_shape(x: numpy.ndarray) -> Tuple[numpy.ndarray]:
    """Return the shape of the input tensor.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#shape-13

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: outputs an 1D int64 tensor (per ONNX spec)
    """

    # As this op is used in a graph to take the shape of an intermediary encrypted
    # tensor, it marks its output as a raw output
    return (numpy.asarray(x.shape, numpy.int64).view(RawOpOutput),)


@onnx_func_raw_args("shape", "value", output_is_raw=True)
def numpy_constant_of_shape(shape: numpy.ndarray, *, value=0.0) -> Tuple[numpy.ndarray]:
    """Create a constant tensor with a specified shape.

    See https://github.com/onnx/onnx/blob/main/docs/Operators.md#ConstantOfShape

    Args:
        shape (numpy.ndarray): the shape of the constant tensor
        value (Optional[Any]): the constant value

    Returns:
        result(Tuple[numpy.ndarray]): the constant tensor
    """

    return (numpy.ones(shape, dtype=numpy.int64) * value,)


@onnx_func_raw_args("starts", "ends", "steps", "axes")
def numpy_slice(
    x: numpy.ndarray,
    starts: numpy.ndarray,
    ends: numpy.ndarray,
    axes: Optional[numpy.ndarray],
    steps: Optional[numpy.ndarray],
) -> Tuple[numpy.ndarray]:
    """Slice the input according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#slice-13

    Args:
        x (numpy.ndarray): input tensor to slice
        starts (numpy.ndarray): the starting indices, one for each axis to slice
        ends (numpy.ndarray): the ending indices, one for each axis to slice
        axes (numpy.ndarray): the axis indices, default is all axes
        steps (numpy.ndarray): the steps along each axis, defaults to 1

    Returns:
        result (Tuple[numpy.ndarray]): the slice(s) of the input tensor as a new tensor
    """

    slices = []
    if steps is None:
        steps = numpy.ones_like(starts)

    if axes is None:
        axes = numpy.arange(x.ndim)
        assert_true(
            starts.shape[0] == x.ndim and steps.shape[0] == x.ndim,
            "The Starts and Ends parameter of Slice must have the same "
            "number of elements as the number of axes of the input when the axes "
            f"parameter is None. Got starts with {starts.shape[0]} elements, ends "
            f"with {steps.shape[0]} elements, while the input has {x.ndim} dimensions.",
        )
    else:
        # Adjust negative axes
        axes = axes.copy()
        axes[axes < 0] += x.ndim
        assert_true(
            steps.shape[0] == starts.shape[0]
            and ends.shape[0] == starts.shape[0]
            and axes.shape[0] == starts.shape[0],
            "The Starts and Ends parameter of Slice must have the same "
            "number of elements as the axes parameter. Got starts with "
            f"{starts.shape[0]} elements, ends "
            f"with {steps.shape[0]} elements, while the axes had {axes.shape[0]} dimensions.",
        )

    # All negative values in starts[i] and ends[i] have dims[axes[i]] added to them,
    # where dims are the dimensions of input. Then start[axes[i]] is the adjusted starts[i]
    # is clamped into the range [0, dims[axes[i]]] for positive stepping and
    # [0, dims[axes[i]]-1] for negative stepping.

    # The clamping for the adjusted ends[i] depends on the sign of steps[i] and must
    # accommodate copying 0 through dims[axes[i]] elements,
    # so for positive stepping end[axes[i]] is clamped to [0, dims[axes[i]]],
    # while for negative stepping it is clamped to [-1, dims[axes[i]]-1].

    starts = starts.copy()
    ends = ends.copy()
    for k in range(starts.size):
        if starts[k] < 0:
            starts[k] += x.shape[axes[k]]
        if ends[k] < 0:
            ends[k] += x.shape[axes[k]]

        if steps[k] < 0:
            starts[k] = numpy.clip(starts[k], -x.shape[axes[k]] - 1, -1)
            ends[k] = numpy.clip(ends[k], -x.shape[axes[k]] - 1, -1)
        else:
            starts[k] = numpy.clip(starts[k], 0, x.shape[axes[k]] - 1)
            ends[k] = numpy.clip(ends[k], 0, x.shape[axes[k]])

    # Check there are no duplicates
    assert_true(
        len(numpy.unique(axes)) == len(axes), "Axes parameter to Slice contained duplicates"
    )

    # Initialize slices to take the whole input tensor
    slices = [slice(0, int(x.shape[axis]), 1) for axis in range(x.ndim)]

    for idx, axis in enumerate(axes):
        slices[axis] = slice(int(starts[idx]), int(ends[idx]), int(steps[idx]))

    return (x[tuple(slices)],)


@onnx_func_raw_args("indices", "axis")
def numpy_gather(
    x: numpy.ndarray, indices: numpy.ndarray, *, axis: int = 0
) -> Tuple[numpy.ndarray]:
    """Gather indices according to ONNX spec.

    Args:
        x (numpy.ndarray): input tensor to slice
        indices (numpy.ndarray): the indices at which to extract values
        axis (int): the axis along which to extract values (defaults to 0)

    Returns:
        result (Tuple[numpy.ndarray]): the values gathered from the input tensor as a new tensor
    """
    # Support both negative and positive axis
    axis = axis % x.ndim

    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3605
    # Convert indices to a list
    indices_list = indices.tolist()

    # Create a tuple of slices for all dimensions except the specified axis
    slices = tuple(slice(None) if i != axis else indices_list for i in range(x.ndim))

    return (x[slices],)
