"""ONNX ops implementation in python + numpy."""

from typing import Optional, Tuple, Union

import numpy
from scipy import special

from ..common.debugging import assert_true


def fake_numpy_where(c: numpy.ndarray, t: numpy.ndarray, f: numpy.ndarray) -> numpy.ndarray:
    """Compute the equivalent of numpy.where.

    Args:
        c (numpy.ndarray): Condition operand.
        t (numpy.ndarray): True operand.
        f (numpy.ndarray): False operand.

    Returns:
        numpy.ndarray: numpy.where(c, t, f)

    # FIXME: can it be improved with a native numpy.where in Concrete Numpy?
    # https://github.com/zama-ai/concrete-numpy-internal/issues/1429
    """
    return c * t + (1 - c) * f


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
def numpy_clip(input: numpy.ndarray, /, min=None, max=None) -> Tuple[numpy.ndarray]:
    """Compute clip in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Clip-13

    Args:
        input (numpy.ndarray): Input tensor whose elements to be clipped.
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

    return (numpy.clip(input, min, max),)


# pylint: enable=redefined-builtin


def numpy_constant(**kwargs):
    """Return the constant passed as kwarg.

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
    # in an efficient way, ie to make tracing directly optimized, without expecting any opt from the
    # compiler here

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

    y = fake_numpy_where(x > 0, x, alpha * (numpy.exp(x) - 1))
    return (y,)


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

    y = fake_numpy_where(x > 0, gamma * x, (gamma * alpha) * (numpy.exp(x) - 1))
    return (y,)


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

    y = fake_numpy_where(x > 0, x, alpha * x)
    return (y,)


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


def numpy_div(x: numpy.ndarray, y: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute div in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Div-14

    Args:
        x (numpy.ndarray): Input tensor
        y (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (x / y,)


def numpy_mul(x: numpy.ndarray, y: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute mul in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Mul-14

    Args:
        x (numpy.ndarray): Input tensor
        y (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (x * y,)


def numpy_sub(x: numpy.ndarray, y: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute sub in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sub-14

    Args:
        x (numpy.ndarray): Input tensor
        y (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (x - y,)


def numpy_log(x: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute log in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Log-13

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (numpy.log(x),)


def numpy_erf(x: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute erf in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Erf-13

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (special.erf(x),)  # pylint: disable=no-member


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


def numpy_identity(x: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute identity in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Identity-14

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """

    return (x,)


def numpy_reshape(x: numpy.ndarray, newshape: numpy.ndarray, /) -> Tuple[numpy.ndarray]:
    """Compute reshqpe in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Reshape-13

    Args:
        x (numpy.ndarray): Input tensor
        newshape (numpy.ndarray): New shape

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.reshape(x, newshape),)


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
