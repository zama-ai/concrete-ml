"""ONNX ops implementation in python + numpy."""

from typing import Optional, Tuple, Union

import numpy

from ..common.debugging import assert_true


def numpy_add(a: numpy.ndarray, b: numpy.ndarray) -> Tuple[numpy.ndarray]:
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
def numpy_clip(input: numpy.ndarray, min=None, max=None) -> Tuple[numpy.ndarray]:
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
        "for the min or max ONNX attributes.",
    )

    return (numpy.clip(input, min, max),)


# pylint: enable=redefined-builtin


def numpy_constant(**kwargs):
    """Return the constant passed as kwarg.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Constant-13

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
def numpy_gemm(
    a: numpy.ndarray,
    b: numpy.ndarray,
    c: Optional[numpy.ndarray] = None,
    alpha: float = 1.0,
    beta: float = 1.0,
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
        c (Optional[numpy.ndarray], optional): Optional input tensor C. If not specified, the
            computation is done as if C is a scalar 0. The shape of C should be unidirectional
            broadcastable to (M, N).
            Defaults to None.
        alpha (float, optional): Scalar multiplier for the product of input tensors A * B.
            Defaults to 1.0.
        beta (float, optional): Scalar multiplier for input tensor C.
            Defaults to 1.0.
        transA (int, optional): Whether A should be transposed. The type is kept as int as it's the
            type used by ONNX and it can easily be interpreted by python as a boolean.
            Defaults to 0.
        transB (int, optional): Whether B should be transposed. The type is kept as int as it's the
            type used by ONNX and it can easily be interpreted by python as a boolean.
            Defaults to 0.

    Returns:
        Tuple[numpy.ndarray]: The tuple containing the result tensor
    """

    a_prime = numpy.transpose(a) if transA else a
    b_prime = numpy.transpose(b) if transB else b
    c_prime: Union[numpy.ndarray, float] = c if c is not None else 0.0

    y = alpha * numpy.matmul(a_prime, b_prime) + beta * c_prime

    return (y,)


def numpy_matmul(a: numpy.ndarray, b: numpy.ndarray) -> Tuple[numpy.ndarray]:
    """Compute matmul in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#MatMul-13

    Args:
        a (numpy.ndarray): N-dimensional matrix A
        b (numpy.ndarray): N-dimensional matrix B

    Returns:
        Tuple[numpy.ndarray]: Matrix multiply results from A * B
    """
    return (numpy.matmul(a, b),)


def numpy_relu(x: numpy.ndarray) -> Tuple[numpy.ndarray]:
    """Compute relu in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Relu-14

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (numpy.maximum(x, 0),)


def numpy_sigmoid(x: numpy.ndarray) -> Tuple[numpy.ndarray]:
    """Compute sigmoid in numpy according to ONNX spec.

    See https://github.com/onnx/onnx/blob/main/docs/Changelog.md#Sigmoid-13

    Args:
        x (numpy.ndarray): Input tensor

    Returns:
        Tuple[numpy.ndarray]: Output tensor
    """
    return (1.0 / (1.0 + numpy.exp(-x)),)
