"""Test custom assert functions."""

import numpy
import pytest

from concrete.ml.onnx.ops_impl import numpy_gemm, onnx_func_raw_args


@pytest.mark.parametrize(
    "alpha",
    [
        pytest.param(0),
        pytest.param(1),
        pytest.param(3.5),
    ],
)
@pytest.mark.parametrize(
    "beta",
    [
        pytest.param(0),
        pytest.param(1),
        pytest.param(3.5),
    ],
)
@pytest.mark.parametrize(
    "trans_a,size_a",
    [
        pytest.param(0, (2, 3)),
        pytest.param(1, (3, 2)),
    ],
)
@pytest.mark.parametrize(
    "trans_b,size_b",
    [
        pytest.param(0, (3, 4)),
        pytest.param(1, (4, 3)),
    ],
)
def test_numpy_gemm(alpha, beta, trans_a, size_a, trans_b, size_b):
    """Test numpy_gemm"""
    a = numpy.random.randint(0, 16, size=size_a)
    b = numpy.random.randint(0, 16, size=size_b)
    c = numpy.random.randint(0, 16, size=(2, 4))

    got = numpy_gemm(a, b, c, alpha=alpha, beta=beta, transA=trans_a, transB=trans_b)

    if trans_a:
        a = numpy.transpose(a)
    if trans_b:
        b = numpy.transpose(b)

    expected = (alpha * numpy.matmul(a, b) + beta * c,)

    # Can be a bit different if we have floats
    if isinstance(alpha, float) or isinstance(beta, float):

        assert numpy.allclose(
            got, expected
        ), f"expected {expected}, got {got}, abs diff is {numpy.abs(got - expected)}"
    else:
        assert numpy.array_equal(
            got, expected
        ), f"expected {expected}, got {got}, abs diff is {numpy.abs(got - expected)}"


def test_raw_argument_impl():
    """Test ONNX implementation function semantics."""

    with pytest.raises(AssertionError):

        @onnx_func_raw_args("y")
        def fake_numpy_impl(x):
            return (x,)

    @onnx_func_raw_args("x")
    def fake_numpy_impl2(x):
        return (x,)

    assert isinstance(fake_numpy_impl2.__name__, str)
