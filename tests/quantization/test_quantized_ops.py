"""Tests for the quantized ONNX ops."""

from typing import Tuple, Union

import numpy
import pytest

from concrete.ml.quantization import QuantizedArray
from concrete.ml.quantization.quantized_ops import (
    OPS_W_ATTRIBUTES,
    OPS_WO_ATTRIBUTES,
    QuantizedClip,
    QuantizedOp,
)

N_BITS_ATOL_TUPLE_LIST = [
    (32, 10 ** -2),
    (28, 10 ** -2),
    (20, 10 ** -2),
    (16, 10 ** -1),
    (8, 10 ** -0),
    (5, 10 ** -0),
]


@pytest.mark.parametrize(
    "n_bits, atol",
    [pytest.param(n_bits, atol) for n_bits, atol in N_BITS_ATOL_TUPLE_LIST],
)
@pytest.mark.parametrize(
    "input_range",
    [pytest.param((-1, 1)), pytest.param((-2, 2)), pytest.param((-10, 10)), pytest.param((0, 20))],
)
@pytest.mark.parametrize(
    "input_shape",
    [pytest.param((10, 40, 20)), pytest.param((100, 400))],
)
@pytest.mark.parametrize(
    "quantized_op_type",
    sorted(OPS_WO_ATTRIBUTES, key=lambda x: x.__name__),  # type: ignore # satisfy mypy for __name__
)
@pytest.mark.parametrize("is_signed", [pytest.param(True), pytest.param(False)])
def test_univariate_ops_no_attrs(
    quantized_op_type: QuantizedOp,
    input_shape: Tuple[int, ...],
    input_range: Tuple[int, int],
    n_bits: int,
    atol: float,
    is_signed: bool,
):
    """Test activation functions."""
    values = numpy.random.uniform(input_range[0], input_range[1], size=input_shape)
    q_inputs = QuantizedArray(n_bits, values, is_signed)
    quantized_op = quantized_op_type(n_bits)
    expected_output = quantized_op.calibrate(values)
    q_output = quantized_op(q_inputs)
    qvalues = q_output.qvalues

    # Quantized values must be contained between 0 and 2**n_bits - 1.
    assert numpy.max(qvalues) <= 2 ** n_bits - 1
    assert numpy.min(qvalues) >= 0

    # Dequantized values must be close to original values
    dequant_values = q_output.dequant()

    # Check that all values are close
    assert numpy.isclose(dequant_values.ravel(), expected_output.ravel(), atol=atol).all()


@pytest.mark.parametrize(
    "n_bits, atol",
    [pytest.param(n_bits, atol) for n_bits, atol in N_BITS_ATOL_TUPLE_LIST],
)
@pytest.mark.parametrize(
    "input_range",
    [pytest.param((-1, 1)), pytest.param((-2, 2)), pytest.param((-10, 10)), pytest.param((0, 20))],
)
@pytest.mark.parametrize(
    "input_shape",
    [pytest.param((10, 40, 20)), pytest.param((100, 400))],
)
@pytest.mark.parametrize("min_max", [(-100, 1), (0, 100), (-2.48, 4.67), (-1, 1), (0, 1)])
@pytest.mark.parametrize("is_signed", [pytest.param(True), pytest.param(False)])
def test_clip_op(
    input_shape: Tuple[int, ...],
    input_range: Tuple[int, int],
    n_bits: int,
    atol: float,
    is_signed: bool,
    min_max: Tuple[Union[int, float], Union[int, float]],
):
    """Test for clip op."""
    values = numpy.random.uniform(input_range[0], input_range[1], size=input_shape)
    q_inputs = QuantizedArray(n_bits, values, is_signed)
    min_, max_ = min_max
    quantized_op = QuantizedClip(n_bits, **{"min": min_, "max": max_})
    expected_output = quantized_op.calibrate(values)
    q_output = quantized_op(q_inputs)
    qvalues = q_output.qvalues

    # Quantized values must be contained between 0 and 2**n_bits - 1.
    assert numpy.max(qvalues) <= 2 ** n_bits - 1
    assert numpy.min(qvalues) >= 0

    # Dequantized values must be close to original values
    dequant_values = q_output.dequant()

    # Check that all values are close
    assert numpy.isclose(dequant_values.ravel(), expected_output.ravel(), atol=atol).all()


def test_all_attribute_ops_were_tested():
    """Defensive test to check the developers added the proper test cases for the quantized ops."""
    # Sanity check: add tests for the missing quantized ops and update to prove you read this line
    # If you can think of a way to make this automatic, please provide a PR!
    currently_tested_ops = {QuantizedClip: test_clip_op}
    assert OPS_W_ATTRIBUTES == currently_tested_ops.keys(), (
        "Missing tests and manual aknowledgement for: "
        f"{', '.join(sorted(cls.__name__ for cls in OPS_W_ATTRIBUTES))}"
    )
