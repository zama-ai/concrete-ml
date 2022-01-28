"""Tests for the quantized ONNX ops."""

from itertools import combinations
from typing import Tuple, Union

import numpy
import pytest

from concrete.ml.quantization import QuantizedArray
from concrete.ml.quantization.quantized_ops import (
    ALL_QUANTIZED_OPS,
    QuantizedClip,
    QuantizedGemm,
    QuantizedOp,
    QuantizedRelu,
    QuantizedSigmoid,
    QuantizedTanh,
)

N_BITS_ATOL_TUPLE_LIST = [
    (32, 10 ** -2),
    (28, 10 ** -2),
    (20, 10 ** -2),
    (16, 10 ** -1),
    (8, 10 ** -0),
    (5, 10 ** -0),
]

INPUT_RANGES = [
    pytest.param((-1, 1)),
    pytest.param((-2, 2)),
    pytest.param((-10, 10)),
    pytest.param((0, 20)),
]

IS_SIGNED = [pytest.param(True), pytest.param(False)]


@pytest.mark.parametrize(
    "n_bits, atol",
    [pytest.param(n_bits, atol) for n_bits, atol in N_BITS_ATOL_TUPLE_LIST],
)
@pytest.mark.parametrize(
    "input_range",
    INPUT_RANGES,
)
@pytest.mark.parametrize(
    "input_shape",
    [pytest.param((10, 40, 20)), pytest.param((100, 400))],
)
@pytest.mark.parametrize(
    "quantized_op_type",
    [
        QuantizedRelu,
        QuantizedTanh,
        QuantizedSigmoid,
    ],
)
@pytest.mark.parametrize("is_signed", IS_SIGNED)
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
@pytest.mark.parametrize("input_range", INPUT_RANGES)
@pytest.mark.parametrize("is_signed", IS_SIGNED)
@pytest.mark.parametrize(
    "input_shape",
    [pytest.param((10, 40, 20)), pytest.param((100, 400))],
)
@pytest.mark.parametrize("cst_inputs", [(-100, 1), (0, 100), (-2.48, 4.67), (-1, 1), (0, 1)])
def test_clip_op(
    input_shape: Tuple[int, ...],
    input_range: Tuple[int, int],
    n_bits: int,
    atol: float,
    is_signed: bool,
    cst_inputs: Tuple[Union[int, float], Union[int, float]],
):
    """Test for clip op."""
    values = numpy.random.uniform(input_range[0], input_range[1], size=input_shape)
    q_inputs = QuantizedArray(n_bits, values, is_signed)

    # This is to easily generate test cases, do not access private properties this way in production
    # code. Class properties are not properly supported in python 3.8 so using this workaround
    # instead.
    input_combinations = combinations(
        QuantizedClip._params_name_to_input_idx.keys(), 2  # pylint: disable=protected-access
    )

    for combination in input_combinations:
        quantized_op = QuantizedClip(n_bits, constant_inputs=dict(zip(combination, cst_inputs)))
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


GEMM_N_BITS_LIST = [20, 16, 8]


@pytest.mark.parametrize("n_bits", GEMM_N_BITS_LIST)
@pytest.mark.parametrize(
    "n_examples, n_features, n_neurons",
    [
        pytest.param(50, 3, 4),
        pytest.param(20, 500, 30),
        pytest.param(200, 300, 50),
        pytest.param(10000, 100, 1),
        pytest.param(10, 20, 1),
    ],
)
@pytest.mark.parametrize("is_signed", IS_SIGNED)
def test_gemm_op(
    n_bits: int,
    is_signed: bool,
    n_examples: int,
    n_features: int,
    n_neurons: int,
):
    """Test for gemm op."""

    inputs = numpy.random.uniform(size=(n_examples, n_features))
    q_inputs = QuantizedArray(n_bits, inputs)

    # shape of weights: (n_features, n_neurons)
    weights = numpy.random.uniform(size=(n_features, n_neurons))
    q_weights = QuantizedArray(n_bits, weights, is_signed)

    bias = numpy.random.uniform(size=(1, n_neurons))
    q_bias = QuantizedArray(n_bits, bias, is_signed)

    # Define our QuantizedLinear layer
    q_linear = QuantizedGemm(n_bits, constant_inputs={"b": q_weights, "c": q_bias})

    # Calibrate the Quantized layer
    expected_outputs = q_linear.calibrate(inputs)

    actual_output = q_linear(q_inputs).dequant()

    assert numpy.isclose(expected_outputs, actual_output, atol=10 ** -0).all()

    # Same test without bias
    q_linear = QuantizedGemm(n_bits, constant_inputs={"b": q_weights})

    # Calibrate the Quantized layer
    expected_outputs = q_linear.calibrate(inputs)
    actual_output = q_linear(q_inputs).dequant()

    assert numpy.isclose(expected_outputs, actual_output, atol=10 ** -0).all()


def test_all_ops_were_tested():
    """Defensive test to check the developers added the proper test cases for the quantized ops."""
    # Sanity check: add tests for the missing quantized ops and update to prove you read this line
    # If you can think of a way to make this automatic, please provide a PR!
    currently_tested_ops = {
        QuantizedGemm: test_gemm_op,
        QuantizedRelu: test_univariate_ops_no_attrs,
        QuantizedTanh: test_univariate_ops_no_attrs,
        QuantizedSigmoid: test_univariate_ops_no_attrs,
        QuantizedClip: test_clip_op,
    }
    assert ALL_QUANTIZED_OPS == currently_tested_ops.keys(), (
        "Missing tests and manual aknowledgement for: "
        f"{', '.join(sorted(cls.__name__ for cls in ALL_QUANTIZED_OPS))}"
    )
