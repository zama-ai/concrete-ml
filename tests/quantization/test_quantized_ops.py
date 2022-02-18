"""Tests for the quantized ONNX ops."""

from functools import partial
from itertools import combinations
from typing import Callable, Tuple, Union

import numpy
import pytest

from concrete.ml.quantization import QuantizedArray
from concrete.ml.quantization.quantized_ops import (
    ALL_QUANTIZED_OPS,
    QuantizedAbs,
    QuantizedAdd,
    QuantizedCelu,
    QuantizedClip,
    QuantizedElu,
    QuantizedExp,
    QuantizedGemm,
    QuantizedHardSigmoid,
    QuantizedLeakyRelu,
    QuantizedLinear,
    QuantizedLog,
    QuantizedMatMul,
    QuantizedOp,
    QuantizedRelu,
    QuantizedSelu,
    QuantizedSigmoid,
    QuantizedSoftplus,
    QuantizedTanh,
)

N_BITS_LIST = [20, 16, 8]

INPUT_RANGES = [
    pytest.param((-1, 1)),
    pytest.param((-2, 2)),
    pytest.param((-10, 10)),
    pytest.param((0, 20)),
]

IS_SIGNED = [pytest.param(True), pytest.param(False)]


@pytest.mark.parametrize(
    "n_bits",
    [pytest.param(n_bits) for n_bits in N_BITS_LIST],
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
        QuantizedHardSigmoid,
        QuantizedLeakyRelu,
        QuantizedElu,
        QuantizedSelu,
        QuantizedCelu,
        QuantizedSoftplus,
    ],
)
@pytest.mark.parametrize("is_signed", IS_SIGNED)
def test_univariate_ops_no_attrs(
    quantized_op_type: QuantizedOp,
    input_shape: Tuple[int, ...],
    input_range: Tuple[int, int],
    n_bits: int,
    is_signed: bool,
    check_r2_score: Callable,
):
    """Test activation functions."""
    values = numpy.random.uniform(input_range[0], input_range[1], size=input_shape)
    q_inputs = QuantizedArray(n_bits, values, is_signed)
    quantized_op = quantized_op_type(n_bits)
    expected_output = quantized_op.calibrate(values)
    q_output = quantized_op(q_inputs)
    qvalues = q_output.qvalues

    # Quantized values must be contained between 0 and 2**n_bits - 1.
    assert numpy.max(qvalues) <= 2**n_bits - 1
    assert numpy.min(qvalues) >= 0

    # Dequantized values must be close to original values
    dequant_values = q_output.dequant()

    # Check that all values are close
    check_r2_score(dequant_values, expected_output)


# TODO: https://github.com/zama-ai/concrete-ml-internal/issues/229
# Manage ranges/improve tests for exponential
@pytest.mark.parametrize(
    "n_bits",
    [pytest.param(n_bits) for n_bits in N_BITS_LIST],
)
@pytest.mark.parametrize(
    "input_range",
    [pytest.param((-1, 1)), pytest.param((-2, 2))],
)
@pytest.mark.parametrize(
    "input_shape",
    [pytest.param((10, 40, 20)), pytest.param((100, 400))],
)
@pytest.mark.parametrize("is_signed", IS_SIGNED)
def test_exp_op(
    input_shape: Tuple[int, ...],
    input_range: Tuple[int, int],
    n_bits: int,
    is_signed: bool,
    check_r2_score: Callable,
):
    """Test activation functions."""
    values = numpy.random.uniform(input_range[0], input_range[1], size=input_shape)
    q_inputs = QuantizedArray(n_bits, values, is_signed)
    quantized_op = QuantizedExp(n_bits)
    expected_output = quantized_op.calibrate(values)
    q_output = quantized_op(q_inputs)
    qvalues = q_output.qvalues

    # Quantized values must be contained between 0 and 2**n_bits - 1.
    assert numpy.max(qvalues) <= 2**n_bits - 1
    assert numpy.min(qvalues) >= 0

    # Dequantized values must be close to original values
    dequant_values = q_output.dequant()

    # Check that all values are close
    check_r2_score(dequant_values, expected_output)


@pytest.mark.parametrize(
    "n_bits",
    [pytest.param(n_bits) for n_bits in N_BITS_LIST],
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
    is_signed: bool,
    cst_inputs: Tuple[Union[int, float], Union[int, float]],
    check_r2_score: Callable,
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
        assert numpy.max(qvalues) <= 2**n_bits - 1
        assert numpy.min(qvalues) >= 0

        # Dequantized values must be close to original values
        dequant_values = q_output.dequant()

        # Check that all values are close
        check_r2_score(dequant_values, expected_output)


ARITH_N_BITS_LIST = [20, 16, 8]


@pytest.mark.parametrize("operator", [QuantizedAdd])
@pytest.mark.parametrize("n_bits", ARITH_N_BITS_LIST)
@pytest.mark.parametrize(
    "params_a, params_b, n_dims",
    [
        pytest.param((-10, 1), (5, 100), 100),
        pytest.param((20, 100), (0, 0.2), 200),
        pytest.param((40, 20), (-10, 500), 300),
        pytest.param((-100, 1), (200, 1), 100),
        pytest.param((0, 0.1), (0, 0.1), 20),
    ],
)
@pytest.mark.parametrize(
    "generator",
    [
        partial(numpy.random.uniform, 0, 1),
        partial(numpy.random.normal, 0, 1),
        partial(numpy.random.gamma, 1, 2),
    ],
)
@pytest.mark.parametrize("is_signed", IS_SIGNED)
def test_all_arith_ops(
    operator: QuantizedOp,
    n_bits: int,
    is_signed: bool,
    params_a: Tuple[float, float],
    params_b: Tuple[float, float],
    n_dims: int,
    generator: Callable,
    check_r2_score: Callable,
    check_float_arrays_equal: Callable,
):
    """Test all quantized arithmetic ops"""

    # Generate inputs with specific distribution
    # But vary the dynamic range and the support of the distributions
    input_0 = generator(size=(n_dims, n_dims)) * params_a[1] + params_a[0]
    input_1 = generator(size=(n_dims, n_dims)) * params_b[1] + params_b[0]

    # Quantize the inputs with n_bits
    q_inputs_0 = QuantizedArray(n_bits, input_0, is_signed=is_signed)
    q_inputs_1 = QuantizedArray(n_bits, input_1, is_signed=is_signed)

    # Create the op with the same n_bits as output
    # Using n_bits is not always desirable in practice as the output could feed into another TLU
    # So using n_bits would waste precision, but we test the worst case scenario here

    # Variable+Variable (V+V) test
    q_op = operator(n_bits)

    # Calibrate the layer
    raw_output_vv = q_op.calibrate(input_0, input_1)

    # Compute the quantized operator result
    quantized_output_vv = q_op(q_inputs_0, q_inputs_1).dequant()

    # Check the R2 of raw output and quantized output
    check_r2_score(raw_output_vv, quantized_output_vv)

    # Variable + Constant test (V+C)
    q_op = operator(n_bits, constant_inputs={"b": q_inputs_1})

    # Calibrate the layer
    raw_output_vc = q_op.calibrate(input_0)

    # Compute the quantized operator result
    quantized_output_vc = q_op(q_inputs_0).dequant()

    # Check the R2 of raw output and quantized output (V+C)
    check_r2_score(raw_output_vc, quantized_output_vc)

    # Constant + Variable test (C+V)
    q_op = operator(n_bits, constant_inputs={"a": q_inputs_0})

    # Calibrate the layer
    raw_output_cv = q_op.calibrate(input_1)

    # Compute the quantized operator result
    quantized_output_cv = q_op(q_inputs_1).dequant()

    # Check the R2 of raw output and quantized output (C+V)
    check_r2_score(raw_output_cv, quantized_output_cv)

    # Check that we get the same results in V+V, V+C and C+V modes
    check_float_arrays_equal(raw_output_vv, raw_output_vc)
    check_float_arrays_equal(raw_output_cv, raw_output_vc)
    check_float_arrays_equal(quantized_output_vc, quantized_output_vv)
    check_float_arrays_equal(quantized_output_cv, quantized_output_vc)


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
@pytest.mark.parametrize(
    "generator",
    [
        partial(numpy.random.uniform, 0, 1),
        partial(numpy.random.normal, 0, 1),
        partial(numpy.random.gamma, 1, 2),
    ],
)
@pytest.mark.parametrize("is_signed", IS_SIGNED)
def test_all_gemm_ops(
    n_bits: int,
    is_signed: bool,
    n_examples: int,
    n_features: int,
    n_neurons: int,
    generator: Callable,
    check_r2_score: Callable,
    check_array_equality: Callable,
):
    """Test for gemm style ops."""

    # Multiply input x weights of sizes (N, K) and (K, M)
    inputs = generator(size=(n_examples, n_features))
    weights = generator(size=(n_features, n_neurons))

    # We can assume uniform distribution for the bias without loss of generality
    bias = numpy.random.uniform(size=(1, n_neurons))

    # Quantize the inputs and weights
    q_inputs = QuantizedArray(n_bits, inputs)
    q_weights = QuantizedArray(n_bits, weights, is_signed)
    q_bias = QuantizedArray(n_bits, bias, is_signed)

    # 1- Test our QuantizedGemm layer
    q_gemm = QuantizedGemm(n_bits, constant_inputs={"b": q_weights, "c": q_bias})
    q_linear = QuantizedLinear(n_bits, q_weights, q_bias)

    # Calibrate the Quantized layer
    expected_gemm_outputs = q_gemm.calibrate(inputs)
    expected_linear_outputs = q_linear.calibrate(inputs)

    actual_gemm_output = q_gemm(q_inputs).dequant()
    actual_linear_output = q_linear(q_inputs).dequant()

    check_r2_score(expected_gemm_outputs, actual_gemm_output)
    check_r2_score(expected_linear_outputs, actual_linear_output)

    # 2- Same test without bias
    q_gemm = QuantizedGemm(n_bits, constant_inputs={"b": q_weights})
    q_linear = QuantizedLinear(n_bits, q_weights)
    q_mm = QuantizedMatMul(n_bits, constant_inputs={"b": q_weights})

    # Calibrate the quantized layers
    expected_gemm_outputs = q_gemm.calibrate(inputs)
    expected_linear_outputs = q_linear.calibrate(inputs)
    expected_mm_outputs = q_mm.calibrate(inputs)

    actual_gemm_output = q_gemm(q_inputs).dequant()
    actual_linear_output = q_linear(q_inputs).dequant()
    actual_mm_output = q_mm(q_inputs).dequant()

    # Now check that the quantized results are close to non quantized
    check_r2_score(expected_gemm_outputs, actual_gemm_output)
    check_r2_score(expected_linear_outputs, actual_linear_output)
    check_r2_score(expected_mm_outputs, actual_mm_output)

    # 3- Same test but with (alpha, beta) = (1, 0)
    q_gemm = QuantizedGemm(n_bits, constant_inputs={"b": q_weights, "c": q_bias}, alpha=1, beta=0)

    # Calibrate the Quantized layer
    expected_gemm_outputs = q_gemm.calibrate(inputs)

    actual_gemm_output = q_gemm(q_inputs).dequant()

    check_r2_score(expected_gemm_outputs, actual_gemm_output)

    # Without a bias, MatMul and Gemm should give the same output
    check_array_equality(actual_mm_output, actual_gemm_output)


def test_all_ops_were_tested():
    """Defensive test to check the developers added the proper test cases for the quantized ops."""
    # Sanity check: add tests for the missing quantized ops and update to prove you read this line
    # If you can think of a way to make this automatic, please provide a PR!
    currently_tested_ops = {
        QuantizedGemm: test_all_gemm_ops,
        QuantizedLinear: test_all_gemm_ops,
        QuantizedMatMul: test_all_gemm_ops,
        QuantizedAdd: test_all_arith_ops,
        QuantizedRelu: test_univariate_ops_no_attrs,
        QuantizedTanh: test_univariate_ops_no_attrs,
        QuantizedSigmoid: test_univariate_ops_no_attrs,
        QuantizedHardSigmoid: test_univariate_ops_no_attrs,
        QuantizedLeakyRelu: test_univariate_ops_no_attrs,
        QuantizedElu: test_univariate_ops_no_attrs,
        QuantizedSelu: test_univariate_ops_no_attrs,
        QuantizedCelu: test_univariate_ops_no_attrs,
        QuantizedSoftplus: test_univariate_ops_no_attrs,
        QuantizedAbs: test_univariate_ops_no_attrs,
        QuantizedLog: test_univariate_ops_no_attrs,
        QuantizedExp: test_exp_op,
        QuantizedClip: test_clip_op,
    }
    assert ALL_QUANTIZED_OPS == currently_tested_ops.keys(), (
        "Missing tests and manual acknowledgement for: "
        f"{', '.join(sorted(cls.__name__ for cls in ALL_QUANTIZED_OPS))}"
    )
