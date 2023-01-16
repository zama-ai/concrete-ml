"""Tests for the quantized ONNX ops."""

# The test_all_ops_were_tested needs all the tests that it references to be in a single file
# pylint: disable=too-many-lines

from functools import partial
from typing import Callable, Tuple, Union

import numpy
import onnx
import pytest
import torch

from concrete.ml.common.utils import MAX_BITWIDTH_BACKWARD_COMPATIBLE
from concrete.ml.quantization import QuantizedArray
from concrete.ml.quantization.base_quantized_op import ALL_QUANTIZED_OPS
from concrete.ml.quantization.quantized_ops import (
    QuantizedAbs,
    QuantizedAdd,
    QuantizedAvgPool,
    QuantizedBatchNormalization,
    QuantizedBrevitasQuant,
    QuantizedCast,
    QuantizedCelu,
    QuantizedClip,
    QuantizedConcat,
    QuantizedConv,
    QuantizedDiv,
    QuantizedElu,
    QuantizedErf,
    QuantizedExp,
    QuantizedFlatten,
    QuantizedFloor,
    QuantizedGemm,
    QuantizedGreater,
    QuantizedGreaterOrEqual,
    QuantizedHardSigmoid,
    QuantizedHardSwish,
    QuantizedIdentity,
    QuantizedLeakyRelu,
    QuantizedLess,
    QuantizedLessOrEqual,
    QuantizedLog,
    QuantizedMatMul,
    QuantizedMax,
    QuantizedMaxPool,
    QuantizedMin,
    QuantizedMul,
    QuantizedNeg,
    QuantizedNot,
    QuantizedOp,
    QuantizedOr,
    QuantizedPad,
    QuantizedPow,
    QuantizedPRelu,
    QuantizedReduceSum,
    QuantizedRelu,
    QuantizedReshape,
    QuantizedRound,
    QuantizedSelu,
    QuantizedSigmoid,
    QuantizedSign,
    QuantizedSoftplus,
    QuantizedSub,
    QuantizedTanh,
    QuantizedTranspose,
    QuantizedUnsqueeze,
    QuantizedWhere,
)

N_BITS_LIST = [20, 16, 8]

INPUT_RANGES = [
    pytest.param((-1, 1)),
    pytest.param((-2, 2)),
    pytest.param((-10, 10)),
    pytest.param((0, 20)),
]

IS_SIGNED = [pytest.param(True), pytest.param(False)]

OP_DEBUG_NAME = "Test_"


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
        QuantizedAbs,
        QuantizedLog,
        QuantizedHardSwish,
        QuantizedRound,
        QuantizedErf,
        QuantizedNot,
        QuantizedSign,
        QuantizedNeg,
        QuantizedFloor,
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
    q_inputs = QuantizedArray(n_bits, values, is_signed=is_signed)
    quantized_op = quantized_op_type(n_bits, quantized_op_type.__name__)
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


# Manage ranges/improve tests for exponential
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/229
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
    q_inputs = QuantizedArray(n_bits, values, is_signed=is_signed)
    quantized_op = QuantizedExp(n_bits, OP_DEBUG_NAME + "QuantizedExp")
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
    q_inputs = QuantizedArray(n_bits, values, is_signed=is_signed)

    q_cst_inputs = (numpy.asarray([inp_value]) for inp_value in cst_inputs)
    quantized_op = QuantizedClip(
        n_bits, OP_DEBUG_NAME + "QuantizedClip", constant_inputs=dict(zip([1, 2], q_cst_inputs))
    )
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


@pytest.mark.parametrize(
    "operator, supports_enc_with_enc",
    [
        (QuantizedAdd, True),
        (QuantizedSub, True),
        (QuantizedMul, False),
        (QuantizedPow, False),
        (QuantizedOr, False),
        (QuantizedDiv, False),
        (QuantizedMin, False),
        (QuantizedMax, False),
    ],
)
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
    supports_enc_with_enc: bool,
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

    if operator is QuantizedPow:

        # Positive values for base
        input_0 = numpy.maximum(input_0, 0)

        # Small range for power
        input_1 = numpy.clip(input_0, 0, 5)

    # Quantize the inputs with n_bits
    q_inputs_0 = QuantizedArray(n_bits, input_0, is_signed=is_signed)
    q_inputs_1 = QuantizedArray(n_bits, input_1, is_signed=is_signed)

    # Create the op with the same n_bits as output
    # Using n_bits is not always desirable in practice as the output could feed into another TLU
    # So using n_bits would waste precision, but we test the worst case scenario here

    if supports_enc_with_enc:
        # Variable+Variable (V+V) test
        q_op = operator(n_bits, operator.__name__, int_input_names={"0", "1"})

        # Calibrate the layer
        raw_output_vv = q_op.calibrate(input_0, input_1)

        # Compute the quantized operator result
        quantized_output_vv = q_op(q_inputs_0, q_inputs_1).dequant()

        # Check the R2 of raw output and quantized output
        check_r2_score(raw_output_vv, quantized_output_vv)
    else:
        with pytest.raises(Exception):
            # Variable+Variable (V+V) test
            q_op = operator(n_bits, operator.__name__, int_input_names={"0", "1"})

    # Variable + Constant test (V+C)
    q_op = operator(
        n_bits, operator.__name__, int_input_names={"0"}, constant_inputs={"b": q_inputs_1}
    )

    # Calibrate the layer
    raw_output_vc = q_op.calibrate(input_0)

    # Compute the quantized operator result
    quantized_output_vc = q_op(q_inputs_0).dequant()

    # Check the R2 of raw output and quantized output (V+C)
    check_r2_score(raw_output_vc, quantized_output_vc)

    # Constant + Variable test (C+V)
    q_op = operator(
        n_bits, operator.__name__, int_input_names={"0"}, constant_inputs={"a": q_inputs_0}
    )

    # Calibrate the layer
    raw_output_cv = q_op.calibrate(input_1)

    # Compute the quantized operator result
    quantized_output_cv = q_op(q_inputs_1).dequant()

    # Check the R2 of raw output and quantized output (C+V)
    check_r2_score(raw_output_cv, quantized_output_cv)

    # Check that we get the same fp32 results in V+V (if supported), V+C and C+V modes
    if supports_enc_with_enc:
        check_float_arrays_equal(raw_output_vv, raw_output_vc)
    check_float_arrays_equal(raw_output_cv, raw_output_vc)

    # Check that V+C and C+V is symmetric (int+float mode)
    check_float_arrays_equal(quantized_output_cv, quantized_output_vc)

    # As V+C and C+V work on float values they will not be exactly equal to
    # the V+V case which works in quantized, we only check R2 for a high bitwidth in this case
    if supports_enc_with_enc:
        check_r2_score(quantized_output_vc, quantized_output_vv)


@pytest.mark.parametrize("n_bits", N_BITS_LIST)
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
@pytest.mark.parametrize("produces_output", [True, False])
def test_all_gemm_ops(
    n_bits: int,
    is_signed: bool,
    produces_output: bool,
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
    q_weights = QuantizedArray(n_bits, weights, is_signed=is_signed)

    # 1- Test our QuantizedGemm layer
    q_gemm = QuantizedGemm(
        n_bits,
        OP_DEBUG_NAME + "QuantizedGemm",
        int_input_names={"0"},
        constant_inputs={"b": q_weights, "c": bias},
    )
    q_gemm.produces_graph_output = produces_output

    # Calibrate the Quantized layer
    expected_gemm_outputs = q_gemm.calibrate(inputs)

    actual_gemm_output = q_gemm(q_inputs).dequant()

    check_r2_score(expected_gemm_outputs, actual_gemm_output)

    # 2- Same test without bias
    q_gemm = QuantizedGemm(
        n_bits,
        OP_DEBUG_NAME + "QuantizedGemm",
        int_input_names={"0"},
        constant_inputs={"b": q_weights},
    )
    q_gemm.produces_graph_output = produces_output

    q_mm = QuantizedMatMul(
        n_bits,
        OP_DEBUG_NAME + "QuantizedMatmul",
        int_input_names={"0"},
        constant_inputs={"b": q_weights},
    )
    q_mm.produces_graph_output = produces_output

    # Calibrate the quantized layers
    expected_gemm_outputs = q_gemm.calibrate(inputs)
    expected_mm_outputs = q_mm.calibrate(inputs)

    actual_gemm_output = q_gemm(q_inputs).dequant()
    actual_mm_output = q_mm(q_inputs).dequant()

    # Now check that the quantized results are close to non quantized
    check_r2_score(expected_gemm_outputs, actual_gemm_output)
    check_r2_score(expected_mm_outputs, actual_mm_output)

    # 3- Same test but with (alpha, beta) = (1, 0)
    q_gemm = QuantizedGemm(
        n_bits,
        OP_DEBUG_NAME + "QuantizedGemm",
        int_input_names={"0"},
        constant_inputs={"b": q_weights, "c": bias},
        alpha=1,
        beta=0,
    )
    q_gemm.produces_graph_output = produces_output

    # Calibrate the Quantized layer
    expected_gemm_outputs = q_gemm.calibrate(inputs)

    actual_gemm_output = q_gemm(q_inputs).dequant()

    check_r2_score(expected_gemm_outputs, actual_gemm_output)

    # Without a bias, MatMul and Gemm should give the same output
    check_array_equality(actual_mm_output, actual_gemm_output)


@pytest.mark.parametrize("n_bits", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("x", [numpy.random.randn(100)])
def test_identity_op(x, n_bits):
    """Tests for the identity op"""
    q_x = QuantizedArray(n_bits=n_bits, values=x)
    qx_bis = QuantizedIdentity(n_bits, OP_DEBUG_NAME + "QuantizedIdentity")(q_x)
    assert numpy.array_equal(qx_bis.qvalues, q_x.qvalues)


@pytest.mark.parametrize("n_bits", [16])
@pytest.mark.parametrize(
    # Convolution parameters: inputs, weights, biases, strides, padding
    # Inputs has size: N (batch) x C (input channels) x H x W
    # Weights has size: O (output channels) x I (input channels) x Kh x Kw
    # Biases has size: O (output channels)
    # Strides and padding have size 2 (padding/stride on y and x)
    # Group is either 1 or a multiple of both C and 0, so that I = C / group
    "params",
    [
        (
            (1, 3, 32, 32),
            4,
            (3, 3, 3, 3),
            3,
            (3,),
            0.01,
            5,
            (2, 2),
            (0, 0, 0, 0),
            1,
        ),
        (
            (10, 1, 16, 16),
            0.2,
            (16, 1, 3, 3),
            0.25,
            (16,),
            5,
            0,
            (1, 1),
            (0, 0, 0, 0),
            1,
        ),
        (
            (2, 32, 4, 4),
            1,
            (3, 32, 2, 2),
            1,
            (3,),
            1,
            0,
            (1, 1),
            (1, 1, 1, 1),
            1,
        ),
        (
            (2, 32, 4, 4),
            1,
            (3, 32, 2, 2),
            1,
            (3,),
            1,
            0,
            (1, 1),
            (1, 1, 1, 1),
            1,
        ),
        (
            (2, 2, 32, 32),
            -1,
            (3, 2, 2, 2),
            1,
            (3,),
            1,
            0,
            (4, 4),
            (7, 1, 7, 1),
            1,
        ),
        (
            (1, 4, 32, 32),
            -1,
            (6, 2, 2, 2),
            1,
            (6,),
            1,
            0,
            (4, 4),
            (7, 1, 7, 1),
            2,
        ),
        (
            (1, 3, 32, 32),
            -1,
            (3, 1, 2, 2),
            1,
            (3,),
            1,
            0,
            (4, 4),
            (7, 1, 7, 1),
            3,
        ),
    ],
)
@pytest.mark.parametrize("produces_output", [True, False])
def test_quantized_conv(params, n_bits, produces_output, check_r2_score, check_float_arrays_equal):
    """Test the quantized convolution operator."""

    # Retrieve arguments
    (
        size_input,
        scale_input,
        size_weights,
        scale_weights,
        size_bias,
        scale_bias,
        offset_bias,
        strides,
        pads,
        group,
    ) = params

    net_input = numpy.random.uniform(size=size_input) * scale_input
    weights = numpy.random.randn(*size_weights) * scale_weights
    biases = numpy.random.uniform(size=size_bias) * scale_bias + offset_bias

    # Create quantized data
    q_input = QuantizedArray(n_bits, net_input, is_signed=False)
    q_weights = QuantizedArray(n_bits, weights, is_signed=True)

    # Create the operator, specifying weights & biases as constants
    q_op = QuantizedConv(
        n_bits,
        OP_DEBUG_NAME + "QuantizedConv",
        int_input_names={"0"},
        constant_inputs={1: q_weights, 2: biases},
        strides=strides,
        pads=pads,
        kernel_shape=(weights.shape[2], weights.shape[3]),
        dilations=(1, 1),
        group=group,
    )
    q_op.produces_graph_output = produces_output

    # Compute the result in floating point
    expected_result = q_op.calibrate(net_input)

    # Compute the reference result

    # Pad the input if needed

    # Torch uses padding  (padding_left,padding_right, padding_top,padding_bottom)
    # While ONNX and Concrete-ML use (padding_top, padding_left, padding_bottom, padding_right)
    tx_pad = torch.nn.functional.pad(
        torch.Tensor(net_input.copy()), (pads[1], pads[3], pads[0], pads[2])
    )

    # Compute the torch convolution
    torch_res = torch.conv2d(
        tx_pad,
        torch.Tensor(weights.copy()),
        torch.Tensor(biases.squeeze().copy()) if biases is not None else None,
        strides,
        groups=group,
    ).numpy()
    check_float_arrays_equal(torch_res, expected_result)

    # Compute the quantized result
    result = q_op(q_input).dequant()

    # The fp32 and quantized results should be very similar when quantization precision is high
    check_r2_score(result, expected_result)


@pytest.mark.parametrize("n_bits", [16])
@pytest.mark.parametrize(
    "params",
    [
        (
            numpy.random.uniform(low=-2.0, high=2.0, size=(1, 1, 32, 32)),
            (3, 3),
            (2, 2),
            (0, 0, 0, 0),
            0,
        ),
        (
            numpy.random.uniform(low=-1.2, high=0.2, size=(10, 1, 16, 16)),
            (2, 2),
            (1, 1),
            (0, 0, 0, 0),
            0,
        ),
        (
            numpy.random.uniform(low=-2.0, high=2.0, size=(2, 32, 4, 4)),
            (2, 2),
            (1, 1),
            (0, 0, 0, 0),
            0,
        ),
        (
            numpy.random.uniform(low=-2.0, high=2.0, size=(2, 32, 4, 4)),
            (2, 4),
            (1, 1),
            (1, 2, 1, 2),
            1,
        ),
        (
            numpy.random.uniform(low=-2.0, high=2.0, size=(2, 32, 4, 4)),
            (2, 4),
            (1, 1),
            (0, 2, 0, 2),
            1,
        ),
        (
            numpy.random.uniform(low=-2.0, high=2.0, size=(2, 32, 5, 5)),
            (3, 3),
            (1, 1),
            (1, 1, 1, 1),
            1,
        ),
        (
            numpy.random.uniform(low=-2.0, high=2.0, size=(2, 1, 7, 5)),
            (5, 1),
            (1, 1),
            (1, 2, 0, 4),
            1,
        ),
        (
            numpy.random.uniform(low=-2.0, high=2.0, size=(1, 1, 16, 16)),
            (2, 2),
            (4, 4),
            (1, 2, 0, 4),
            1,
        ),
    ],
)
@pytest.mark.parametrize("is_signed", [True, False])
def test_quantized_avg_pool(params, n_bits, is_signed, check_r2_score, check_float_arrays_equal):
    """Test the quantized average pool operator."""

    # Retrieve arguments
    net_input, kernel_shape, strides, pads, ceil_mode = params

    # Create quantized data
    q_input = QuantizedArray(n_bits, net_input, is_signed=is_signed)

    q_op = QuantizedAvgPool(
        n_bits,
        OP_DEBUG_NAME + "QuantizedAvgPool",
        strides=strides,
        pads=pads,
        kernel_shape=kernel_shape,
        ceil_mode=ceil_mode,
        input_quant_opts=q_input.quantizer.quant_options,
    )

    # Compute the result in floating point
    expected_result = q_op.calibrate(net_input)

    # Pad the input if needed
    tinputs = torch.Tensor(net_input.copy())

    # Torch uses padding  (padding_left,padding_right, padding_top,padding_bottom)
    # While ONNX and Concrete-ML use (padding_top, padding_left, padding_bottom, padding_right)
    tx_pad = torch.nn.functional.pad(tinputs, (pads[1], pads[3], pads[0], pads[2]))

    # Compute the torch average pool
    bceil_mode = bool(ceil_mode)
    torch_res = torch.nn.functional.avg_pool2d(tx_pad, kernel_shape, strides, 0, bceil_mode).numpy()
    check_float_arrays_equal(torch_res, expected_result)

    # Compute the quantized result
    result = q_op(q_input).dequant()

    # The fp32 and quantized results should be very similar when quantization precision is high
    check_r2_score(expected_result, result)


@pytest.mark.parametrize("n_bits", [16])
@pytest.mark.parametrize(
    "params",
    [
        (
            numpy.random.uniform(low=-4.0, high=4.0, size=(1, 1, 32, 32)),
            (3, 3),
            (2, 2),
            (0, 0, 0, 0),
            1,
            0,
        ),
        (
            numpy.random.uniform(low=-0.2, high=0.2, size=(10, 1, 16, 16)),
            (2, 2),
            (1, 1),
            (0, 0, 0, 0),
            1,
            0,
        ),
        (
            numpy.random.uniform(low=-5.0, high=3.0, size=(2, 32, 4, 4)),
            (2, 2),
            (1, 1),
            (0, 0, 0, 0),
            1,
            0,
        ),
        (
            numpy.random.uniform(low=-1.0, high=1.0, size=(1, 1, 6, 7)),
            (2, 4),
            (1, 1),
            (1, 2, 1, 2),
            2,
            0,
        ),
        (
            numpy.random.uniform(low=0.0, high=1.0, size=(2, 32, 14, 14)),
            (2, 4),
            (1, 1),
            (0, 2, 0, 2),
            2,
            0,
        ),
        (
            numpy.random.uniform(low=0.0, high=1.0, size=(2, 32, 15, 15)),
            (3, 3),
            (1, 1),
            (1, 1, 1, 1),
            3,
            0,
        ),
        (
            numpy.random.uniform(low=-3.0, high=4.0, size=(2, 1, 17, 15)),
            (5, 4),
            (1, 1),
            (1, 2, 1, 2),
            1,
            0,
        ),
        (
            numpy.random.uniform(low=-3.0, high=1.0, size=(1, 1, 16, 16)),
            (5, 5),
            (4, 4),
            (1, 2, 1, 2),
            1,
            0,
        ),
    ],
)
@pytest.mark.parametrize("is_signed", [True, False])
def test_quantized_max_pool(params, n_bits, is_signed, check_r2_score, check_float_arrays_equal):
    """Test the quantized max pool operator."""

    # Retrieve arguments
    net_input, kernel_shape, strides, pads, dilation, ceil_mode = params

    # Create quantized data
    q_input = QuantizedArray(n_bits, net_input, is_signed=is_signed)

    dilations = tuple([dilation] * (len(kernel_shape)))

    q_op = QuantizedMaxPool(
        n_bits,
        OP_DEBUG_NAME + "QuantizedMaxPool",
        dilations=dilations,
        strides=strides,
        pads=pads,
        kernel_shape=kernel_shape,
        ceil_mode=ceil_mode,
        input_quant_opts=q_input.quantizer.quant_options,
    )

    # Compute the result in floating point
    expected_result = q_op.calibrate(net_input)

    # Pad the input if needed
    tinputs = torch.Tensor(net_input.copy())

    # Torch uses padding  (padding_left,padding_right, padding_top,padding_bottom)
    # While ONNX and Concrete-ML use (padding_top, padding_left, padding_bottom, padding_right)
    tx_pad = torch.nn.functional.pad(tinputs, (pads[1], pads[3], pads[0], pads[2]))

    # Compute the torch max pool
    bceil_mode = bool(ceil_mode)

    torch_res = torch.nn.functional.max_pool2d(
        tx_pad, kernel_shape, strides, 0, dilation, bceil_mode
    ).numpy()

    print("Torch")
    print(torch_res)

    print("Expected")
    print(expected_result)

    check_float_arrays_equal(torch_res, expected_result)

    # Compute the quantized result
    result = q_op(q_input).dequant()

    # The fp32 and quantized results should be very similar when quantization precision is high
    check_r2_score(expected_result, result)


def test_quantized_conv_args():
    """Check that conv arguments are validated"""
    n_bits = 2

    weights = numpy.random.uniform(size=(10, 1, 16, 16)) * 0.2
    biases = numpy.random.uniform(size=(16,)) * 5
    q_weights = QuantizedArray(n_bits, weights, is_signed=True)
    q_bias = QuantizedArray(n_bits, biases, is_signed=True)

    args_ok = {
        "strides": (1, 1),
        "pads": (0, 0, 0, 0),
        "kernel_shape": (3, 3),
        "dilations": (1, 1),
    }
    args_and_errors = {
        ("strides", (2, 2, 1)): ".*strides.*",
        ("pads", (0, 0)): ".*(pads|padding).*",
        ("kernel_shape", (3, 1, 2)): ".*2d.*",
        ("dilations", (3, 3, 3)): ".dilation.*",
    }
    for (name, value), error in args_and_errors.items():
        kwargs_op = {**args_ok, **{name: value}}
        with pytest.raises(AssertionError, match=error):
            QuantizedConv(
                n_bits,
                "QuantizedConv",
                int_input_names={"0"},
                constant_inputs={1: q_weights, 2: q_bias},
                **kwargs_op,
            )


def test_quantized_pad():
    """Test the padding operator on quantized values."""

    # First we test that the operator result is the same as the input
    # when the padding is 0
    # This is currently the only supported mode
    data = numpy.random.uniform(size=(1, 1, 32, 32)) * 4
    q_data = QuantizedArray(2, data)

    pads = numpy.asarray([0, 0, 0, 0, 0, 0, 0, 0])
    q_op = QuantizedPad(
        2,
        OP_DEBUG_NAME + "QuantizedPad",
        int_input_names={"0"},
        constant_inputs={1: pads},
        mode="constant",
    )

    pad_value = QuantizedArray(2, numpy.asarray([0]).astype(numpy.float64))

    q_op.calibrate(q_data.values, pad_value.values)

    q_pad_output = q_op(q_data, pad_value)
    assert numpy.array_equal(q_pad_output.qvalues, q_data.qvalues)

    # Test that padding an input tensor is not yet supported, this operation is
    # only a stub for now
    # Change this when we have a real solution for the Pad operator
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/747

    pads_invalid = numpy.asarray([0, 1, 0, 0, 0, 1, 0, 0])
    with pytest.raises(AssertionError):
        QuantizedPad(
            2,
            OP_DEBUG_NAME + "QuantizedPad",
            int_input_names={"0"},
            constant_inputs={1: pads_invalid},
            mode="constant",
        )

    # Now check that we assert when a different padding mode is given
    with pytest.raises(AssertionError):
        QuantizedPad(2, OP_DEBUG_NAME + "QuantizedPad", int_input_names={"0"}, mode="reflect")


@pytest.mark.parametrize("shape", [(1,), (10, 5), (10, 5, 2)])
def test_quantized_reshape(shape):
    """Test quantized reshape."""

    n_bits_reshape = MAX_BITWIDTH_BACKWARD_COMPATIBLE

    num_values = numpy.prod(numpy.asarray(shape))
    data = numpy.arange(num_values).astype(numpy.float32)
    data = data.reshape(shape)

    q_arr0 = QuantizedArray(n_bits_reshape, data)

    new_shape = numpy.asarray((num_values,))
    reshape = QuantizedReshape(
        n_bits_reshape,
        OP_DEBUG_NAME + "QuantizedReshape",
        constant_inputs={1: new_shape},
        input_quant_opts=q_arr0.quantizer.quant_options,
    )

    q_reshaped = reshape(q_arr0)

    assert q_reshaped.quantizer.zero_point == q_arr0.quantizer.zero_point
    assert q_reshaped.quantizer.scale == q_arr0.quantizer.scale
    assert numpy.all(numpy.reshape(q_arr0.qvalues, new_shape) == q_reshaped.qvalues)

    reshape_back = QuantizedReshape(
        n_bits_reshape,
        OP_DEBUG_NAME + "QuantizedReshape",
        constant_inputs={1: numpy.asarray(shape)},
        input_quant_opts=q_arr0.quantizer.quant_options,
    )

    q_arr1 = reshape_back(q_reshaped)
    assert q_arr1.quantizer.zero_point == q_arr0.quantizer.zero_point
    assert q_arr1.quantizer.scale == q_arr0.quantizer.scale
    assert numpy.all(q_arr0.qvalues == q_arr1.qvalues)


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
    [pytest.param((10, 40, 20))],
)
@pytest.mark.parametrize("slope", [[1], numpy.ones((20,))])
@pytest.mark.parametrize("is_signed", IS_SIGNED)
def test_quantized_prelu(n_bits, input_range, input_shape, slope, is_signed, check_r2_score):
    """Test quantized PRelu."""

    values = numpy.random.uniform(input_range[0], input_range[1], size=input_shape)
    q_inputs = QuantizedArray(n_bits, values, is_signed=is_signed)

    q_cst_inputs = numpy.asarray(slope).astype(numpy.float64)

    quantized_op = QuantizedPRelu(
        n_bits, OP_DEBUG_NAME + "QuantizedPRelu", constant_inputs={"slope": q_cst_inputs}
    )
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
    "params",
    [
        (
            numpy.random.uniform(size=(1, 3, 32, 32)) * 4,
            numpy.random.uniform() * 0.7 + 2,
            numpy.random.uniform(),
            numpy.random.uniform(),
        ),
        (
            numpy.random.uniform(size=(1, 32)) * 100 - 50,
            numpy.random.uniform() * 50 - 25,
            0,
            -1,
        ),
        (
            numpy.random.uniform(size=(1024,)),
            numpy.random.uniform(),
            -100,
            100,
        ),
    ],
)
@pytest.mark.parametrize("n_bits", [16])
@pytest.mark.parametrize(
    "comparator", [QuantizedGreater, QuantizedGreaterOrEqual, QuantizedLess, QuantizedLessOrEqual]
)
def test_quantized_comparators_and_where(params, n_bits, comparator, check_r2_score):
    """Test a conditional pattern that is very common in quantization aware training."""
    values, threshold, val_if_true, val_if_false = params

    q_values = QuantizedArray(n_bits, values)
    q_op_comparator = comparator(
        n_bits,
        OP_DEBUG_NAME + comparator.__name__,
        constant_inputs={1: QuantizedArray(n_bits, threshold)},
    )
    q_cast = QuantizedCast(n_bits, OP_DEBUG_NAME + "QuantizedCast", to=onnx.TensorProto.BOOL)
    q_op_where = QuantizedWhere(
        n_bits,
        OP_DEBUG_NAME + "QuantizedWhere",
        constant_inputs={
            1: QuantizedArray(n_bits, float(val_if_true)),
            2: QuantizedArray(n_bits, float(val_if_false)),
        },
    )

    reference_value = q_op_where.calibrate(q_cast.calibrate(q_op_comparator.calibrate(values)))

    q_result = q_op_where(q_cast(q_op_comparator(q_values)))

    result = q_result.dequant()

    check_r2_score(reference_value, result)


@pytest.mark.parametrize(
    "tensor_shape",
    [
        pytest.param((1, 10)),
        pytest.param((5, 10)),
        pytest.param((1, 10, 3, 3)),
        pytest.param((5, 10, 1, 1)),
    ],
)
@pytest.mark.parametrize(
    "n_bits",
    [pytest.param(16)],
)
def test_batch_normalization(tensor_shape, n_bits, check_r2_score):
    """Test that batchnormalization gives good results under quantization."""

    # Generate some data both 1d and 2d
    x = numpy.random.randn(*tensor_shape)

    # Generate random parameters for the normalization, with the right shapes
    mean = QuantizedArray(n_bits, numpy.random.uniform(size=(tensor_shape[1],)))
    var = QuantizedArray(n_bits, numpy.random.uniform(size=(tensor_shape[1],)))
    bias = QuantizedArray(n_bits, numpy.random.uniform(size=(tensor_shape[1],)))
    scale = QuantizedArray(n_bits, numpy.random.uniform(size=(tensor_shape[1],)))

    q_op = QuantizedBatchNormalization(
        n_bits,
        OP_DEBUG_NAME + "QuantizedBatchNormalization",
        constant_inputs={"input_mean": mean, "input_var": var, "scale": scale, "bias": bias},
    )

    # Compute the fp32 results
    raw_result = q_op.calibrate(x)

    # Compute the results when using a quantized array
    # Note that this will actually use the float32 values since BatchNormalization is an op
    # that must be fused to a TLU and has a single encrypted input
    q_x = QuantizedArray(n_bits, x)
    quant_result = q_op(q_x).dequant()

    # Check that all values are close
    check_r2_score(raw_result, quant_result)


@pytest.mark.parametrize(
    "data_generator",
    [
        pytest.param(partial(numpy.random.uniform, 0, 1), id="uniform"),
        pytest.param(partial(numpy.random.normal, 0, 1), id="normal"),
        pytest.param(partial(numpy.random.gamma, 1, 2), id="gamma"),
    ],
)
@pytest.mark.parametrize(
    "keepdims", [pytest.param(keepdims, id=f"keepdims-{keepdims}") for keepdims in [0, 1]]
)
@pytest.mark.parametrize(
    "size, axes, noop_with_empty_axes",
    [
        pytest.param(size, axes, noop, id=f"size-{size}-axes-{axes}-noop-{noop}")
        for (size, axes, noop) in [
            ((1,), (0,), 0),
            ((100, 1), (1,), 0),
            ((100, 10), None, 0),
            ((100, 10), None, 1),
            ((10, 100), (0,), 0),
            ((10, 10, 1000), (2,), 0),
            ((10, 10, 1000), (0, 2), 0),
        ]
    ],
)
@pytest.mark.parametrize(
    "n_bits", [pytest.param(n_bits, id=f"n_bits-{n_bits}") for n_bits in N_BITS_LIST]
)
def test_reduce_sum(
    n_bits, size, data_generator, axes, keepdims, noop_with_empty_axes, check_r2_score
):
    """Test the QuantizedReduceSum operator."""
    # Generate the inputs
    inputs = data_generator(size=(10,) + size)

    # Instantiate the operator
    quantized_reduce_sum = QuantizedReduceSum(
        n_bits,
        OP_DEBUG_NAME + "QuantizedReduceSum",
        constant_inputs={"axes": numpy.array(axes) if axes is not None else None},
        keepdims=keepdims,
        noop_with_empty_axes=noop_with_empty_axes,
    )

    # Calibrate the quantized op and retrieve the expected results
    expected_outputs = quantized_reduce_sum.calibrate(inputs)

    # Retrieve the results computed by the quantized op applied on quantized inputs
    q_inputs = QuantizedArray(n_bits, inputs)
    actual_output = quantized_reduce_sum(q_inputs)
    actual_output = actual_output.dequant()

    check_r2_score(expected_outputs, actual_output)


def test_all_ops_were_tested():
    """Defensive test to check the developers added the proper test cases for the quantized ops."""
    # Sanity check: add tests for the missing quantized ops and update to prove you read this line
    # If you can think of a way to make this automatic, please provide a PR!
    currently_tested_ops = {
        QuantizedGemm: test_all_gemm_ops,
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
        QuantizedSign: test_univariate_ops_no_attrs,
        QuantizedNeg: test_univariate_ops_no_attrs,
        QuantizedExp: test_exp_op,
        QuantizedClip: test_clip_op,
        QuantizedIdentity: test_identity_op,
        QuantizedNot: test_univariate_ops_no_attrs,
        QuantizedConv: test_quantized_conv,
        QuantizedReshape: test_quantized_reshape,
        QuantizedPRelu: test_quantized_prelu,
        QuantizedHardSwish: test_univariate_ops_no_attrs,
        QuantizedAvgPool: test_quantized_avg_pool,
        QuantizedMaxPool: test_quantized_max_pool,
        QuantizedPad: test_quantized_pad,
        QuantizedCast: test_quantized_comparators_and_where,
        QuantizedWhere: test_quantized_comparators_and_where,
        QuantizedGreater: test_quantized_comparators_and_where,
        QuantizedGreaterOrEqual: test_quantized_comparators_and_where,
        QuantizedLess: test_quantized_comparators_and_where,
        QuantizedLessOrEqual: test_quantized_comparators_and_where,
        QuantizedMul: test_all_arith_ops,
        QuantizedSub: test_all_arith_ops,
        QuantizedBatchNormalization: test_batch_normalization,
        QuantizedFlatten: test_quantized_flatten,
        QuantizedRound: test_univariate_ops_no_attrs,
        QuantizedOr: test_all_arith_ops,
        QuantizedDiv: test_all_arith_ops,
        QuantizedPow: test_all_arith_ops,
        QuantizedMax: test_all_arith_ops,
        QuantizedMin: test_all_arith_ops,
        QuantizedReduceSum: test_reduce_sum,
        QuantizedErf: test_univariate_ops_no_attrs,
        QuantizedFloor: test_univariate_ops_no_attrs,
        QuantizedBrevitasQuant: test_brevitas_quant,
        QuantizedTranspose: test_quantized_transpose,
        QuantizedUnsqueeze: test_quantized_unsqueeze,
        QuantizedConcat: test_quantized_concat,
    }
    not_tested = [cls.__name__ for cls in ALL_QUANTIZED_OPS if cls not in currently_tested_ops]
    assert ALL_QUANTIZED_OPS == currently_tested_ops.keys(), (
        "Missing tests and manual acknowledgement for: " f"{', '.join(sorted(not_tested))}"
    )


@pytest.mark.parametrize(
    "input_shape, expected_shape, axis",
    [
        pytest.param((1,), (1,), 0),
        pytest.param((10, 5), (10, 5), 1),
        pytest.param((10, 5, 2), (10, 10), 1),
        pytest.param((10, 5, 2, 2), (10, 20), 1),
        pytest.param((10, 5, 2, 2), (10, 5, 4), 2),
        pytest.param((10, 5, 2, 2), (10, 5, 2, 2), 3),
    ],
)
@pytest.mark.parametrize("is_signed", [True, False])
def test_quantized_flatten(input_shape, expected_shape, axis, is_signed):
    """Test the flatten operator on quantized data."""

    n_bits_reshape = 7

    # Generate data with the desired shape
    num_values = numpy.prod(numpy.asarray(input_shape))
    data = numpy.arange(num_values).astype(numpy.float32)
    data = data.reshape(input_shape)
    q_data = QuantizedArray(n_bits_reshape, data, is_signed=is_signed)

    # Create the operator and calibrate
    flatten_op = QuantizedFlatten(
        n_bits_reshape,
        OP_DEBUG_NAME + "QuantizedFlatten",
        axis=axis,
        input_quant_opts=q_data.quantizer.quant_options,
    )
    flatten_op.calibrate(data)

    # Flatten a quantized array of data
    q_reshaped = flatten_op(q_data)

    # Check that the output calibration parameters did not change as flatten should not change
    # the data
    assert q_reshaped.quantizer.zero_point == q_data.quantizer.zero_point
    assert q_reshaped.quantizer.scale == q_data.quantizer.scale

    # Check that the data was not changed and is in the same order (i.e. no transposition)
    assert numpy.all(q_data.qvalues.ravel() == q_reshaped.qvalues.ravel())

    # Check that the output has the expected shape
    assert numpy.all(q_reshaped.qvalues.shape == expected_shape)


@pytest.mark.parametrize("is_signed", [True, False])
@pytest.mark.parametrize("narrow", [True, False])
def test_brevitas_quant(check_r2_score, is_signed: bool, narrow: bool):
    """Test the brevitas quantization op that produces a QuantizedArray."""

    idx_name = {"scale": 1, "zero_point": 2, "bit_width": 3}

    def create_layer(is_signed, narrow):
        return QuantizedBrevitasQuant(
            7,
            OP_DEBUG_NAME + "QuantizedBrevitasQuant",
            constant_inputs={
                idx_name["scale"]: numpy.random.uniform(0.1, 10),
                idx_name["zero_point"]: float(numpy.random.randint(-10, 10)),
                idx_name["bit_width"]: numpy.random.randint(2, 16),
            },
            rounding_mode="ROUND",
            signed=is_signed,
            narrow=narrow,
        )

    if not is_signed and narrow:
        with pytest.raises(AssertionError, match=r"Can not use narrow range.*"):
            quant = create_layer(1 if is_signed else 0, 1 if narrow else 0)
        return

    quant = create_layer(1 if is_signed else 0, 1 if narrow else 0)

    cinp = {
        1: 1.0,
        2: 0.0,
        3: 7,
    }

    # Verify that "signed" is checked
    with pytest.raises(AssertionError):
        QuantizedBrevitasQuant(
            7,
            OP_DEBUG_NAME + "QuantizedBrevitasQuant",
            constant_inputs=cinp,
            rounding_mode="ROUND",
            signed=5,
            narrow=0,
        )

    # Verify that "rounding_mode" is checked
    with pytest.raises(AssertionError):
        QuantizedBrevitasQuant(
            7,
            OP_DEBUG_NAME + "QuantizedBrevitasQuant",
            constant_inputs=cinp,
            rounding_mode="FLOOR",
            signed=1,
            narrow=0,
        )

    x = numpy.random.randn(100)
    q_data = QuantizedArray(7, x, is_signed=True)

    res_fp32 = quant.calibrate(x)
    res_q = quant(q_data).dequant()
    check_r2_score(res_fp32, res_q)


@pytest.mark.parametrize(
    "shape, axes", [pytest.param((10, 5), [1, 0]), pytest.param((10, 5, 2), [0, 2, 1])]
)
def test_quantized_transpose(shape, axes):
    """Test quantized transpose."""

    n_bits_transpose = MAX_BITWIDTH_BACKWARD_COMPATIBLE

    num_values = numpy.prod(numpy.asarray(shape))
    data = numpy.arange(num_values).astype(numpy.float32)
    data = data.reshape(shape)

    q_arr0 = QuantizedArray(n_bits_transpose, data)

    transpose = QuantizedTranspose(
        n_bits_transpose,
        OP_DEBUG_NAME + "QuantizedTranspose",
        input_quant_opts=q_arr0.quantizer.quant_options,
        perm=axes,
    )

    q_transposed = transpose(q_arr0)

    assert q_transposed.quantizer.zero_point == q_arr0.quantizer.zero_point
    assert q_transposed.quantizer.scale == q_arr0.quantizer.scale
    assert numpy.all(numpy.transpose(q_arr0.qvalues, axes) == q_transposed.qvalues)


@pytest.mark.parametrize(
    "shape1, shape2, axis",
    [pytest.param((10, 5), (10, 5), 1), pytest.param((50, 40), (50, 40), 1)],
)
def test_quantized_concat(shape1, shape2, axis):
    """Test quantized reshape."""

    n_bits_concat_output = MAX_BITWIDTH_BACKWARD_COMPATIBLE
    data = numpy.random.randn(*shape1)
    data_to_concatenate = numpy.random.randn(*shape2)

    q_arr = QuantizedArray(n_bits=n_bits_concat_output, values=data)

    q_arr_to_concat = QuantizedArray(n_bits=n_bits_concat_output, values=data_to_concatenate)

    q_concatenated = QuantizedConcat(
        n_bits_concat_output,
        OP_DEBUG_NAME + "QuantizedConcat",
        input_quant_opts=q_arr.quantizer.quant_options,
        axis=axis,
    )

    with pytest.raises(
        AssertionError,
        match="All inputs must have the same scale and zero_point to be concatenated.",
    ):
        q_result = q_concatenated(q_arr, q_arr_to_concat)

    q_arr_to_concat.quantizer.scale = q_arr.quantizer.scale
    q_arr_to_concat.quantizer.zero_point = q_arr.quantizer.zero_point

    q_result = q_concatenated(q_arr, q_arr_to_concat)

    assert q_result.quantizer.zero_point == q_arr.quantizer.zero_point
    assert q_result.quantizer.scale == q_arr.quantizer.scale
    assert numpy.all(
        numpy.concatenate([q_arr.qvalues, q_arr_to_concat.qvalues], axis) == q_result.qvalues
    )


@pytest.mark.parametrize("shape, axis", [pytest.param((10, 5), [1]), pytest.param((50, 40), [0])])
def test_quantized_unsqueeze(shape, axis):
    """Test quantized reshape."""

    def custom_numpy_unsqueeze(x, axes):
        for axis in axes:
            x = numpy.expand_dims(x, axis=axis)
        return x

    n_bits_concat_output = MAX_BITWIDTH_BACKWARD_COMPATIBLE
    data = numpy.random.randn(*shape)

    q_arr = QuantizedArray(n_bits=n_bits_concat_output, values=data)

    q_unsqueeze = QuantizedUnsqueeze(
        n_bits_concat_output,
        OP_DEBUG_NAME + "QuantizedUnsqueeze",
        input_quant_opts=q_arr.quantizer.quant_options,
        constant_inputs={1: axis},
    )
    q_result = q_unsqueeze(q_arr)

    assert q_result.quantizer.zero_point == q_arr.quantizer.zero_point
    assert q_result.quantizer.scale == q_arr.quantizer.scale
    assert numpy.all(custom_numpy_unsqueeze(q_arr.qvalues, axes=axis) == q_result.qvalues)
