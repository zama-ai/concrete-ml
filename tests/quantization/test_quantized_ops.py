"""Tests for the quantized ONNX ops."""

# The test_all_ops_were_tested needs all the tests that it references to be in a single file
# pylint: disable=too-many-lines

import io
from collections import OrderedDict
from functools import partial
from typing import Callable, Optional, Tuple, Union

import numpy
import onnx
import onnx.checker
import onnx.helper
import onnx.mapping
import onnxruntime as ort
import pytest
import torch

from concrete.ml.common.utils import MAX_BITWIDTH_BACKWARD_COMPATIBLE
from concrete.ml.pytest.utils import check_serialization, values_are_equal
from concrete.ml.quantization import QuantizedArray
from concrete.ml.quantization.base_quantized_op import ALL_QUANTIZED_OPS
from concrete.ml.quantization.quantized_ops import (
    ONNXConstantOfShape,
    ONNXGather,
    ONNXShape,
    ONNXSlice,
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
    QuantizedEqual,
    QuantizedErf,
    QuantizedExp,
    QuantizedExpand,
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
    QuantizedSqueeze,
    QuantizedSub,
    QuantizedTanh,
    QuantizedTranspose,
    QuantizedUnfold,
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


def quantized_op_results_are_equal(
    quantized_op_1: QuantizedOp,
    quantized_op_2: QuantizedOp,
    q_input: Optional[numpy.ndarray] = None,
):
    """Check if two quantized operator instances are equal.

    Args:
        quantized_op_1 (QuantizedOp): The first quantized operator object to consider.
        quantized_op_2 (QuantizedOp): The second quantized operator object to consider.
        x (numpy.ndarray): The input to use for running the call.

    Returns:
        bool: If both instances are equal.
    """
    if q_input is None:
        value_1, value_2 = quantized_op_1(), quantized_op_2()
    elif isinstance(q_input, tuple):
        value_1, value_2 = quantized_op_1(*q_input), quantized_op_2(*q_input)
    else:
        value_1, value_2 = quantized_op_1(q_input), quantized_op_2(q_input)

    return values_are_equal(value_1, value_2)


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

    # De-quantized values must be close to original values
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

    # De-quantized values must be close to original values
    dequant_values = q_output.dequant()

    # Check that all values are close
    check_r2_score(dequant_values, expected_output)

    # Test the serialization of QuantizedExp
    check_serialization(
        quantized_op,
        QuantizedExp,
        equal_method=partial(quantized_op_results_are_equal, q_input=q_inputs),
    )


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

    # De-quantized values must be close to original values
    dequant_values = q_output.dequant()

    # Check that all values are close
    check_r2_score(dequant_values, expected_output)

    # Test the serialization of QuantizedClip
    check_serialization(
        quantized_op,
        QuantizedClip,
        equal_method=partial(quantized_op_results_are_equal, q_input=q_inputs),
    )


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
        pytest.param((-10, 1), (5, 10), 10),
        pytest.param((20, 10), (0, 0.2), 20),
        pytest.param((40, 2), (-10, 50), 30),
        pytest.param((-10, 1), (20, 1), 10),
        pytest.param((0, 0.1), (0, 0.1), 5),
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
    check_float_array_equal: Callable,
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

        # Test the serialization of all arithmetic operators that supports enc x enc
        check_serialization(
            q_op,
            operator,
            equal_method=partial(quantized_op_results_are_equal, q_input=(q_inputs_0, q_inputs_1)),
        )

    else:
        with pytest.raises(
            AssertionError, match="Do not support this type of operation between encrypted tensors"
        ):
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
        check_float_array_equal(raw_output_vv, raw_output_vc)
    check_float_array_equal(raw_output_cv, raw_output_vc)

    # Check that V+C and C+V is symmetric (int+float mode)
    check_float_array_equal(quantized_output_cv, quantized_output_vc)

    # As V+C and C+V work on float values they will not be exactly equal to
    # the V+V case which works in quantized, we only check R2 for a high bit-width in this case
    if supports_enc_with_enc:
        check_r2_score(quantized_output_vc, quantized_output_vv)

    # Test the serialization of all arithmetic operators
    check_serialization(
        q_op, operator, equal_method=partial(quantized_op_results_are_equal, q_input=q_inputs_1)
    )


@pytest.mark.parametrize("n_bits", N_BITS_LIST)
@pytest.mark.parametrize("batch_size", [None, 10, 100])
@pytest.mark.parametrize(
    "n_examples, n_features, n_neurons",
    [
        pytest.param(50, 3, 4),
        pytest.param(20, 50, 30),
        pytest.param(10, 20, 1),
        pytest.param(10, 100, 10),
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
    batch_size: int,
    n_examples: int,
    n_features: int,
    n_neurons: int,
    generator: Callable,
    check_r2_score: Callable,
    check_array_equal: Callable,
):
    """Test for gemm style ops."""

    if batch_size is None:
        inputs_shape = (n_examples, n_features)
    else:
        inputs_shape = (batch_size, n_examples, n_features)
    inputs = generator(size=inputs_shape)

    weights_shape = (n_features, n_neurons)
    weights = generator(size=weights_shape)

    # We can assume uniform distribution for the bias without loss of generality
    bias = numpy.random.uniform(size=(1, n_neurons))

    # Quantize the inputs and weights
    q_inputs = QuantizedArray(n_bits, inputs)
    q_weights = QuantizedArray(n_bits, weights, is_signed=is_signed)
    q_bias = QuantizedArray(n_bits, bias)

    # 1- Test our QuantizedGemm layer
    q_gemm = QuantizedGemm(
        n_bits,
        OP_DEBUG_NAME + "QuantizedGemm",
        int_input_names={"0"},
        constant_inputs={"b": q_weights, "c": q_bias},
    )
    q_gemm.produces_graph_output = produces_output

    # Calibrate the Quantized layer
    expected_gemm_outputs = q_gemm.calibrate(inputs)

    actual_gemm_output = q_gemm(q_inputs).dequant()

    check_r2_score(expected_gemm_outputs, actual_gemm_output)

    # Test the serialization of QuantizedGemm
    check_serialization(
        q_gemm,
        QuantizedGemm,
        equal_method=partial(quantized_op_results_are_equal, q_input=q_inputs),
    )

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

    # Test the serialization of QuantizedGemm without bias
    check_serialization(
        q_gemm,
        QuantizedGemm,
        equal_method=partial(quantized_op_results_are_equal, q_input=q_inputs),
    )

    # 3- Same test but with (alpha, beta) = (1, 0)
    q_gemm = QuantizedGemm(
        n_bits,
        OP_DEBUG_NAME + "QuantizedGemm",
        int_input_names={"0"},
        constant_inputs={"b": q_weights, "c": q_bias},
        alpha=1,
        beta=0,
    )
    q_gemm.produces_graph_output = produces_output

    # Calibrate the Quantized layer
    expected_gemm_outputs = q_gemm.calibrate(inputs)

    actual_gemm_output = q_gemm(q_inputs).dequant()

    check_r2_score(expected_gemm_outputs, actual_gemm_output)

    # Without a bias, MatMul and Gemm should give the same output
    check_array_equal(actual_mm_output, actual_gemm_output)

    # Test the serialization of QuantizedGemm with (alpha, beta) = (1, 0)
    check_serialization(
        q_gemm,
        QuantizedGemm,
        equal_method=partial(quantized_op_results_are_equal, q_input=q_inputs),
    )

    # 4- Test with 2 int_input_names and empty constant_inputs (encrypted gemm)
    q_gemm = QuantizedGemm(
        n_bits,
        OP_DEBUG_NAME + "QuantizedGemm",
        int_input_names={"0", "1"},
        constant_inputs={},
    )
    q_gemm.produces_graph_output = produces_output
    expected_gemm_outputs = q_gemm.calibrate(*(inputs, weights))
    actual_gemm_output = q_gemm(q_inputs, q_weights).dequant()
    check_r2_score(expected_gemm_outputs, actual_gemm_output)

    check_serialization(
        q_gemm,
        QuantizedGemm,
        equal_method=partial(quantized_op_results_are_equal, q_input=(q_inputs, q_weights)),
    )


@pytest.mark.parametrize("n_bits", [1, 2, 3, 4, 5, 6])
@pytest.mark.parametrize("x", [numpy.random.randn(100)])
def test_identity_op(x, n_bits):
    """Tests for the identity op"""
    q_x = QuantizedArray(n_bits=n_bits, values=x)
    quantized_identity = QuantizedIdentity(n_bits, OP_DEBUG_NAME + "QuantizedIdentity")
    qx_bis = quantized_identity(q_x)
    assert numpy.array_equal(qx_bis.qvalues, q_x.qvalues)

    # Test the serialization of QuantizedIdentity
    check_serialization(
        quantized_identity,
        QuantizedIdentity,
        equal_method=partial(quantized_op_results_are_equal, q_input=q_x),
    )


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
@pytest.mark.parametrize("produces_output", [True, False], ids=["produces_output", ""])
@pytest.mark.parametrize("is_conv1d", [True, False], ids=["is_conv1d", "is_conv2d"])
# @pytest.mark.parametrize("is_conv1d", [True], ids=["is_conv1d"])
# pylint: disable-next=too-many-locals
def test_quantized_conv(
    params, n_bits, produces_output, is_conv1d, check_r2_score, check_float_array_equal
):
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

    # If testing the conv1d operator, make the parameters represent 1D inputs
    if is_conv1d:
        size_input = size_input[:3]
        size_weights = size_weights[:3]
        strides = strides[:1]
        pads = pads[:2]
        dilations = (1,)
        conv_torch_op = torch.conv1d

    else:
        dilations = (1, 1)  # type: ignore[assignment]
        conv_torch_op = torch.conv2d

    net_input = numpy.random.uniform(size=size_input) * scale_input
    weights = numpy.random.randn(*size_weights) * scale_weights
    biases = numpy.random.uniform(size=size_bias) * scale_bias + offset_bias

    # Create quantized data
    q_input = QuantizedArray(n_bits, net_input, is_signed=False)
    q_weights = QuantizedArray(n_bits, weights, is_signed=True)
    q_bias = QuantizedArray(n_bits, biases)

    # Create the operator, specifying weights & biases as constants
    q_op = QuantizedConv(
        n_bits,
        OP_DEBUG_NAME + "QuantizedConv",
        int_input_names={"0"},
        constant_inputs={1: q_weights, 2: q_bias},
        strides=strides,
        pads=pads,
        kernel_shape=weights.shape[2:],
        dilations=dilations,
        group=group,
    )
    q_op.produces_graph_output = produces_output

    # Compute the result in floating point
    expected_result = q_op.calibrate(net_input)

    # For Conv1d, torch and ONNX both follow the same padding convention
    if is_conv1d:
        input_padded = torch.nn.functional.pad(torch.Tensor(net_input.copy()), pads)

    # For Conv2d, torch uses padding  (padding_left, padding_right, padding_top, padding_bottom)
    # While ONNX and Concrete ML use (padding_top, padding_left, padding_bottom, padding_right)
    else:
        input_padded = torch.nn.functional.pad(
            torch.Tensor(net_input.copy()), (pads[1], pads[3], pads[0], pads[2])
        )

    # Compute the reference result using the torch convolution operator
    torch_res = conv_torch_op(
        input=input_padded,
        weight=torch.Tensor(weights.copy()),
        bias=torch.Tensor(biases.squeeze().copy()) if biases is not None else None,
        stride=strides,
        groups=group,
    ).numpy()

    check_float_array_equal(torch_res, expected_result)

    # Compute the quantized result
    result = q_op(q_input).dequant()

    # The fp32 and quantized results should be very similar when quantization precision is high
    check_r2_score(result, expected_result)

    # Test the serialization of QuantizedConv
    check_serialization(
        q_op, QuantizedConv, equal_method=partial(quantized_op_results_are_equal, q_input=q_input)
    )


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
def test_quantized_avg_pool(params, n_bits, is_signed, check_r2_score, check_float_array_equal):
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
    # While ONNX and Concrete ML use (padding_top, padding_left, padding_bottom, padding_right)
    tx_pad = torch.nn.functional.pad(tinputs, (pads[1], pads[3], pads[0], pads[2]))

    # Compute the torch average pool
    bceil_mode = bool(ceil_mode)
    torch_res = torch.nn.functional.avg_pool2d(tx_pad, kernel_shape, strides, 0, bceil_mode).numpy()
    check_float_array_equal(torch_res, expected_result)

    # Compute the quantized result
    result = q_op(q_input).dequant()

    # The fp32 and quantized results should be very similar when quantization precision is high
    check_r2_score(expected_result, result)

    # Test the serialization of QuantizedAvgPool
    check_serialization(
        q_op,
        QuantizedAvgPool,
        equal_method=partial(quantized_op_results_are_equal, q_input=q_input),
    )


def test_quantized_avg_pool_args():
    """Check that unsupported parameters for AvgPool properly raise errors."""
    n_bits = 2

    with pytest.raises(AssertionError, match=r"Setting parameter 'kernel_shape' is required."):
        QuantizedAvgPool(
            n_bits,
            OP_DEBUG_NAME + "QuantizedAvgPool",
        )

    with pytest.raises(AssertionError, match=r"The 'auto_pad' parameter is not supported.*"):
        QuantizedAvgPool(
            n_bits,
            OP_DEBUG_NAME + "QuantizedAvgPool",
            kernel_shape=(1, 1),
            auto_pad="SAME_UPPER",
        )

    with pytest.raises(
        AssertionError, match=r"The Average Pool operator currently supports only 2d kernels."
    ):
        QuantizedAvgPool(
            n_bits,
            OP_DEBUG_NAME + "QuantizedAvgPool",
            kernel_shape=(1,),
        )

    with pytest.raises(
        AssertionError, match=r"Pad pixels must be included when calculating values on the edges.*"
    ):
        QuantizedAvgPool(
            n_bits,
            OP_DEBUG_NAME + "QuantizedAvgPool",
            kernel_shape=(1, 1),
            count_include_pad=0,
        )

    with pytest.raises(
        AssertionError,
        match=r"The Average Pool operator requires the number of strides to be the same.*",
    ):
        QuantizedAvgPool(
            n_bits,
            OP_DEBUG_NAME + "QuantizedAvgPool",
            kernel_shape=(1, 1),
            strides=(1,),
        )

    with pytest.raises(
        AssertionError, match=r"The Average Pool operator in Concrete ML requires padding.*"
    ):
        QuantizedAvgPool(
            n_bits,
            OP_DEBUG_NAME + "QuantizedAvgPool",
            kernel_shape=(1, 1),
            pads=(0, 0),
        )


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
def test_quantized_max_pool(params, n_bits, is_signed, check_r2_score, check_float_array_equal):
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
    # While ONNX and Concrete ML use (padding_top, padding_left, padding_bottom, padding_right)
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

    check_float_array_equal(torch_res, expected_result)

    # Compute the quantized result
    result = q_op(q_input).dequant()

    # The fp32 and quantized results should be very similar when quantization precision is high
    check_r2_score(expected_result, result)

    # Test the serialization of QuantizedMaxPool
    check_serialization(
        q_op,
        QuantizedMaxPool,
        equal_method=partial(quantized_op_results_are_equal, q_input=q_input),
    )


def test_quantized_conv_args():
    """Check that conv arguments are validated."""
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


@pytest.mark.parametrize(
    "pads", [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 1, 0], [0, 0, 2, 1, 0, 0, 5, 3]]
)
def test_quantized_pad(pads):
    """Test the padding operator on quantized values."""

    # First we test that the operator result is the same as the input
    # when the padding is 0
    # This is currently the only supported mode
    data = numpy.random.uniform(size=(1, 1, 4, 4))
    q_data = QuantizedArray(2, data)

    pads_torch = [pads[3], pads[7], pads[2], pads[6], pads[1], pads[5], pads[0], pads[4]]
    pad_torch = (
        torch.nn.functional.pad(
            torch.Tensor(q_data.qvalues), pads_torch, value=q_data.quantizer.zero_point
        )
        .int()
        .numpy()
    )

    pads = numpy.asarray(pads)
    pad_value = numpy.asarray(0)

    q_op = QuantizedPad(
        2,
        OP_DEBUG_NAME + "QuantizedPad",
        int_input_names={"0"},
        constant_inputs={1: pads, 2: pad_value},
        mode="constant",
        input_quant_opts=q_data.quantizer.quant_options,
    )

    q_op.calibrate(q_data.values)

    q_pad_output = q_op(q_data)
    assert numpy.array_equal(q_pad_output.qvalues, pad_torch)

    # Test the serialization of QuantizedPad
    check_serialization(
        q_op, QuantizedPad, equal_method=partial(quantized_op_results_are_equal, q_input=q_data)
    )


@pytest.mark.parametrize("mode", ["reflect", "wrap", "edge"])
def test_quantized_pad_mode_invalid(mode):
    """Check that invalid padding modes raise an assert."""
    # Now check that we assert when a different padding mode is given
    with pytest.raises(AssertionError):
        QuantizedPad(2, OP_DEBUG_NAME + "QuantizedPad", int_input_names={"0"}, mode=mode)


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

    # Test the serialization of QuantizedReshape
    check_serialization(
        reshape,
        QuantizedReshape,
        equal_method=partial(quantized_op_results_are_equal, q_input=q_arr0),
    )


@pytest.mark.parametrize(
    "original_shape, expand_shape",
    [
        ((1, 1, 1), (10, 5, 2)),  # Basic expansion in 3D
        ((10, 1, 2), (10, 5, 2)),  # Expansion with one dimension already matching
        ((1,), (3, 1)),  # Expansion from 1D to 2D
        ((1, 1), (10, 5)),  # Expansion in 2D
        ((3, 1, 1), (3, 5, 7)),  # 3D expansion with first dimension already matching
        ((1, 4), (3, 4)),  # 2D expansion where one dimension needs no expansion
        ((1, 1, 3), (2, 6, 3)),  # 3D, last dimension matches, others expand
        ((1,), (1, 1, 1, 1)),  # Expansion from 1D to 4D
        ((2, 2), (1, 2, 2)),  # 2D to 3D where original dimensions are in the middle
        ((1, 1, 1, 1), (5, 4, 3, 2)),  # 4D expansion from all ones
        ((1, 7, 1), (3, 7, 5)),  # 3D with middle dimension already matching
        ((2, 1), (2, 3)),  # 2D expansion where first dimension needs no expansion
        ((1, 2, 1, 4), (3, 2, 5, 4)),  # 4D, two dimensions match, others expand
    ],
)
def test_quantized_expand(original_shape, expand_shape):
    """Test quantized expand."""

    n_bits_expand = MAX_BITWIDTH_BACKWARD_COMPATIBLE

    num_values = numpy.prod(numpy.asarray(original_shape))
    data = numpy.arange(num_values).astype(numpy.float32)
    data = data.reshape(original_shape)

    q_arr0 = QuantizedArray(n_bits_expand, data)

    # Apply QuantizedExpand operation
    expand = QuantizedExpand(
        n_bits_expand,
        OP_DEBUG_NAME + "QuantizedExpand",
        constant_inputs={1: numpy.asarray(expand_shape)},
        input_quant_opts=q_arr0.quantizer.quant_options,
    )

    q_expanded = expand(q_arr0)

    # Assertions for expanded array
    assert q_expanded.quantizer.zero_point == q_arr0.quantizer.zero_point
    assert q_expanded.quantizer.scale == q_arr0.quantizer.scale
    assert q_expanded.qvalues.shape == expand_shape
    assert numpy.all(numpy.broadcast_to(q_arr0.qvalues, expand_shape) == q_expanded.qvalues)

    # Test the serialization of QuantizedExpand (if applicable)
    # Replace with appropriate serialization test if needed
    check_serialization(
        expand,
        QuantizedExpand,
        equal_method=partial(quantized_op_results_are_equal, q_input=q_arr0),
    )


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

    # De-quantized values must be close to original values
    dequant_values = q_output.dequant()

    # Check that all values are close
    check_r2_score(dequant_values, expected_output)

    # Test the serialization of QuantizedPRelu
    check_serialization(
        quantized_op,
        QuantizedPRelu,
        equal_method=partial(quantized_op_results_are_equal, q_input=q_inputs),
    )


@pytest.fixture(scope="module")
def random_test_data():
    """Generate data for comparisons operators."""
    return [
        (
            numpy.random.uniform(size=(1, 3, 32, 32)) * 4,
            numpy.random.uniform() * 0.7 + 2,
            numpy.random.uniform(),
            numpy.random.uniform(),
        ),
        (numpy.random.uniform(size=(1, 32)) * 100 - 50, numpy.random.uniform() * 50 - 25, 0, -1),
        (numpy.random.uniform(size=(1024,)), numpy.random.uniform(), -100, 100),
    ]


@pytest.mark.parametrize("n_bits", [16])
@pytest.mark.parametrize(
    "comparator",
    [
        QuantizedGreater,
        QuantizedGreaterOrEqual,
        QuantizedLess,
        QuantizedLessOrEqual,
        QuantizedEqual,
    ],
)
# pylint: disable-next=redefined-outer-name
def test_quantized_comparators_and_where(random_test_data, n_bits, comparator, check_r2_score):
    """Test a conditional pattern that is very common in quantization aware training."""
    for values, threshold, val_if_true, val_if_false in random_test_data:
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

        # Test the serialization of each Quantized operation
        for q_op in [q_op_comparator, q_cast, q_op_where]:
            check_serialization(
                q_op,
                type(q_op),
                equal_method=partial(quantized_op_results_are_equal, q_input=q_values),
            )


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

    # Test the serialization of QuantizedBatchNormalization
    check_serialization(
        q_op,
        QuantizedBatchNormalization,
        equal_method=partial(quantized_op_results_are_equal, q_input=q_x),
    )


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
# In Concrete ML, we consider that all inputs' first dimension should be a batch size
# even in single batch cases. This is why the following test parameters are considering axes that
# are sometimes equal to the input size's dimension, as the batch size is added within the
# test itself.
# Finally, the axis parameter should neither be None nor contain axis 0 as this dimension is used
# for batching the inference
@pytest.mark.parametrize(
    "size, axes, noop_with_empty_axes",
    [
        pytest.param(size, axes, noop, id=f"size-{size}-axes-{axes}-noop-{noop}")
        for (size, axes, noop) in [
            ((1,), (1,), 0),
            ((100, 1), (2,), 0),
            ((10, 100), (1,), 0),
            ((10, 10, 1000), (3,), 0),
            ((10, 10, 1000), (1, 3), 0),
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
    inputs = data_generator(size=(100,) + size)

    # Instantiate the operator
    quantized_reduce_sum = QuantizedReduceSum(
        n_bits,
        OP_DEBUG_NAME + "QuantizedReduceSum",
        constant_inputs={"axes": numpy.array(axes)},
        keepdims=keepdims,
        noop_with_empty_axes=noop_with_empty_axes,
    )

    # Calibrate the quantized op and retrieve the expected results
    expected_sum = quantized_reduce_sum.calibrate(inputs)

    # Retrieve the results computed by the quantized op applied on quantized inputs
    q_inputs = QuantizedArray(n_bits, inputs)
    q_computed_sum = quantized_reduce_sum(q_inputs)
    computed_sum = q_computed_sum.dequant()

    assert computed_sum.shape == expected_sum.shape, (
        f"Mismatch found in output shapes. Got {computed_sum.shape} but expected "
        f"{expected_sum.shape}."
    )

    check_r2_score(expected_sum, computed_sum)

    # Test the serialization of QuantizedReduceSum
    check_serialization(
        quantized_reduce_sum,
        QuantizedReduceSum,
        equal_method=partial(quantized_op_results_are_equal, q_input=q_inputs),
    )


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
        QuantizedSqueeze: test_quantized_squeeze,
        QuantizedExpand: test_quantized_expand,
        QuantizedEqual: test_quantized_comparators_and_where,
        QuantizedUnfold: test_quantized_unfold,
        ONNXSlice: test_quantized_slice,
        ONNXGather: test_quantized_gather,
        ONNXShape: test_quantized_shape,
        ONNXConstantOfShape: test_constant_of_shape,
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

    # Check that the data was not changed and is in the same order (i.e., no transposition)
    assert numpy.all(q_data.qvalues.ravel() == q_reshaped.qvalues.ravel())

    # Check that the output has the expected shape
    assert numpy.all(q_reshaped.qvalues.shape == expected_shape)

    # Test the serialization of QuantizedFlatten
    check_serialization(
        flatten_op,
        QuantizedFlatten,
        equal_method=partial(quantized_op_results_are_equal, q_input=q_data),
    )


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

    # Test the serialization of QuantizedBrevitasQuant
    check_serialization(
        quant,
        QuantizedBrevitasQuant,
        equal_method=partial(quantized_op_results_are_equal, q_input=q_data),
    )


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

    # Test the serialization of QuantizedTranspose
    check_serialization(
        transpose,
        QuantizedTranspose,
        equal_method=partial(quantized_op_results_are_equal, q_input=q_arr0),
    )


@pytest.mark.parametrize(
    "shape1, shape2, axis",
    [pytest.param((10, 5), (10, 5), 1), pytest.param((50, 40), (50, 40), 1)],
)
def test_quantized_concat(shape1, shape2, axis):
    """Test quantized concat."""

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

    # Test the serialization of QuantizedConcat
    check_serialization(
        q_concatenated,
        QuantizedConcat,
        equal_method=partial(quantized_op_results_are_equal, q_input=q_arr),
    )


@pytest.mark.parametrize(
    "shape, axis",
    [pytest.param((10, 5), (1,)), pytest.param((50, 40), (0,)), pytest.param((50, 40), (1, 2))],
)
def test_quantized_unsqueeze(shape, axis):
    """Test quantized unsqueeze."""

    def custom_numpy_unsqueeze(x, axes):
        for i, axis_ in enumerate(sorted(axes)):
            x = numpy.expand_dims(x, axis=axis_ + i)
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

    # Test the serialization of QuantizedUnsqueeze
    check_serialization(
        q_unsqueeze,
        QuantizedUnsqueeze,
        equal_method=partial(quantized_op_results_are_equal, q_input=q_arr),
    )


@pytest.mark.parametrize(
    "shape, axis",
    [
        pytest.param((10, 1, 5), (1,)),
        pytest.param((1, 50, 40), (0,)),
        pytest.param((1, 50, 1, 40), None),
    ],
)
def test_quantized_squeeze(shape, axis):
    """Test quantized squeeze."""

    n_bits_concat_output = MAX_BITWIDTH_BACKWARD_COMPATIBLE
    data = numpy.random.randn(*shape)

    q_arr = QuantizedArray(n_bits=n_bits_concat_output, values=data)

    q_squeeze = QuantizedSqueeze(
        n_bits_concat_output,
        OP_DEBUG_NAME + "QuantizedSqueeze",
        input_quant_opts=q_arr.quantizer.quant_options,
        constant_inputs={1: axis},
    )
    q_result = q_squeeze(q_arr)

    assert q_result.quantizer.zero_point == q_arr.quantizer.zero_point
    assert q_result.quantizer.scale == q_arr.quantizer.scale
    assert numpy.all(numpy.squeeze(q_arr.qvalues, axis=axis) == q_result.qvalues)

    # Test the serialization of QuantizedSqueeze
    check_serialization(
        q_squeeze,
        QuantizedSqueeze,
        equal_method=partial(quantized_op_results_are_equal, q_input=q_arr),
    )


def make_single_function_onnx_and_run(onnx_op, op_args_dict, op_attrs_dict, input_value, out_shape):
    """Make an ONNX model with a single operation of the specified type and run it."""

    def make_initializer(name, values):
        """Create a ONNX initializer node from a named ndarray.

        Args:
            name (str): name of the initializer
            values (numpy.ndarray): values that will be stored in ONNX

        Returns:
            result (NodeProto): the protobuf structure of an initializer node
        """

        return onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=[name],
            name=f"init_{name}",
            value=onnx.helper.make_tensor(
                f"cst_{name}",
                data_type=onnx.helper.np_dtype_to_tensor_dtype(values.dtype),
                dims=values.shape if isinstance(values, numpy.ndarray) else (len(values),),
                vals=values,
            ),
        )

    all_nodes = [make_initializer(k, v) for k, v in op_args_dict.items()]
    inputs = ["input"] + list(op_args_dict.keys())

    op_node = onnx.helper.make_node(
        onnx_op,
        inputs=inputs,
        name=onnx_op,
        outputs=["output"],
    )
    for k, v in op_attrs_dict.items():
        op_node.attribute.append(onnx.helper.make_attribute(k, v))

    all_nodes.append(op_node)

    dtype_onnx = onnx.helper.np_dtype_to_tensor_dtype(input_value.dtype)
    graph_proto = onnx.helper.make_graph(
        all_nodes,
        f"{onnx_op}_test",
        [onnx.helper.make_tensor_value_info("input", dtype_onnx, input_value.shape)],
        [onnx.helper.make_tensor_value_info("output", dtype_onnx, out_shape)],
    )

    op_proto = onnx.OperatorSetIdProto()
    op_proto.version = 14

    onnx_model = onnx.helper.make_model(graph_proto, producer_name="test", opset_imports=[op_proto])
    onnx.checker.check_model(onnx_model)

    buf = io.BytesIO()
    onnx.save_model(onnx_model, buf)
    buf.seek(0)

    sess = ort.InferenceSession(buf.read())
    onnx_output = sess.run(["output"], {onnx_model.graph.input[0].name: input_value})
    return onnx_output[0]


@pytest.mark.parametrize(
    "starts, ends, steps, axes",
    [
        pytest.param([0, 0, 0, 0], [-1, -1, -1, -1], [1, 1, 1, 1], [0, 1, 2, 3]),
        pytest.param([1], [5], [1], [0]),
        pytest.param([1, 1], [31, 31], [2, 2], [2, 3]),
        pytest.param([0, 0, 0, 0], [-1, -1, -1, -1], [1, 1, 1, 1], [0, -1, -2, -3]),
        pytest.param([0, 0, 0, 0], [-1, -1, -1, -1], None, None),
    ],
)
def test_quantized_slice(starts, ends, steps, axes):
    """Check that the Concrete ML Slice operator is equivalent to the ONNX slice operator."""

    # Cast all inputs to numpy arrays
    starts = numpy.asarray(starts)
    ends = numpy.asarray(ends)
    steps = numpy.asarray(steps) if steps is not None else None
    axes = numpy.asarray(axes) if axes is not None else None

    # Initialize data and the op
    data = numpy.random.randn(32, 16, 32, 32)
    q_arr = QuantizedArray(n_bits=8, values=data)

    q_op = ONNXSlice(
        8,
        "slice_op",
        constant_inputs={1: starts, 2: ends, 3: axes, 4: steps},
        input_quant_opts=q_arr.quantizer.quant_options,
    )

    result = q_op(q_arr)

    list_args = [("starts", starts), ("ends", ends)]
    if axes is not None:
        list_args.append(("axes", axes))
        if steps is not None:
            list_args.append(("steps", steps))

    onnx_output = make_single_function_onnx_and_run(
        "Slice",
        OrderedDict(list_args),
        {},
        q_arr.qvalues,
        result.qvalues.shape,
    )
    assert numpy.array_equal(onnx_output, result.qvalues)

    # Test the serialization of ONNXSlice
    check_serialization(
        q_op, ONNXSlice, equal_method=partial(quantized_op_results_are_equal, q_input=q_arr)
    )


@pytest.mark.parametrize(
    "indices, axis",
    [
        pytest.param([1], 0),
        pytest.param([1, 5, 7], 0),
        pytest.param([1, 15], 3),
        pytest.param([0, -1, -3, -16], -2),
    ],
)
def test_quantized_gather(indices, axis):
    """Test the Gather operator."""

    # Cast all inputs to numpy arrays
    indices = numpy.asarray(indices)

    # Initialize data and the op
    data = numpy.random.randn(32, 16, 32, 32)
    q_arr = QuantizedArray(n_bits=8, values=data)

    q_op = ONNXGather(
        8,
        "gather_op",
        constant_inputs={1: indices},
        input_quant_opts=q_arr.quantizer.quant_options,
        axis=numpy.asarray(axis),
    )

    result = q_op(q_arr)

    onnx_output = make_single_function_onnx_and_run(
        "Gather",
        OrderedDict([("indices", indices)]),
        {"axis": axis},
        q_arr.qvalues,
        result.qvalues.shape,
    )
    assert numpy.array_equal(onnx_output, result.qvalues)

    # Test the serialization of ONNXGather
    check_serialization(
        q_op, ONNXGather, equal_method=partial(quantized_op_results_are_equal, q_input=q_arr)
    )


@pytest.mark.parametrize("shape", [[1], [10, 10], [1, 50, 20, 20]])
@pytest.mark.parametrize("value", [[0], [-1], [1]])
def test_constant_of_shape(shape, value):
    """Test the ConstantOfShape operator."""

    # Create the op and apply it
    q_op = ONNXConstantOfShape(8, "cst_of_shape_op", constant_inputs={0: shape}, value=value)

    result = q_op()

    # Check that the created tensor contains the correct values
    assert numpy.array_equal(numpy.ones(tuple(shape)) * value, result)

    # Test the serialization of ONNXConstantOfShape
    check_serialization(q_op, ONNXConstantOfShape, equal_method=quantized_op_results_are_equal)


@pytest.mark.parametrize("shape", [(1,), (10, 10), (1, 50, 20, 20)])
def test_quantized_shape(shape):
    """Test the shape operator."""

    # Create an input
    np_input = numpy.zeros(tuple(shape))
    q_input = QuantizedArray(8, np_input)

    # Create the op and apply it
    q_op = ONNXShape(8, "shape_op")
    result = q_op(q_input)

    # Check that the Concrete ML op returns the shape
    assert np_input.shape == tuple(result)

    # Test the serialization of ONNXShape
    check_serialization(
        q_op, ONNXShape, equal_method=partial(quantized_op_results_are_equal, q_input=q_input)
    )


@pytest.mark.parametrize("n_bits", [16])
@pytest.mark.parametrize(
    "params",
    [
        (
            numpy.random.uniform(low=-2.0, high=2.0, size=(1, 1, 32, 32)),
            (3, 3),
            (2, 2),
            (0, 0, 0, 0),
        ),
        (
            numpy.random.uniform(low=-1.2, high=0.2, size=(10, 1, 16, 16)),
            (2, 2),
            (1, 1),
            (0, 0, 0, 0),
        ),
        (
            numpy.random.uniform(low=-2.0, high=2.0, size=(2, 32, 4, 4)),
            (2, 2),
            (1, 1),
            (0, 0, 0, 0),
        ),
        (
            numpy.random.uniform(low=-2.0, high=2.0, size=(2, 32, 4, 4)),
            (2, 4),
            (1, 1),
            (1, 2, 1, 2),
        ),
        (
            numpy.random.uniform(low=-2.0, high=2.0, size=(2, 32, 4, 4)),
            (2, 4),
            (1, 1),
            (0, 2, 0, 2),
        ),
        (
            numpy.random.uniform(low=-2.0, high=2.0, size=(2, 32, 5, 5)),
            (3, 3),
            (1, 1),
            (1, 1, 1, 1),
        ),
        (
            numpy.random.uniform(low=-2.0, high=2.0, size=(2, 1, 7, 5)),
            (5, 1),
            (1, 1),
            (1, 2, 0, 4),
        ),
        (
            numpy.random.uniform(low=-2.0, high=2.0, size=(1, 1, 16, 16)),
            (2, 2),
            (4, 4),
            (1, 2, 0, 4),
        ),
    ],
)
@pytest.mark.parametrize("is_signed", [True, False])
def test_quantized_unfold(params, n_bits, is_signed, check_r2_score, check_float_array_equal):
    """Test the quantized average pool operator."""

    # Retrieve arguments
    net_input, kernel_shape, strides, pads = params

    # Create quantized data
    q_input = QuantizedArray(n_bits, net_input, is_signed=is_signed)

    q_op = QuantizedUnfold(
        n_bits,
        OP_DEBUG_NAME + "QuantizedUnfold",
        strides=strides,
        pads=pads,
        kernel_shape=kernel_shape,
        # ceil_mode=ceil_mode,
        input_quant_opts=q_input.quantizer.quant_options,
    )

    # Compute the result in floating point
    expected_result = q_op.calibrate(net_input)

    # Pad the input if needed
    tinputs = torch.Tensor(net_input.copy())

    # Torch uses padding  (padding_left,padding_right, padding_top,padding_bottom)
    # While ONNX and Concrete ML use (padding_top, padding_left, padding_bottom, padding_right)
    tx_pad = torch.nn.functional.pad(tinputs, (pads[1], pads[3], pads[0], pads[2]))

    # Compute the torch unfold
    torch_res = torch.nn.functional.unfold(tx_pad, kernel_shape, 1, 0, strides).numpy()

    check_float_array_equal(torch_res, expected_result)

    # Compute the quantized result
    result = q_op(q_input).dequant()

    # The fp32 and quantized results should be very similar when quantization precision is high
    check_r2_score(expected_result, result)

    # Test the serialization of QuantizedUnfold
    check_serialization(
        q_op,
        QuantizedUnfold,
        equal_method=partial(quantized_op_results_are_equal, q_input=q_input),
    )
