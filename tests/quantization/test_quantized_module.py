"""Tests for the quantized module."""
import re
from functools import partial

import numpy
import pytest
import torch
from torch import nn

from concrete.ml.pytest.torch_models import CNN, FC, CNNMaxPool
from concrete.ml.quantization import PostTrainingAffineQuantization
from concrete.ml.torch import NumpyModule

N_BITS_LIST = [
    20,
    16,
    8,
    {"model_inputs": 8, "op_weights": 8, "op_inputs": 8, "model_outputs": 8},
    {"model_inputs": 12, "op_weights": 15, "op_inputs": 15, "model_outputs": 16},
]


@pytest.mark.parametrize("n_bits", [pytest.param(n_bits) for n_bits in N_BITS_LIST])
@pytest.mark.parametrize(
    "model, input_shape",
    [
        pytest.param(FC, (100, 32 * 32 * 3)),
        pytest.param(partial(CNN, input_output=3), (100, 3, 32, 32)),
        pytest.param(partial(CNNMaxPool, input_output=3), (100, 3, 32, 32)),
    ],
)
@pytest.mark.parametrize(
    "activation_function",
    [
        pytest.param(nn.Sigmoid, id="Sigmoid"),
        pytest.param(nn.ReLU, id="ReLU"),
        pytest.param(nn.ReLU6, id="ReLU6"),
        pytest.param(nn.Tanh, id="Tanh"),
        pytest.param(nn.ELU, id="ELU"),
        pytest.param(nn.Hardsigmoid, id="Hardsigmoid"),
        pytest.param(nn.Hardtanh, id="Hardtanh"),
        pytest.param(nn.LeakyReLU, id="LeakyReLU"),
        pytest.param(nn.SELU, id="SELU"),
        pytest.param(nn.CELU, id="CELU"),
        pytest.param(nn.Softplus, id="Softplus"),
        pytest.param(nn.PReLU, id="PReLU"),
        pytest.param(nn.Hardswish, id="Hardswish"),
        pytest.param(nn.SiLU, id="SiLU"),
        pytest.param(nn.Mish, id="Mish"),
        pytest.param(nn.Tanhshrink, id="Tanhshrink"),
        pytest.param(partial(nn.Threshold, threshold=0, value=0), id="Threshold"),
        pytest.param(nn.Softshrink, id="Softshrink"),
        pytest.param(nn.Hardshrink, id="Hardshrink"),
        pytest.param(nn.Softsign, id="Softsign"),
        pytest.param(nn.GELU, id="GELU"),
        # Works but accuracy issues sometimes in compilation
        pytest.param(nn.LogSigmoid, id="LogSigmoid"),
        # Works within the conversion but will not compile
        # FIXME, #335: still some issues with these activations
        #
        # Other problems, certainly related to tests:
        # Required positional arguments: 'embed_dim' and 'num_heads' and fails with a partial
        # pytest.param(nn.MultiheadAttention, id="MultiheadAttention"),
        # Activation with a RandomUniformLike
        # pytest.param(nn.RReLU, id="RReLU"),
        # Halving dimension must be even, but dimension 3 is size 3
        # pytest.param(nn.GLU, id="GLU"),
    ],
)
def test_quantized_linear(model, input_shape, n_bits, activation_function, check_r2_score):
    """Test the quantized module with a post-training static quantization.

    With n_bits>>0 we expect the results of the quantized module
    to be the same as the standard module.
    """
    # Define the torch model
    torch_fc_model = model(activation_function=activation_function)
    torch_fc_model.eval()

    # Create random input
    numpy_input = numpy.random.uniform(size=input_shape)

    torch_input = torch.from_numpy(numpy_input).float()
    # Create corresponding numpy model
    numpy_fc_model = NumpyModule(torch_fc_model, torch_input)

    torch_prediction = torch_fc_model(torch_input).detach().numpy()

    # Predict with real model
    numpy_prediction = numpy_fc_model(numpy_input)

    # Check that numpy execution is accurate w.r.t torch execution
    check_r2_score(torch_prediction, numpy_prediction)

    # Quantize with post-training static method
    post_training_quant = PostTrainingAffineQuantization(n_bits, numpy_fc_model)
    quantized_model = post_training_quant.quantize_module(numpy_input)

    # Quantize input
    qinput = quantized_model.quantize_input(numpy_input)
    prediction = quantized_model(qinput)
    dequant_manually_prediction = quantized_model.dequantize_output(prediction)

    # Forward and Dequantize to get back to real values
    dequant_prediction = quantized_model.forward_and_dequant(qinput)

    # Both dequant prediction should be equal
    assert numpy.array_equal(dequant_manually_prediction, dequant_prediction)

    # Check that the actual prediction are close to the expected predictions
    check_r2_score(numpy_prediction, dequant_prediction)


@pytest.mark.parametrize(
    "model, input_shape",
    [
        pytest.param(FC, (100, 32 * 32 * 3)),
    ],
)
@pytest.mark.parametrize(
    "dtype, err_msg",
    [
        pytest.param(
            numpy.float32,
            re.escape(
                "Inputs: #0 (float32) are not integer types. "
                "Make sure you quantize your input before calling forward."
            ),
        ),
    ],
)
def test_raises_on_float_inputs(model, input_shape, dtype, err_msg):
    """Function to test incompatible inputs."""

    # Define the torch model
    torch_fc_model = model(nn.ReLU)
    # Create random input
    numpy_input = numpy.random.uniform(size=input_shape).astype(dtype)
    # Create corresponding numpy model
    numpy_fc_model = NumpyModule(torch_fc_model, torch.from_numpy(numpy_input).float())
    # Quantize with post-training static method
    post_training_quant = PostTrainingAffineQuantization(8, numpy_fc_model)
    quantized_model = post_training_quant.quantize_module(numpy_input)

    with pytest.raises(
        ValueError,
        match=err_msg,
    ):
        quantized_model(numpy_input)


@pytest.mark.parametrize("n_bits", [2, 7])
@pytest.mark.parametrize(
    "model, input_shape",
    [
        pytest.param(FC, (100, 32 * 32 * 3)),
        pytest.param(partial(CNN, input_output=3), (100, 3, 32, 32)),
    ],
)
@pytest.mark.parametrize(
    "activation_function",
    [
        pytest.param(nn.ReLU, id="ReLU"),
    ],
)
def test_intermediary_values(n_bits, model, input_shape, activation_function):
    """Check that all torch linear/conv layers return debug values in the QuantizedModule."""

    torch_fc_model = model(activation_function=activation_function)
    torch_fc_model.eval()

    # Create random input
    numpy_input = numpy.random.uniform(size=input_shape)

    torch_input = torch.from_numpy(numpy_input).float()
    # Create corresponding numpy model
    numpy_fc_model = NumpyModule(torch_fc_model, torch_input)

    # Quantize with post-training static method
    post_training_quant = PostTrainingAffineQuantization(n_bits, numpy_fc_model)
    quantized_model = post_training_quant.quantize_module(numpy_input)

    # Quantize input
    qinput = quantized_model.quantize_input(numpy_input)
    _, debug_values = quantized_model.forward(qinput, debug=True)

    # Count the number of Gemm/Conv layers in the CML debug values
    num_gemm_conv = 0
    for layer_name in debug_values:
        if "Gemm" not in layer_name and "Conv" not in layer_name:
            continue

        num_gemm_conv += 1

    # Count the number of Gemm/Conv layers in the pytorch model
    num_torch_gemm_conv = 0
    for layer in torch_fc_model.modules():
        if not isinstance(layer, (nn.Conv2d, nn.Linear)):
            continue
        num_torch_gemm_conv += 1

    # Make sure we have debug output for all conv/gemm layers in CML
    assert num_gemm_conv == num_torch_gemm_conv
