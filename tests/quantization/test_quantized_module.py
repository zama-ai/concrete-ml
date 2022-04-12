"""Tests for the quantized module."""
import re

import numpy
import pytest
import torch
from torch import nn

from concrete.ml.quantization import PostTrainingAffineQuantization
from concrete.ml.torch import NumpyModule


class CNN(nn.Module):
    """Torch CNN model for the tests."""

    def __init__(self, activation_function):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.act_f = activation_function()

    def forward(self, x):
        """Forward pass."""
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.act_f(self.fc1(x))
        x = self.act_f(self.fc2(x))
        x = self.fc3(x)
        return x


class FC(nn.Module):
    """Torch model for the tests"""

    def __init__(self, activation_function):
        super().__init__()
        self.fc1 = nn.Linear(in_features=32 * 32 * 3, out_features=128)
        self.act_1 = activation_function()
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.act_2 = activation_function()
        self.fc3 = nn.Linear(in_features=64, out_features=64)
        self.act_3 = activation_function()
        self.fc4 = nn.Linear(in_features=64, out_features=64)
        self.act_4 = activation_function()
        self.fc5 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        """Forward pass."""
        out = self.fc1(x)
        out = self.act_1(out)
        out = self.fc2(out)
        out = self.act_2(out)
        out = self.fc3(out)
        out = self.act_3(out)
        out = self.fc4(out)
        out = self.act_4(out)
        out = self.fc5(out)

        return out


N_BITS_LIST = [20, 16, 8, {"inputs": 8, "weights": 8}]


@pytest.mark.parametrize("n_bits", [pytest.param(n_bits) for n_bits in N_BITS_LIST])
@pytest.mark.parametrize(
    "model, input_shape",
    [
        pytest.param(FC, (100, 32 * 32 * 3)),
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
        pytest.param(nn.LogSigmoid, id="LogSigmoid"),
        pytest.param(nn.SELU, id="SELU"),
        pytest.param(nn.CELU, id="CELU"),
        pytest.param(nn.Softplus, id="Softplus"),
        pytest.param(nn.PReLU, id="PReLU"),
        pytest.param(nn.Hardswish, id="Hardswish"),
        # Are currently not supported for various reasons:
        #
        # No quantized operators
        # pytest.param(nn.GELU, id="GELU"), # No quantized Div
        # pytest.param(nn.SiLU, id="SiLU"), # No quantized Mul
        # pytest.param(nn.Mish, id="Mish"), # No quantized Mul
        # pytest.param(nn.Softsign, id="Softsign"), # No quantized Div
        # pytest.param(nn.Tanhshrink, id="Tanhshrink"), # No quantized Sub
        #
        # Missing operators in ONNX
        # pytest.param(nn.Hardshrink, id="Hardshrink"), # Missing Cast, Or, Where in ONNX
        # pytest.param(nn.RReLU, id="RReLU"), # Missing RandomUniformLike in ONNX
        # pytest.param(nn.Softshrink, id="Softshrink"), # Missing Where in ONNX
        #
        # Other issues
        # pytest.param(nn.MultiheadAttention, id="MultiheadAttention"),
        #       TypeError: __init__() missing 2 required positional arguments: 'embed_dim' and
        #       'num_heads'
        # pytest.param(nn.Threshold, id="Threshold"),
        #       TypeError: __init__() missing 2 required positional arguments: 'threshold' and
        #       'value'
        # pytest.param(nn.GLU, id="GLU"),
        #       RuntimeError: mat1 and mat2 shapes cannot be multiplied (100x64 and 128x64)
    ],
)
@pytest.mark.parametrize("is_signed", [pytest.param(True), pytest.param(False)])
def test_quantized_linear(
    model, input_shape, n_bits, activation_function, is_signed, check_r2_score
):
    """Test the quantized module with a post-training static quantization.

    With n_bits>>0 we expect the results of the quantized module
    to be the same as the standard module.
    """
    # Define the torch model
    torch_fc_model = model(activation_function)
    # Create random input
    numpy_input = numpy.random.uniform(size=input_shape)
    # Create corresponding numpy model
    numpy_fc_model = NumpyModule(torch_fc_model, torch.from_numpy(numpy_input).float())
    # Predict with real model
    numpy_prediction = numpy_fc_model(numpy_input)
    # Quantize with post-training static method
    post_training_quant = PostTrainingAffineQuantization(
        n_bits, numpy_fc_model, is_signed=is_signed
    )
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
