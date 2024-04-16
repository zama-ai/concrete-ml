"""Tests for the torch to numpy module."""

from functools import partial

import numpy
import pytest
import torch
from torch import nn

from concrete.ml.pytest.torch_models import (
    CNN,
    FC,
    CNNGrouped,
    CNNInvalid,
    CNNMaxPool,
    NetWithConcatUnsqueeze,
    NetWithLoops,
    QATTestModule,
)
from concrete.ml.torch import NumpyModule


@pytest.mark.parametrize(
    "model_class, input_shape",
    [
        pytest.param(FC, (100, 32 * 32 * 3)),
        pytest.param(partial(CNN, input_output=3), (5, 3, 32, 32)),
        pytest.param(partial(CNNGrouped, input_output=6, groups=3), (5, 6, 7, 7)),
        pytest.param(partial(CNNMaxPool, input_output=3), (5, 3, 32, 32)),
        pytest.param(
            partial(NetWithLoops, input_output=32 * 32 * 3, n_fc_layers=4), (100, 32 * 32 * 3)
        ),
        pytest.param(
            partial(NetWithConcatUnsqueeze, input_output=32 * 32 * 3, n_fc_layers=4),
            (100, 32 * 32 * 3),
        ),
        pytest.param(QATTestModule, (5, 3, 6, 6)),
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
        pytest.param(nn.LogSigmoid, id="LogSigmoid"),
        pytest.param(nn.GELU, id="GELU"),
        # Some issues are still encountered with some activations
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/335
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
def test_torch_to_numpy(model_class, input_shape, activation_function, check_r2_score):
    """Test the different model architecture from torch numpy."""

    # Define the torch model
    torch_fc_model = model_class(activation_function=activation_function)

    # Since we have networks with Batch Normalization, we need to manually set them to evaluation
    # mode. ONNX export does the same, so to ensure torch results are the same as the results
    # when running the ONNX graph through NumpyMode we need to call .eval()
    # Calling .eval() fixes the mean/var of the batchnormalization layer and stops is being updated
    # during .forward
    torch_fc_model.eval()

    # Create random input
    torch_input_1 = torch.randn(input_shape)

    # Predict with torch model
    torch_predictions = torch_fc_model(torch_input_1).detach().numpy()

    # Create corresponding numpy model
    numpy_fc_model = NumpyModule(torch_fc_model, torch_input_1)

    # Torch input to numpy
    numpy_input_1 = torch_input_1.detach().numpy()

    # Predict with numpy model
    numpy_predictions = numpy_fc_model(numpy_input_1)

    # Test: the output of the numpy model is the same as the torch model.
    assert numpy_predictions.shape == torch_predictions.shape

    # Test: prediction from the numpy model are the same as the torch model.
    check_r2_score(torch_predictions, numpy_predictions)

    # Test: dynamics between layers is working (quantized input and activations)
    torch_input_2 = torch.randn(input_shape)

    # Make sure both inputs are different
    assert (torch_input_1 != torch_input_2).any()

    # Predict with torch
    torch_predictions = torch_fc_model(torch_input_2).detach().numpy()

    # Torch input to numpy
    numpy_input_2 = torch_input_2.detach().numpy()

    # Numpy predictions using the previous model
    numpy_predictions = numpy_fc_model(numpy_input_2)
    check_r2_score(torch_predictions, numpy_predictions)


def test_raises():
    """Function to test incompatible layers."""

    torch_incompatible_model = CNNInvalid(nn.ReLU, False)

    error_msg_pattern = (
        "The following ONNX operators are required to convert the torch model to numpy but are"
        " not currently implemented: ReduceMean*"
    )

    with pytest.raises(
        Exception,
        match=error_msg_pattern,
    ):
        dummy_input = torch.randn(1, 3, 32, 32)
        mod = NumpyModule(torch_incompatible_model, dummy_input)
        mod(dummy_input.numpy())


class AddTest(nn.Module):
    """Simple torch module to test ONNX 'Add' operator."""

    def __init__(self) -> None:
        super().__init__()
        self.bias = 1.0

    def forward(self, x):
        """Forward pass."""
        return x + self.bias


class MatmulTest(nn.Module):
    """Simple torch module to test ONNX 'Matmul' operator."""

    def __init__(self) -> None:
        super().__init__()
        self.matmul = nn.Linear(3, 10, bias=False)

    def forward(self, x):
        """Forward pass."""
        return self.matmul(x)


class ReluTest(nn.Module):
    """Simple torch module to test ONNX 'Relu' operator."""

    # Store a ReLU op to avoid pylint warnings
    def __init__(self) -> None:
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        """Forward pass."""
        return self.relu(x)


@pytest.mark.parametrize(
    "model_class,input_shape",
    [
        pytest.param(AddTest, (10, 10)),
        pytest.param(MatmulTest, (10, 3)),
        pytest.param(ReluTest, (10, 10)),
    ],
)
def test_torch_to_numpy_onnx_ops(model_class, input_shape, check_r2_score):
    """Test the different model architecture from torch numpy."""

    # Define the torch model
    torch_fc_model = model_class()
    # Create random input
    torch_input_1 = torch.randn(input_shape)
    # Predict with torch model
    torch_predictions = torch_fc_model(torch_input_1).detach().numpy()
    # Create corresponding numpy model
    numpy_fc_model = NumpyModule(torch_fc_model, torch_input_1)
    # Torch input to numpy
    numpy_input_1 = torch_input_1.detach().numpy()
    # Predict with numpy model
    numpy_predictions = numpy_fc_model(numpy_input_1)

    # Test: the output of the numpy model is the same as the torch model.
    assert numpy_predictions.shape == torch_predictions.shape
    # Test: prediction from the numpy model are the same as the torch model.
    check_r2_score(torch_predictions, numpy_predictions)

    # Test: dynamics between layers is working (quantized input and activations)
    torch_input_2 = torch.randn(input_shape)
    # Make sure both inputs are different
    assert (torch_input_1 != torch_input_2).any()
    # Predict with torch
    torch_predictions = torch_fc_model(torch_input_2).detach().numpy()
    # Torch input to numpy
    numpy_input_2 = torch_input_2.detach().numpy()
    # Numpy predictions using the previous model
    numpy_predictions = numpy_fc_model(numpy_input_2)
    check_r2_score(torch_predictions, numpy_predictions)


@pytest.mark.parametrize(
    "incompatible_model",
    [pytest.param("STRING")],
)
def test_raises_incompatible_model_type(incompatible_model):
    """Test an incompatible model type."""
    with pytest.raises(
        ValueError,
        match=("model must be a torch.nn.Module or an onnx.ModelProto, got str"),
    ):
        _ = NumpyModule(incompatible_model, torch.from_numpy(numpy.array([1, 2, 3])))
