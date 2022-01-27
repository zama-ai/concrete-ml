"""Tests for the torch to numpy module."""
import numpy
import pytest
import torch
from torch import nn

from concrete.ml.torch import NumpyModule
from concrete.ml.torch.numpy_module import NewNumpyModule


class CNN(nn.Module):
    """Torch CNN model for the tests."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """Forward pass."""
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
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


@pytest.mark.parametrize(
    "model, input_shape",
    [
        pytest.param(FC, (100, 32 * 32 * 3)),
    ],
)
@pytest.mark.parametrize(
    "activation_function",
    [
        pytest.param(nn.Sigmoid, id="sigmoid"),
        pytest.param(nn.ReLU, id="relu"),
        pytest.param(nn.ReLU6, id="relu6"),
        pytest.param(nn.Tanh, id="tanh"),
    ],
)
@pytest.mark.parametrize(
    "numpy_module_type",
    [
        NumpyModule,
        NewNumpyModule,
    ],
)
def test_torch_to_numpy(model, input_shape, activation_function, numpy_module_type, seed_torch):
    """Test the different model architecture from torch numpy."""

    # Seed torch
    seed_torch()
    # Define the torch model
    torch_fc_model = model(activation_function)
    # Create random input
    torch_input_1 = torch.randn(input_shape)
    # Predict with torch model
    torch_predictions = torch_fc_model(torch_input_1).detach().numpy()
    # Create corresponding numpy model
    numpy_fc_model = (
        numpy_module_type(torch_fc_model)
        if numpy_module_type == NumpyModule
        else numpy_module_type(torch_fc_model, torch_input_1)
    )
    # Torch input to numpy
    numpy_input_1 = torch_input_1.detach().numpy()
    # Predict with numpy model
    numpy_predictions = numpy_fc_model(numpy_input_1)

    # Test: the output of the numpy model is the same as the torch model.
    assert numpy_predictions.shape == torch_predictions.shape
    # Test: prediction from the numpy model are the same as the torh model.
    assert numpy.isclose(torch_predictions, numpy_predictions, rtol=10 - 3).all()

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
    assert numpy.isclose(torch_predictions, numpy_predictions, rtol=10 - 3).all()


@pytest.mark.parametrize(
    "model",
    [pytest.param(CNN)],
)
@pytest.mark.parametrize(
    "numpy_module_type,err_msg",
    [
        pytest.param(
            NewNumpyModule,
            "The following ONNX operators are required to convert the torch model to numpy but are"
            " not currently implemented: AveragePool, Conv, Flatten, Pad\\..*",
        ),
        pytest.param(
            NumpyModule,
            "The following module is currently not implemented: Conv2d\\. "
            "Please stick to the available torch modules: "
            f"{', '.join(sorted(module.__name__ for module in NumpyModule.IMPLEMENTED_MODULES))}.",
        ),
    ],
)
def test_raises(model, numpy_module_type, err_msg, seed_torch):
    """Function to test incompatible layers."""

    seed_torch()
    torch_incompatible_model = model()

    with pytest.raises(
        ValueError,
        match=err_msg,
    ):
        if numpy_module_type == NumpyModule:
            NumpyModule(torch_incompatible_model)
        else:
            # NCHW
            dummy_input = torch.randn(1, 3, 32, 32)
            NewNumpyModule(torch_incompatible_model, dummy_input)


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
    "model,input_shape",
    [
        pytest.param(AddTest, (10, 10)),
        pytest.param(MatmulTest, (10, 3)),
        pytest.param(ReluTest, (10, 10)),
    ],
)
def test_torch_to_numpy_onnx_ops(model, input_shape, seed_torch):
    """Test the different model architecture from torch numpy."""

    # Seed torch
    seed_torch()
    # Define the torch model
    torch_fc_model = model()
    # Create random input
    torch_input_1 = torch.randn(input_shape)
    # Predict with torch model
    torch_predictions = torch_fc_model(torch_input_1).detach().numpy()
    # Create corresponding numpy model
    numpy_fc_model = NewNumpyModule(torch_fc_model, torch_input_1)
    # Torch input to numpy
    numpy_input_1 = torch_input_1.detach().numpy()
    # Predict with numpy model
    numpy_predictions = numpy_fc_model(numpy_input_1)

    # Test: the output of the numpy model is the same as the torch model.
    assert numpy_predictions.shape == torch_predictions.shape
    # Test: prediction from the numpy model are the same as the torh model.
    assert numpy.isclose(torch_predictions, numpy_predictions, rtol=10 - 3).all()

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
    assert numpy.isclose(torch_predictions, numpy_predictions, rtol=10 - 3).all()
