"""Tests for the torch to numpy module."""
import numpy
import pytest
from torch import nn

from concrete.ml.torch.compile import compile_torch_model

# INPUT_OUTPUT_FEATURE is the number of input and output of each of the network layers.
# (as well as the input of the network itself)
INPUT_OUTPUT_FEATURE = [1, 2]


class FC(nn.Module):
    """Torch model for the tests"""

    def __init__(self, input_output, activation_function):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_output, out_features=input_output)
        self.act_f = activation_function()
        self.fc2 = nn.Linear(in_features=input_output, out_features=input_output)

    def forward(self, x):
        """Forward pass."""
        out = self.fc1(x)
        out = self.act_f(out)
        out = self.fc2(out)

        return out


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
    "model",
    [pytest.param(FC)],
)
@pytest.mark.parametrize(
    "input_output_feature",
    [pytest.param(input_output_feature) for input_output_feature in INPUT_OUTPUT_FEATURE],
)
def test_compile_torch(
    input_output_feature,
    model,
    activation_function,
    seed_torch,
    default_compilation_configuration,
    check_is_good_execution,
):
    """Test the different model architecture from torch numpy."""

    # Seed torch
    seed_torch()

    n_bits = 2

    # Define an input shape (n_examples, n_features)
    n_examples = 50

    # Define the torch model
    torch_fc_model = model(input_output_feature, activation_function)
    # Create random input
    inputset = numpy.random.uniform(-100, 100, size=(n_examples, input_output_feature))

    # Compile
    quantized_numpy_module = compile_torch_model(
        torch_fc_model,
        inputset,
        default_compilation_configuration,
        n_bits=n_bits,
    )

    # Create test data from the same distribution and quantize using
    # learned quantization parameters during compilation
    x_test = numpy.random.uniform(-100, 100, size=(10, input_output_feature))
    qtest = quantized_numpy_module.quantize_input(x_test)

    # Compare predictions between FHE and QuantizedModule
    for x_q in qtest:
        x_q = numpy.expand_dims(x_q, 0)
        check_is_good_execution(
            fhe_circuit=quantized_numpy_module.forward_fhe,
            function=quantized_numpy_module.forward,
            args=[x_q.astype(numpy.uint8)],
            check_function=numpy.array_equal,
            verbose=False,
        )
