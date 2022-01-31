"""Test Neural Networks compilations"""
import numpy
import pytest
import torch
from torch import nn

from concrete.ml.quantization import QuantizedArray
from concrete.ml.quantization.post_training import PostTrainingAffineQuantization
from concrete.ml.torch.numpy_module import NumpyModule

# INPUT_OUTPUT_FEATURE is the number of input and output of each of the network layers.
# (as well as the input of the network itself)
# Currently, with 7 bits maximum, we can use 15 weights max in the theoretical case.
INPUT_OUTPUT_FEATURE = [1, 2, 3]


class FC(nn.Module):
    """Torch model for the tests"""

    def __init__(self, input_output, act):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_output, out_features=input_output)
        self.act = act()
        self.fc2 = nn.Linear(in_features=input_output, out_features=input_output)

    def forward(self, x):
        """Forward pass."""
        out = self.fc1(x)
        out = self.act(out)
        out = self.fc2(out)

        return out


@pytest.mark.parametrize(
    "model",
    [pytest.param(FC)],
)
@pytest.mark.parametrize(
    "input_output_feature",
    [pytest.param(input_output_feature) for input_output_feature in INPUT_OUTPUT_FEATURE],
)
@pytest.mark.parametrize("activation", [nn.ReLU6, nn.Sigmoid])
def test_quantized_module_compilation(
    input_output_feature,
    model,
    activation,
    seed_torch,
    default_compilation_configuration,
    check_is_good_execution,
):
    """Test a neural network compilation for FHE inference."""
    # Seed torch
    seed_torch()

    n_bits = 2

    # Define an input shape (n_examples, n_features)
    input_shape = (50, input_output_feature)

    # Build a random Quantized Fully Connected Neural Network

    # Define the torch model
    torch_fc_model = model(input_output_feature, activation)
    # Create random input
    numpy_input = numpy.random.uniform(-100, 100, size=input_shape)

    # Create corresponding numpy model
    numpy_fc_model = NumpyModule(torch_fc_model, torch.from_numpy(numpy_input).float())
    # Quantize with post-training static method
    post_training_quant = PostTrainingAffineQuantization(n_bits, numpy_fc_model)
    quantized_model = post_training_quant.quantize_module(numpy_input)
    # Quantize input
    q_input = QuantizedArray(n_bits, numpy_input)

    # Compile
    quantized_model.compile(q_input, default_compilation_configuration)

    for x_q in q_input.qvalues:
        x_q = numpy.expand_dims(x_q, 0)
        check_is_good_execution(
            fhe_circuit=quantized_model.forward_fhe,
            function=quantized_model.forward,
            args=[x_q.astype(numpy.uint8)],
            check_function=numpy.array_equal,
            verbose=False,
        )
