"""Tests for the torch to numpy module."""
from functools import partial
from inspect import signature

import numpy
import pytest
import torch
import torch.quantization
from torch import nn

# pylint sees separated imports from concrete but does not understand they come from two different
# packages/projects, disable the warning
# pylint: disable=ungrouped-imports
from concrete.ml.torch.compile import compile_torch_model

# pylint: enable=ungrouped-imports

# INPUT_OUTPUT_FEATURE is the number of input and output of each of the network layers.
# (as well as the input of the network itself)
# Note that when comparing two predictions with few features the r2 score is brittle
# thus we prefer to avoid values that are too low (eg. 1, 2)
INPUT_OUTPUT_FEATURE = [5, 10]


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


class CNN(nn.Module):
    """Torch CNN model for the tests."""

    def __init__(self, input_output, activation_function):
        super().__init__()

        self.activation_function = activation_function()
        self.conv1 = nn.Conv2d(input_output, 3, 3)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 3, 1)
        self.fc1 = nn.Linear(3 * 2 * 2, 5)
        self.fc2 = nn.Linear(5, 3)
        self.fc3 = nn.Linear(3, 2)

    def forward(self, x):
        """Forward pass."""
        x = self.pool(self.activation_function(self.conv1(x)))
        x = self.activation_function(self.conv2(x))
        x = x.flatten(1)
        x = self.activation_function(self.fc1(x))
        x = self.activation_function(self.fc2(x))
        x = self.fc3(x)
        return x


class MultiInputNN(nn.Module):
    """Torch model to test multiple inputs forward."""

    def __init__(self, input_output, activation_function):  # pylint: disable=unused-argument
        super().__init__()
        self.act = activation_function()

    def forward(self, x, y):
        """Forward pass."""
        return self.act(x + y)


class NetWithLoops(nn.Module):
    """Torch model, where we reuse some elements in a loop in the forward and don't expect the
    user to define these elements in a particular order"""

    def __init__(self, n_feat, activation_function, n_fc_layers):
        super().__init__()
        self.ifc = nn.Sequential()
        for i in range(n_fc_layers):
            self.ifc.add_module(f"fc{i+1}", nn.Linear(n_feat, n_feat))
        self.act = activation_function()

    def forward(self, x):
        """Forward pass."""
        for m in self.ifc:
            x = self.act(m(x))

        return x


class BranchingModule(nn.Module):
    """Torch model with some branching and skip connections."""

    def __init__(self, _n_feat, activation_function):
        super().__init__()

        self.act = activation_function()

    def forward(self, x):
        """Forward pass."""
        return x + self.act(x + 1.0) - self.act(x * 2.0)


class BranchingGemmModule(nn.Module):
    """Torch model with some branching and skip connections."""

    def __init__(self, _n_feat, activation_function):
        super().__init__()

        self.act = activation_function()
        self.fc1 = nn.Linear(_n_feat, _n_feat)

    def forward(self, x):
        """Forward pass."""
        return x + self.act(x + 1.0) - self.act(self.fc1(x * 2.0))


class UnivariateModule(nn.Module):
    """Torch model that calls univariate and shape functions of torch."""

    def __init__(self, _n_feat, activation_function):
        super().__init__()

        self.act = activation_function()

    def forward(self, x):
        """Forward pass."""
        x = x.view(-1, 1)
        x = torch.reshape(x, (-1, 1))
        x = x.flatten(1)
        x = self.act(torch.abs(torch.exp(torch.log(1.0 + torch.sigmoid(x)))))
        return x


class StepActivationModule(nn.Module):
    """Torch model implements a step function that needs Greater, Cast and Where."""

    def __init__(self, _n_feat, activation_function):
        super().__init__()

        self.act = activation_function()

    def forward(self, x):
        """Forward pass with a quantizer built into the computation graph."""

        def step(x, bias):
            """The step function for quantization."""
            y = torch.zeros_like(x)
            mask = torch.gt(x - bias, 0.0)
            y[mask] = 1.0
            return y

        x = step(x, 0.5) * 2.0
        x = self.act(x)
        return x


def compile_and_test_torch(
    input_output_feature,
    model,
    activation_function,
    default_configuration,
    use_virtual_lib,
    check_r2_score,
):
    """Test the different model architecture from torch numpy."""

    n_bits = 2

    # Define an input shape (n_examples, n_features)
    n_examples = 50

    # Define the torch model
    if not isinstance(input_output_feature, tuple):
        input_output_feature = (input_output_feature,)

    torch_fc_model = model(input_output_feature[0], activation_function=activation_function)

    num_inputs = len(signature(torch_fc_model.forward).parameters)

    # Create random input
    inputset = (
        tuple(
            numpy.random.uniform(-100, 100, size=(n_examples, *input_output_feature))
            for _ in range(num_inputs)
        )
        if num_inputs > 1
        else numpy.random.uniform(-100, 100, size=(n_examples, *input_output_feature))
    )

    # Compile
    quantized_numpy_module = compile_torch_model(
        torch_fc_model,
        inputset,
        default_configuration,
        n_bits=n_bits,
        use_virtual_lib=use_virtual_lib,
    )

    # Create test data from the same distribution and quantize using
    # learned quantization parameters during compilation
    x_test = tuple(
        numpy.random.uniform(-100, 100, size=(1, *input_output_feature)) for _ in range(num_inputs)
    )
    qtest = quantized_numpy_module.quantize_input(*x_test)
    if not isinstance(qtest, tuple):
        qtest = (qtest,)
    assert quantized_numpy_module.is_compiled
    quantized_numpy_module.forward_fhe.encrypt_run_decrypt(*qtest)

    # FHE vs Quantized are not done in the test anymore (see issue #177)

    if not use_virtual_lib:
        return

    # Compile our network with 16 bits
    # to compare to torch (8b weights + float 32 activations)
    n_bits = 16

    # Compile with higher quantization bitwidth
    quantized_numpy_module = compile_torch_model(
        torch_fc_model,
        inputset,
        default_configuration,
        n_bits=n_bits,
        use_virtual_lib=use_virtual_lib,
    )

    # Check the forward works with the high bitwidth
    qtest = quantized_numpy_module.quantize_input(*x_test)
    if not isinstance(qtest, tuple):
        qtest = (qtest,)
    assert quantized_numpy_module.is_compiled
    q_result = quantized_numpy_module.forward_fhe.encrypt_run_decrypt(*qtest)
    result = quantized_numpy_module.dequantize_output(q_result)

    # Run the network through torch, using dynamic quantization
    # This mode only quantizes the weights and keeps all activations in float32
    # Our quantization approach quantizes the activations for some layers and fuses all other
    # float32 computations to table lookups. Thus the outputs from torch and Concrete ML
    # will not match, but they should be close
    # see: https://pytorch.org/tutorials/recipes/recipes/dynamic_quantization.html
    torch_quantized_model = torch.quantization.quantize_dynamic(
        torch_fc_model, {nn.Linear}, dtype=torch.qint8
    )
    torch_input = (torch.from_numpy(x).float() for x in x_test)
    torch_result = torch_quantized_model(*torch_input).numpy()

    # Check that we have similar results between CML and torch in 8 bits
    # Due to differences in quantization approach, we allow a lower R2 in the comparison
    check_r2_score(result, torch_result, 0.9)


@pytest.mark.parametrize(
    "activation_function",
    [
        pytest.param(nn.ReLU, id="relu"),
    ],
)
@pytest.mark.parametrize(
    "model",
    [
        pytest.param(FC),
        pytest.param(partial(NetWithLoops, n_fc_layers=2)),
        pytest.param(BranchingModule),
        pytest.param(BranchingGemmModule),
        pytest.param(MultiInputNN),
        pytest.param(UnivariateModule),
        pytest.param(StepActivationModule),
    ],
)
@pytest.mark.parametrize(
    "input_output_feature",
    [pytest.param(input_output_feature) for input_output_feature in INPUT_OUTPUT_FEATURE],
)
@pytest.mark.parametrize("use_virtual_lib", [True, False])
def test_compile_torch_networks(
    input_output_feature,
    model,
    activation_function,
    default_configuration,
    use_virtual_lib,
    check_r2_score,
):
    """Test the different model architecture from torch numpy."""
    compile_and_test_torch(
        input_output_feature,
        model,
        activation_function,
        default_configuration,
        use_virtual_lib,
        check_r2_score,
    )


@pytest.mark.parametrize(
    "activation_function",
    [
        pytest.param(nn.ReLU, id="relu"),
    ],
)
@pytest.mark.parametrize(
    "model",
    [
        pytest.param(CNN),
    ],
)
@pytest.mark.parametrize("use_virtual_lib", [True, False])
def test_compile_torch_conv_networks(  # pylint: disable=unused-argument
    model,
    activation_function,
    default_configuration,
    use_virtual_lib,
    check_r2_score,
):
    """Test the different model architecture from torch numpy."""

    compile_and_test_torch(
        (1, 7, 7),
        model,
        activation_function,
        default_configuration,
        use_virtual_lib,
        check_r2_score,
    )


@pytest.mark.parametrize(
    "activation_function",
    [
        pytest.param(nn.Sigmoid, id="sigmoid"),
        pytest.param(nn.ReLU, id="relu"),
        pytest.param(nn.ReLU6, id="relu6"),
        pytest.param(nn.Tanh, id="tanh"),
        pytest.param(nn.ELU, id="ELU"),
        pytest.param(nn.Hardsigmoid, id="Hardsigmoid"),
        pytest.param(nn.Hardtanh, id="Hardtanh"),
        pytest.param(nn.LeakyReLU, id="LeakyReLU"),
        pytest.param(nn.SELU, id="SELU"),
        pytest.param(nn.CELU, id="CELU"),
        pytest.param(nn.Softplus, id="Softplus"),
        pytest.param(nn.PReLU, id="PReLU"),
        pytest.param(nn.Hardswish, id="Hardswish"),
        # FIXME: to be done, https://github.com/zama-ai/concrete-ml-internal/issues/335
        #   FIXME: because of missing quantized ONNX ops (e.g. Div)
        # pytest.param(nn.Tanhshrink, id="Tanhshrink"),
        # pytest.param(nn.Hardshrink, id="Hardshrink"),
        # pytest.param(nn.Mish, id="Mish"),
        # pytest.param(nn.SiLU, id="SiLU"),
        # pytest.param(nn.GELU, id="GELU"),
        # pytest.param(nn.Softsign, id="Softsign"),
        # pytest.param(nn.LogSigmoid, id="LogSigmoid"),
    ],
)
@pytest.mark.parametrize(
    "model",
    [
        pytest.param(FC),
    ],
)
@pytest.mark.parametrize(
    "input_output_feature",
    [pytest.param(input_output_feature) for input_output_feature in INPUT_OUTPUT_FEATURE],
)
@pytest.mark.parametrize("use_virtual_lib", [True, False])
def test_compile_torch_activations(
    input_output_feature,
    model,
    activation_function,
    default_configuration,
    use_virtual_lib,
    check_r2_score,
):
    """Test the different model architecture from torch numpy."""
    compile_and_test_torch(
        input_output_feature,
        model,
        activation_function,
        default_configuration,
        use_virtual_lib,
        check_r2_score,
    )
