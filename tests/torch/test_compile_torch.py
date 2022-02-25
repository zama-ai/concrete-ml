"""Tests for the torch to numpy module."""
from functools import partial

import numpy
import pytest
from concrete.common.mlir.utils import ACCEPTABLE_MAXIMAL_BITWIDTH_FROM_CONCRETE_LIB
from torch import nn

# pylint sees separated imports from concrete but does not understand they come from two different
# packages/projects, disable the warning
# pylint: disable=ungrouped-imports
from concrete.ml.torch.compile import compile_torch_model
from concrete.ml.virtual_lib.virtual_fhe_circuit import VirtualFHECircuit

# pylint: enable=ungrouped-imports

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
        return x + self.act(x + 1.0)


@pytest.mark.parametrize(
    "activation_function",
    [
        pytest.param(nn.Sigmoid, id="sigmoid"),
        pytest.param(nn.ReLU, id="relu"),
        pytest.param(nn.ReLU6, id="relu6"),
        pytest.param(nn.Tanh, id="tanh"),
        pytest.param(nn.Hardtanh, id="Hardtanh"),
        pytest.param(nn.CELU, id="CELU"),
        pytest.param(nn.Softplus, id="Softplus"),
        pytest.param(nn.ELU, id="ELU"),
        pytest.param(nn.LeakyReLU, id="LeakyReLU"),
        pytest.param(nn.SELU, id="SELU"),
        # FIXME: to be done, https://github.com/zama-ai/concrete-ml-internal/issues/335
        #   FIXME: because of missing quantized Mul / Div
        # pytest.param(nn.Mish, id="Mish"),
        # pytest.param(nn.SiLU, id="SiLU"),
        # pytest.param(nn.GELU, id="GELU"),
        # pytest.param(nn.Softsign, id="Softsign"),
        #   FIXME: divisions by 0
        # pytest.param(nn.LogSigmoid, id="LogSigmoid"),
    ],
)
@pytest.mark.parametrize(
    "model",
    [
        pytest.param(FC),
        pytest.param(partial(NetWithLoops, n_fc_layers=2)),
        pytest.param(BranchingModule),
    ],
)
@pytest.mark.parametrize(
    "input_output_feature",
    [pytest.param(input_output_feature) for input_output_feature in INPUT_OUTPUT_FEATURE],
)
@pytest.mark.parametrize("use_virtual_lib", [True, False])
def test_compile_torch(
    input_output_feature,
    model,
    activation_function,
    default_compilation_configuration,
    use_virtual_lib,
):
    """Test the different model architecture from torch numpy."""

    n_bits = 2

    # Define an input shape (n_examples, n_features)
    n_examples = 50

    # Define the torch model
    torch_fc_model = model(input_output_feature, activation_function=activation_function)

    # Create random input
    inputset = numpy.random.uniform(-100, 100, size=(n_examples, input_output_feature))

    # Compile
    quantized_numpy_module = compile_torch_model(
        torch_fc_model,
        inputset,
        default_compilation_configuration,
        n_bits=n_bits,
        use_virtual_lib=use_virtual_lib,
    )

    # pylint does not understand that we have a VirtualFHECircuit so disable its warning here
    if use_virtual_lib:
        # pylint: disable=no-member
        assert isinstance(quantized_numpy_module.forward_fhe, VirtualFHECircuit)
        check_ok, _, _ = quantized_numpy_module.forward_fhe.check_circuit_uses_n_bits_or_less(0)
        assert not check_ok
        # TODO: https://github.com/zama-ai/concrete-ml-internal/issues/404
        # re-enable once https://github.com/zama-ai/concrete-ml-internal/issues/274 is done
        # check_ok, _, _ = quantized_numpy_module.forward_fhe.check_circuit_uses_n_bits_or_less(
        #     ACCEPTABLE_MAXIMAL_BITWIDTH_FROM_CONCRETE_LIB
        # )
        # assert check_ok
        # pylint: enable=no-member

    # Create test data from the same distribution and quantize using
    # learned quantization parameters during compilation
    x_test = numpy.random.uniform(-100, 100, size=(1, input_output_feature))
    qtest = quantized_numpy_module.quantize_input(x_test)
    assert quantized_numpy_module.is_compiled
    quantized_numpy_module.forward_fhe.run(qtest)

    # FHE vs Quantized are not done in the test anymore (see issue #177)

    if not use_virtual_lib:
        return

    # If we are in the virtual lib test also test a bit width that should not be supported
    n_bits = ACCEPTABLE_MAXIMAL_BITWIDTH_FROM_CONCRETE_LIB + 1

    # Let's be sure to request something that the compiler does not normaly support.
    assert n_bits > ACCEPTABLE_MAXIMAL_BITWIDTH_FROM_CONCRETE_LIB

    # Compile again with more bits
    quantized_numpy_module = compile_torch_model(
        torch_fc_model,
        inputset,
        default_compilation_configuration,
        n_bits=n_bits,
        use_virtual_lib=use_virtual_lib,
    )

    assert isinstance(quantized_numpy_module.forward_fhe, VirtualFHECircuit)
    # pylint does not understand that we have a VirtualFHECircuit so disable its warning here
    # Check we went overboard for the number of bits
    # pylint: disable=no-member
    (
        check_ok,
        max_bit_width,
        _,
    ) = quantized_numpy_module.forward_fhe.check_circuit_uses_n_bits_or_less(
        ACCEPTABLE_MAXIMAL_BITWIDTH_FROM_CONCRETE_LIB
    )
    # pylint: enable=no-member
    assert not check_ok
    assert max_bit_width > ACCEPTABLE_MAXIMAL_BITWIDTH_FROM_CONCRETE_LIB

    # Check the forward works with the high bitwidth
    qtest = quantized_numpy_module.quantize_input(x_test)
    assert quantized_numpy_module.is_compiled
    quantized_numpy_module.forward_fhe.run(qtest)
