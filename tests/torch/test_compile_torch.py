"""Tests for the torch to numpy module."""
import io
import tempfile
import zipfile
from functools import partial
from inspect import signature
from pathlib import Path

import numpy
import onnx
import onnxruntime as ort
import pytest
import torch
import torch.quantization
from torch import nn

from concrete.ml.onnx.convert import OPSET_VERSION_FOR_ONNX_EXPORT

# pylint sees separated imports from concrete but does not understand they come from two different
# packages/projects, disable the warning
# pylint: disable=ungrouped-imports
from concrete.ml.torch.compile import compile_onnx_model, compile_torch_model

# pylint: enable=ungrouped-imports

# INPUT_OUTPUT_FEATURE is the number of input and output of each of the network layers.
# (as well as the input of the network itself)
# Note that when comparing two predictions with few features the r2 score is brittle
# thus we prefer to avoid values that are too low (e.g. 1, 2)
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


class SimpleQAT(nn.Module):
    """Torch model implements a step function that needs Greater, Cast and Where."""

    def __init__(self, _n_feat, activation_function, n_bits=2):
        super().__init__()

        self.act = activation_function()
        self.fc1 = nn.Linear(_n_feat, _n_feat)

        # Create pre-quantized weights
        # Note the weights in the network are not integers, but uniformly spaced float values
        # that are selected from a discrete set
        weight_scale = 1.5

        n_bits_weights = n_bits

        # Generate the pattern 0, 1, ..., 2^N-1, 0, 1, .. 2^N-1, 0, 1..
        all_weights = numpy.mod(
            numpy.arange(numpy.prod(self.fc1.weight.shape)), 2**n_bits_weights
        )

        # Shuffle the pattern and reshape to weight shape
        numpy.random.shuffle(all_weights)
        int_weights = all_weights.reshape(self.fc1.weight.shape)

        # Ensure we have the correct max/min that produces the correct scale in Quantized Array
        assert numpy.max(int_weights) - numpy.min(int_weights) == (2**n_bits_weights - 1)

        # We want signed weights, so offset the generated weights
        int_weights = int_weights - 2 ** (n_bits_weights - 1)

        # Initialize with scaled float weights
        self.fc1.weight.data = torch.from_numpy(int_weights * weight_scale).float()

        self.n_bits = n_bits

    def forward(self, x):
        """Forward pass with a quantizer built into the computation graph."""

        def step(x, bias):
            """The step function for quantization."""
            y = torch.zeros_like(x)
            mask = torch.gt(x - bias, 0.0)
            y[mask] = 1.0
            return y

        # A step quantizer with steps at -5, 0, 5, ...
        # For example at n_bits == 2
        #         /  0  if x < -5                \
        # f(x) = {   5  if x >= 0 and x < 5       }
        #         \  10 if x >= 5 and x < 10     /
        #          \ 15 if x >= 10              /

        x_q = step(x, -5)
        for i in range(1, 2**self.n_bits - 1):
            x_q += step(x, (i - 1) * 5)

        x_q = x_q.mul(5)

        result_fc1 = self.fc1(x_q)

        return self.act(result_fc1)


def compile_and_test_torch_or_onnx(  # pylint: disable=too-many-locals, too-many-statements
    input_output_feature,
    model,
    activation_function,
    qat_bits,
    default_configuration,
    use_virtual_lib,
    is_onnx,
    dump_onnx=False,
    expected_onnx_str=None,
):
    """Test the different model architecture from torch numpy."""

    # Define an input shape (n_examples, n_features)
    n_examples = 50

    # Define the torch model
    if not isinstance(input_output_feature, tuple):
        input_output_feature = (input_output_feature,)

    torch_model = model(input_output_feature[0], activation_function=activation_function)

    num_inputs = len(signature(torch_model.forward).parameters)

    # Create random input
    inputset = (
        tuple(
            numpy.random.uniform(-100, 100, size=(n_examples, *input_output_feature))
            for _ in range(num_inputs)
        )
        if num_inputs > 1
        else numpy.random.uniform(-100, 100, size=(n_examples, *input_output_feature))
    )

    # FHE vs Quantized are not done in the test anymore (see issue #177)
    if not use_virtual_lib:
        n_bits = (
            {"net_inputs": 2, "net_outputs": 2, "op_inputs": 2, "op_weights": 2}
            if qat_bits == 0
            else qat_bits
        )

        if is_onnx:

            output_onnx_file_path = Path(tempfile.mkstemp(suffix=".onnx")[1])
            inputset_as_numpy_tuple = (
                (val for val in inputset) if isinstance(inputset, tuple) else (inputset,)
            )
            dummy_input = tuple(
                torch.from_numpy(val[[0], ::]).float() for val in inputset_as_numpy_tuple
            )
            torch.onnx.export(
                torch_model,
                dummy_input,
                str(output_onnx_file_path),
                opset_version=OPSET_VERSION_FOR_ONNX_EXPORT,
            )
            onnx_model = onnx.load_model(output_onnx_file_path)
            onnx.checker.check_model(onnx_model)

            quantized_numpy_module = compile_onnx_model(
                onnx_model,
                inputset,
                import_qat=qat_bits != 0,
                configuration=default_configuration,
                n_bits=n_bits,
                use_virtual_lib=use_virtual_lib,
            )
        else:
            quantized_numpy_module = compile_torch_model(
                torch_model,
                inputset,
                import_qat=qat_bits != 0,
                configuration=default_configuration,
                n_bits=n_bits,
                use_virtual_lib=use_virtual_lib,
            )

        # Create test data from the same distribution and quantize using
        # learned quantization parameters during compilation
        x_test = tuple(
            numpy.random.uniform(-100, 100, size=(1, *input_output_feature))
            for _ in range(num_inputs)
        )
        qtest = quantized_numpy_module.quantize_input(*x_test)
        if not isinstance(qtest, tuple):
            qtest = (qtest,)
        assert quantized_numpy_module.is_compiled
        quantized_numpy_module.forward_fhe.encrypt_run_decrypt(*qtest)
    else:
        # Compile our network with 16 bits
        # to compare to torch (8b weights + float 32 activations)
        if qat_bits == 0:
            n_bits = 16
        else:
            n_bits = {
                "net_inputs": 16,
                "op_weights": qat_bits,
                "op_inputs": qat_bits,
                "net_outputs": 16,
            }

        # Compile with higher quantization bitwidth
        quantized_numpy_module = compile_torch_model(
            torch_model,
            inputset,
            import_qat=qat_bits != 0,
            configuration=default_configuration,
            n_bits=n_bits,
            use_virtual_lib=use_virtual_lib,
        )

        # Create test data from the same distribution and quantize using.
        n_examples_test = 100
        x_test = tuple(
            numpy.random.uniform(-100, 100, size=(n_examples_test, *input_output_feature))
            for _ in range(num_inputs)
        )

        # Check the forward works with the high bitwidth
        qtest = quantized_numpy_module.quantize_input(*x_test)
        if not isinstance(qtest, tuple):
            qtest = (qtest,)
        assert quantized_numpy_module.is_compiled
        results = []
        for i in range(n_examples_test):
            q_x = tuple(qtest[input][[i]] for input in range(len(qtest)))
            q_result = quantized_numpy_module.forward_fhe.encrypt_run_decrypt(*q_x)
            result = quantized_numpy_module.dequantize_output(q_result)
            results.append(result)

        # FIXME implement a test specific to torch vs CML quantization and use it here #1248
        # See commit 6c40a0c7c3765012766592a74db99430c5a373e5 for
        # the implementation of dynamic quantization.
        # (ref issues: #1230 #1218 #1147 #1073)

        onnx_model = quantized_numpy_module.onnx_model

        if dump_onnx:
            str_model = onnx.helper.printable_graph(onnx_model.graph)
            print("ONNX model:")
            print(str_model)
            assert str_model == expected_onnx_str


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
@pytest.mark.parametrize("is_onnx", [True, False])
def test_compile_torch_or_onnx_networks(
    input_output_feature,
    model,
    activation_function,
    default_configuration,
    use_virtual_lib,
    is_onnx,
    is_vl_only_option,
):
    """Test the different model architecture from torch numpy."""
    if not use_virtual_lib and is_vl_only_option:
        print("Warning, skipping non VL tests")
        return

    # To signal that this network is not using QAT set the QAT bits to 0
    qat_bits = 0

    compile_and_test_torch_or_onnx(
        input_output_feature,
        model,
        activation_function,
        qat_bits,
        default_configuration,
        use_virtual_lib,
        is_onnx,
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
@pytest.mark.parametrize("is_onnx", [True, False])
def test_compile_torch_or_onnx_conv_networks(  # pylint: disable=unused-argument
    model,
    activation_function,
    default_configuration,
    use_virtual_lib,
    is_onnx,
    is_vl_only_option,
):
    """Test the different model architecture from torch numpy."""
    if not use_virtual_lib and is_vl_only_option:
        print("Warning, skipping non VL tests")
        return

    # To signal that this network is not using QAT set the QAT bits to 0
    qat_bits = 0

    compile_and_test_torch_or_onnx(
        (1, 7, 7),
        model,
        activation_function,
        qat_bits,
        default_configuration,
        use_virtual_lib,
        is_onnx,
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
        pytest.param(nn.SiLU, id="SiLU"),  # Sometimes bad accuracy
        pytest.param(nn.Mish, id="Mish"),
        pytest.param(nn.Tanhshrink, id="Tanhshrink"),
        pytest.param(partial(nn.Threshold, threshold=0, value=0), id="Threshold"),
        pytest.param(nn.Softshrink, id="Softshrink"),
        pytest.param(nn.Hardshrink, id="Hardshrink"),
        pytest.param(nn.Softsign, id="Softsign"),
        pytest.param(nn.GELU, id="GELU"),
        # FIXME, #335: still some issues with these activations
        #
        # - Works but sometimes issues with the accuracy
        # pytest.param(nn.LogSigmoid, id="LogSigmoid"),
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
@pytest.mark.parametrize("is_onnx", [True, False])
def test_compile_torch_or_onnx_activations(
    input_output_feature,
    model,
    activation_function,
    default_configuration,
    use_virtual_lib,
    is_onnx,
    is_vl_only_option,
):
    """Test the different model architecture from torch numpy."""
    if not use_virtual_lib and is_vl_only_option:
        print("Warning, skipping non VL tests")
        return

    # To signal that this network is not using QAT set the QAT bits to 0
    qat_bits = 0

    compile_and_test_torch_or_onnx(
        input_output_feature,
        model,
        activation_function,
        qat_bits,
        default_configuration,
        use_virtual_lib,
        is_onnx,
    )


@pytest.mark.parametrize(
    "model",
    [
        pytest.param(SimpleQAT),
    ],
)
@pytest.mark.parametrize(
    "input_output_feature",
    [pytest.param(input_output_feature) for input_output_feature in [2, 4]],
)
@pytest.mark.parametrize(
    "n_bits",
    [pytest.param(n_bits) for n_bits in [1, 2]],
)
@pytest.mark.parametrize("use_virtual_lib", [True, False])
def test_compile_torch_qat(
    input_output_feature,
    model,
    n_bits,
    default_configuration,
    use_virtual_lib,
    is_vl_only_option,
):
    """Test the different model architecture from torch numpy."""
    if not use_virtual_lib and is_vl_only_option:
        print("Warning, skipping non VL tests")
        return

    model = partial(model, n_bits=n_bits)

    # Import these networks from torch directly
    is_onnx = False
    qat_bits = n_bits

    compile_and_test_torch_or_onnx(
        input_output_feature,
        model,
        nn.Sigmoid,
        qat_bits,
        default_configuration,
        use_virtual_lib,
        is_onnx,
    )


@pytest.mark.parametrize(
    "model, expected_onnx_str",
    [
        pytest.param(
            FC,
            """graph torch_jit (
  %onnx::Gemm_0[FLOAT, 1x7]
) initializers (
  %fc1.weight[FLOAT, 7x7]
  %fc1.bias[FLOAT, 7]
  %fc2.weight[FLOAT, 7x7]
  %fc2.bias[FLOAT, 7]
) {
  %input = Gemm[alpha = 1, beta = 1, transB = 1](%onnx::Gemm_0, %fc1.weight, %fc1.bias)
  %onnx::Gemm_6 = Relu(%input)
  %7 = Gemm[alpha = 1, beta = 1, transB = 1](%onnx::Gemm_6, %fc2.weight, %fc2.bias)
  return %7
}""",
        ),
    ],
)
@pytest.mark.parametrize(
    "activation_function",
    [
        pytest.param(nn.ReLU, id="relu"),
    ],
)
def test_dump_torch_network(
    model,
    expected_onnx_str,
    activation_function,
    default_configuration,
):
    """This is a test which is equivalent to tests in test_dump_onnx.py, but for torch modules."""
    input_output_feature = 7
    use_virtual_lib = True
    is_onnx = False
    qat_bits = 0

    compile_and_test_torch_or_onnx(
        input_output_feature,
        model,
        activation_function,
        qat_bits,
        default_configuration,
        use_virtual_lib,
        is_onnx,
        dump_onnx=True,
        expected_onnx_str=expected_onnx_str,
    )


def test_pretrained_mnist_qat(default_configuration, check_accuracy):
    """Load a QAT MNIST model and make sure we get the same results in VL as with ONNX."""

    onnx_file_path = "tests/data/mnist_2b_s1_1.zip"
    mnist_test_path = "tests/data/mnist_test_batch.zip"

    # Load ONNX model from zip file
    with zipfile.ZipFile(onnx_file_path, "r") as archive_model:
        onnx_model_serialized = io.BytesIO(archive_model.read("mnist_2b_s1_1.onnx")).read()
        onnx_model = onnx.load_model_from_string(onnx_model_serialized)

    onnx.checker.check_model(onnx_model)

    # Load test data and ground truth from zip file
    with zipfile.ZipFile(mnist_test_path, "r") as archive_data:
        mnist_data = numpy.load(
            io.BytesIO(archive_data.read("mnist_test_batch.npy")), allow_pickle=True
        ).item()

    # Get the test data
    inputset = mnist_data["test_data"]

    # Run through ONNX runtime and collect results
    ort_session = ort.InferenceSession(onnx_model_serialized)

    onnx_results = numpy.zeros((inputset.shape[0],), dtype=numpy.int64)
    for i, x_test in enumerate(inputset):
        onnx_outputs = ort_session.run(
            None,
            {onnx_model.graph.input[0].name: x_test.reshape(1, -1)},
        )
        onnx_results[i] = numpy.argmax(onnx_outputs[0])

    # Compile to Concrete ML with the Virtual Library, with a high bitwidth
    n_bits = {
        "net_inputs": 16,
        "op_weights": 2,
        "op_inputs": 2,
        "net_outputs": 16,
    }

    quantized_numpy_module = compile_onnx_model(
        onnx_model,
        inputset,
        import_qat=True,
        configuration=default_configuration,
        n_bits=n_bits,
        use_virtual_lib=True,
    )

    num_inputs = 1

    # Create test data tuple
    x_test = tuple(inputset for _ in range(num_inputs))

    # Check the forward works with the high bitwidth
    qtest = quantized_numpy_module.quantize_input(*x_test)
    if not isinstance(qtest, tuple):
        qtest = (qtest,)

    assert quantized_numpy_module.is_compiled

    # Collect VL results
    results = []
    for i in range(inputset.shape[0]):
        q_x = tuple(qtest[input][[i]] for input in range(len(qtest)))
        q_result = quantized_numpy_module.forward_fhe.encrypt_run_decrypt(*q_x)
        result = quantized_numpy_module.dequantize_output(q_result)
        result = numpy.argmax(result)
        results.append(result)

    # Compare ONNX runtime vs Virtual Lib
    check_accuracy(onnx_results, results, threshold=0.999)

    # Make sure absolute accuracy is good, this model should have at least 90% accuracy
    check_accuracy(mnist_data["gt"], results, threshold=0.9)

    # Compile to Concrete ML with the Virtual Library with an FHE compatible bitwidth
    n_bits = {
        "net_inputs": 7,
        "op_weights": 2,
        "op_inputs": 2,
        "net_outputs": 7,
    }

    quantized_numpy_module = compile_onnx_model(
        onnx_model,
        inputset,
        import_qat=True,
        configuration=default_configuration,
        n_bits=n_bits,
        use_virtual_lib=False,
    )

    assert quantized_numpy_module.forward_fhe.graph.maximum_integer_bit_width() <= 8
