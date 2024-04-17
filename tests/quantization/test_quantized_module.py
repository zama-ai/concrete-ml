"""Tests for the quantized module."""

from functools import partial

import numpy
import pytest
import torch
from torch import nn

from concrete.ml.pytest.torch_models import CNN, FC, CNNMaxPool
from concrete.ml.pytest.utils import check_serialization, values_are_equal
from concrete.ml.quantization import PostTrainingAffineQuantization, QuantizedModule
from concrete.ml.torch import NumpyModule
from concrete.ml.torch.compile import compile_torch_model

N_BITS_LIST = [
    20,
    16,
    8,
    {"model_inputs": 8, "op_weights": 8, "op_inputs": 8, "model_outputs": 8},
    {"model_inputs": 12, "op_weights": 15, "op_inputs": 15, "model_outputs": 16},
]


@pytest.mark.parametrize("n_bits", [pytest.param(n_bits) for n_bits in N_BITS_LIST])
@pytest.mark.parametrize(
    "model_class, input_shape",
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
        pytest.param(nn.LogSigmoid, id="LogSigmoid"),
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
def test_quantized_linear(model_class, input_shape, n_bits, activation_function, check_r2_score):
    """Test the quantized module with a post-training static quantization.

    With n_bits>>0 we expect the results of the quantized module
    to be the same as the standard module.
    """
    # Define the torch model
    torch_fc_model = model_class(activation_function=activation_function)
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
    quantized_module = post_training_quant.quantize_module(numpy_input)

    # Execution the predictions in the clear using the __call__ method
    dequant_prediction_call = quantized_module(numpy_input)

    # Execution the predictions in the clear using the forward method
    dequant_prediction_forward = quantized_module.forward(numpy_input, fhe="disable")

    # Both prediction values should be equal
    assert numpy.array_equal(dequant_prediction_call, dequant_prediction_forward)

    # Check that the actual prediction are close to the expected prediction
    check_r2_score(numpy_prediction, dequant_prediction_forward)


@pytest.mark.parametrize(
    "model_class, input_shape",
    [
        pytest.param(FC, (100, 32 * 32 * 3)),
    ],
)
@pytest.mark.parametrize(
    "method, dtype, error_message",
    [
        pytest.param(
            "quantized_forward",
            numpy.float32,
            "Input values are expected to be integers of dtype .*",
        ),
        pytest.param(
            "forward",
            numpy.int64,
            "Input values are expected to be floating points of dtype .*",
        ),
    ],
)
def test_raises_on_inputs_with_wrong_dtypes(model_class, input_shape, method, dtype, error_message):
    """Function to test incompatible inputs."""

    # Define the torch model
    torch_fc_model = model_class(nn.ReLU)

    # Create random input
    numpy_input = numpy.random.uniform(size=input_shape)

    # Create corresponding numpy model
    numpy_fc_model = NumpyModule(torch_fc_model, torch.from_numpy(numpy_input).float())

    # Quantize with post-training static method
    post_training_quant = PostTrainingAffineQuantization(8, numpy_fc_model)
    quantized_model = post_training_quant.quantize_module(numpy_input)

    with pytest.raises(
        TypeError,
        match=error_message,
    ):
        forward_method = getattr(quantized_model, method)
        forward_method(numpy_input.astype(dtype))


@pytest.mark.parametrize("n_bits", [2, 7])
@pytest.mark.parametrize(
    "model_class, input_shape",
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
def test_intermediary_values(n_bits, model_class, input_shape, activation_function):
    """Check that all torch linear/conv layers return debug values in the QuantizedModule."""

    torch_fc_model = model_class(activation_function=activation_function)
    torch_fc_model.eval()

    # Create random input
    numpy_input = numpy.random.uniform(size=input_shape)

    torch_input = torch.from_numpy(numpy_input).float()
    # Create corresponding numpy model
    numpy_fc_model = NumpyModule(torch_fc_model, torch_input)

    # Quantize with post-training static method
    post_training_quant = PostTrainingAffineQuantization(n_bits, numpy_fc_model)
    quantized_model = post_training_quant.quantize_module(numpy_input)

    # Execute the forward pass in the clear
    _, debug_values = quantized_model.forward(numpy_input, debug=True, fhe="disable")

    # Count the number of Gemm/Conv layers in the Concrete ML debug values
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

    # Make sure we have debug output for all conv/gemm layers in Concrete ML
    assert num_gemm_conv == num_torch_gemm_conv


@pytest.mark.parametrize(
    "model_class, input_shape",
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
def test_bitwidth_report(model_class, input_shape, activation_function, default_configuration):
    """Check that the quantized module bit-width report executes without error."""
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    numpy.random.seed(42)

    torch_fc_model = model_class(activation_function=activation_function)
    torch_fc_model.eval()

    # Create random input
    numpy_input = numpy.random.uniform(size=input_shape)
    torch_input = torch.from_numpy(numpy_input).float()

    # First test a model that is not compiled
    numpy_fc_model = NumpyModule(torch_fc_model, torch_input)
    post_training_quant = PostTrainingAffineQuantization(2, numpy_fc_model)
    quantized_model = post_training_quant.quantize_module(numpy_input)

    # A QuantizedModule that is not compiled can not report bit-widths and value ranges
    assert quantized_model.bitwidth_and_range_report() is None

    # Finally test a compiled QuantizedModule
    quantized_model = compile_torch_model(
        torch_fc_model,
        torch_input,
        False,
        default_configuration,
        n_bits=2,
        p_error=0.01,
    )

    # Get all ops for which the user would want to know the bit-widths
    ops_check_have_report = set()
    for node in quantized_model.onnx_model.graph.node:
        if "Gemm" in node.op_type or "Conv" in node.op_type:
            ops_check_have_report.add(node.name)

    # Get the report
    op_names_to_report = quantized_model.bitwidth_and_range_report()

    assert (
        op_names_to_report is not None
    ), "Please compile the module before accessing the bitwidth and range reports."

    # Check that all interesting ops are in the report
    assert ops_check_have_report.issubset(op_names_to_report.keys())

    expected_class_to_reports = {
        FC: {
            "/fc1/Gemm": {"range": (-251, 208), "bitwidth": 9},
            "/fc2/Gemm": {"range": (-20, 20), "bitwidth": 6},
            "/fc3/Gemm": {"range": (-14, 15), "bitwidth": 5},
            "/fc4/Gemm": {"range": (-17, 16), "bitwidth": 6},
            "/fc5/Gemm": {"range": (-9, 9), "bitwidth": 5},
        }
    }

    # Check that the report includes the correct information
    for op_name, op_report in op_names_to_report.items():

        # Ensure the report contains the right info
        assert "range" in op_report and "bitwidth" in op_report, (
            f"Report for {op_name} is not correct. Expected `range` and `bitwidth` keys, got "
            f"{op_report.keys()}"
        )

        model_reports = expected_class_to_reports.get(model_class, None)

        # If we have actual values for this trained model
        if model_reports is not None:
            expected_report = None
            for potential_op, potential_report in model_reports.items():
                # Expected key name should be contained in the ONNX layer name
                if potential_op in op_name:
                    expected_report = potential_report
                    break

            # Ensure we find the expected layer name though it maybe be transformed
            # with a prefix or suffix
            assert expected_report is not None, (
                f"No proper report has been found for {model_class.__name__}'s expected {op_name} "
                f"operator. Got {model_reports.keys()}."
            )

            # Ensure the value are the expected ones
            assert isinstance(expected_report, dict)
            assert isinstance(op_report, dict)
            op_report_range = op_report["range"]
            expected_report_range = expected_report["range"]
            assert isinstance(op_report_range, (tuple, list))
            assert isinstance(expected_report_range, (tuple, list))
            assert op_report_range[0] == expected_report_range[0]
            assert op_report_range[1] == expected_report_range[1]
            assert op_report["bitwidth"] == expected_report["bitwidth"]


@pytest.mark.parametrize("model_class, input_shape", [pytest.param(FC, (100, 32 * 32 * 3))])
def test_quantized_module_rounding_fhe(model_class, input_shape, default_configuration):
    """Check that rounding is only allowed in simulation mode."""

    torch_fc_model = model_class(activation_function=nn.ReLU)
    torch_fc_model.eval()

    # Create random input
    numpy_input = numpy.random.uniform(size=input_shape)
    torch_input = torch.from_numpy(numpy_input).float()
    assert isinstance(torch_input, torch.Tensor)
    assert isinstance(numpy_input, numpy.ndarray)
    numpy_test = numpy_input[:1]

    # First test a model that is not compiled
    numpy_fc_model = NumpyModule(torch_fc_model, torch_input)
    post_training_quant = PostTrainingAffineQuantization(2, numpy_fc_model)
    quantized_model = post_training_quant.quantize_module(numpy_input)

    # Compile with rounding activated
    quantized_model = compile_torch_model(
        torch_fc_model,
        torch_input,
        False,
        default_configuration,
        n_bits=2,
        p_error=0.01,
        rounding_threshold_bits=6,
    )

    # Run quantized_model in simulation
    quantized_model.forward(numpy_test, fhe="simulate")

    # Execute the model with rounding in FHE execution mode
    quantized_model.forward(numpy_test, fhe="execute")

    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3800


# Extend this test with multi-input encryption status
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4011
@pytest.mark.parametrize("model_class, input_shape", [pytest.param(FC, (100, 32 * 32 * 3))])
def test_inputs_encryption_status(model_class, input_shape, default_configuration):
    """Check that giving inputs_encryption_status work properly."""

    torch_fc_model = model_class(activation_function=nn.ReLU)
    torch_fc_model.eval()

    # Create random input
    numpy_input = numpy.random.uniform(size=input_shape)
    torch_input = torch.from_numpy(numpy_input).float()
    assert isinstance(torch_input, torch.Tensor)
    assert isinstance(numpy_input, numpy.ndarray)

    # Compile with rounding activated
    with pytest.raises(ValueError, match="Missing arguments from '.*', expected 1 arguments."):
        # Empty status
        compile_torch_model(
            torch_fc_model,
            torch_input,
            False,
            default_configuration,
            n_bits=2,
            p_error=0.01,
            rounding_threshold_bits=6,
            inputs_encryption_status=[],
        )

    # Wrong encryption status
    with pytest.raises(
        ValueError, match="Unexpected status from '.*', expected 'clear' or 'encrypted'."
    ):
        compile_torch_model(
            torch_fc_model,
            torch_input,
            False,
            default_configuration,
            n_bits=2,
            p_error=0.01,
            rounding_threshold_bits=6,
            inputs_encryption_status=("random",),
        )

    # Additional argument (error from Concrete Python)
    with pytest.raises(ValueError, match="Too many arguments in '.*', expected 1 arguments."):
        compile_torch_model(
            torch_fc_model,
            torch_input,
            False,
            default_configuration,
            n_bits=2,
            p_error=0.01,
            rounding_threshold_bits=6,
            inputs_encryption_status=(
                "encrypted",
                "encrypted",
            ),
        )

    # No encrypted value
    with pytest.raises(ValueError, match="At least one input should be encrypted but got .*"):
        compile_torch_model(
            torch_fc_model,
            torch_input,
            False,
            default_configuration,
            n_bits=2,
            p_error=0.01,
            rounding_threshold_bits=6,
            inputs_encryption_status=("clear",),
        )

    # Correct
    compile_torch_model(
        torch_fc_model,
        torch_input,
        False,
        default_configuration,
        n_bits=2,
        p_error=0.01,
        rounding_threshold_bits=6,
        inputs_encryption_status=("encrypted",),
    )

    # Default (redundant with other test)
    compile_torch_model(
        torch_fc_model,
        torch_input,
        False,
        default_configuration,
        n_bits=2,
        p_error=0.01,
        rounding_threshold_bits=6,
        inputs_encryption_status=None,
    )


def quantized_module_predictions_are_equal(
    quantized_module_1: QuantizedModule,
    quantized_module_2: QuantizedModule,
    x: numpy.ndarray,
) -> bool:
    """Check that the predictions made by both quantized modules are identical."""

    predictions_1 = quantized_module_1.forward(x, fhe="disable")
    predictions_2 = quantized_module_2.forward(x, fhe="disable")
    return values_are_equal(predictions_1, predictions_2)


@pytest.mark.parametrize(
    "model_class, input_shape",
    [
        pytest.param(FC, (100, 32 * 32 * 3)),
        pytest.param(partial(CNN, input_output=3), (100, 3, 32, 32)),
        pytest.param(partial(CNNMaxPool, input_output=3), (100, 3, 32, 32)),
    ],
)
def test_serialization(model_class, input_shape):
    """Test the serialization of quantized modules."""

    # Define the torch model
    torch_fc_model = model_class(activation_function=nn.ReLU)
    torch_fc_model.eval()

    # Create a random input
    numpy_input = numpy.random.uniform(size=input_shape)
    torch_input = torch.from_numpy(numpy_input).float()

    # Create the corresponding numpy model
    numpy_fc_model = NumpyModule(torch_fc_model, torch_input)

    # Quantize with post-training static method
    post_training_quant = PostTrainingAffineQuantization(8, numpy_fc_model)
    quantized_module = post_training_quant.quantize_module(numpy_input)

    check_serialization(
        quantized_module,
        QuantizedModule,
        equal_method=partial(quantized_module_predictions_are_equal, x=numpy_input),
    )


def test_quantized_module_initialization_error():
    """Test initialization fails with mismatched parameters."""
    # Initialize with invalid parameters
    with pytest.raises(
        ValueError,
        match=r"Please either set all three 'ordered_module_input_names', "
        r"'ordered_module_output_names' and 'quant_layers_dict' or none of them.",
    ):
        QuantizedModule(
            ordered_module_input_names=["input1", "input2"],
            ordered_module_output_names=None,  # This makes the combination invalid
            quant_layers_dict={"layer1": (["input1"], "QuantizedOp")},
        )


@pytest.mark.parametrize("model_class, input_shape", [pytest.param(FC, (100, 32 * 32 * 3))])
def test_crt_circuit_creation_with_rounding(model_class, input_shape, default_configuration):
    """Test the creation of CRT and non-CRT circuits based on rounding settings."""

    torch_fc_model = model_class(activation_function=nn.ReLU)
    torch_fc_model.eval()

    # Create random input
    numpy_input = numpy.random.uniform(size=input_shape)
    torch_input = torch.from_numpy(numpy_input).float()

    # Compile with rounding_threshold_bits = 6
    quantized_model_with_rounding = compile_torch_model(
        torch_fc_model,
        torch_input,
        False,
        default_configuration,
        n_bits=6,
        p_error=0.01,
        rounding_threshold_bits=4,
    )

    # Check that the packing key switch count is not present
    # This should not be using CRT encoding
    assert (
        quantized_model_with_rounding.fhe_circuit.statistics.get("packing_key_switch_count", 0) == 0
    ), "Packing key switch count should be 0 when rounding_threshold_bits is set."

    # Compile with rounding_threshold_bits = None
    # Which should create a CRT bsaed encoding circuit
    quantized_model_without_rounding = compile_torch_model(
        torch_fc_model,
        torch_input,
        False,
        default_configuration,
        n_bits=4,
        rounding_threshold_bits=None,
    )

    # Check that the packing key switch count is not zero
    assert (
        quantized_model_without_rounding.fhe_circuit.statistics.get("packing_key_switch_count", 0)
        > 0
    ), "Packing key switch count should be > 0 when rounding_threshold_bits is not set."
