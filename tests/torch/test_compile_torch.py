"""Tests for the torch to numpy module."""
# pylint: disable=too-many-lines
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

from concrete.ml.common.utils import manage_parameters_for_pbs_errors, to_tuple
from concrete.ml.onnx.convert import OPSET_VERSION_FOR_ONNX_EXPORT
from concrete.ml.pytest.torch_models import (
    FC,
    BranchingGemmModule,
    BranchingModule,
    CNNGrouped,
    CNNOther,
    DoubleQuantQATMixNet,
    FCSmall,
    MultiInputNN,
    MultiInputNNConfigurable,
    MultiInputNNDifferentSize,
    NetWithLoops,
    PaddingNet,
    ShapeOperationsNet,
    SimpleQAT,
    SingleMixNet,
    StepActivationModule,
    TinyQATCNN,
    UnivariateModule,
)
from concrete.ml.quantization import QuantizedModule

# pylint sees separated imports from concrete but does not understand they come from two different
# packages/projects, disable the warning
# pylint: disable=ungrouped-imports
from concrete.ml.torch.compile import (
    build_quantized_module,
    compile_brevitas_qat_model,
    compile_onnx_model,
    compile_torch_model,
)

# pylint: enable=ungrouped-imports


def create_test_inputset(inputset, n_percent_inputset_examples_test):
    """Create a test input-set from a given input-set and percentage of examples."""
    n_examples_test = int(n_percent_inputset_examples_test * to_tuple(inputset)[0].shape[0])
    x_test = tuple(inputs[:n_examples_test] for inputs in to_tuple(inputset))
    return x_test


def get_and_compile_quantized_module(model, inputset, import_qat, n_bits, configuration, verbose):
    """Get and compile the quantized module built from the given model."""
    quantized_numpy_module = build_quantized_module(
        model,
        inputset,
        import_qat=import_qat,
        n_bits=n_bits,
    )

    p_error, global_p_error = manage_parameters_for_pbs_errors(None, None)

    quantized_numpy_module.compile(
        inputset,
        configuration=configuration,
        p_error=p_error,
        global_p_error=global_p_error,
        verbose=verbose,
    )

    return quantized_numpy_module


# pylint: disable-next=too-many-arguments, too-many-branches
def compile_and_test_torch_or_onnx(  # pylint: disable=too-many-locals, too-many-statements
    input_output_feature,
    model_class,
    activation_function,
    qat_bits,
    default_configuration,
    simulate,
    is_onnx,
    check_is_good_execution_for_cml_vs_circuit,
    dump_onnx=False,
    expected_onnx_str=None,
    verbose=False,
    get_and_compile=False,
    input_shape=None,
    is_brevitas_qat=False,
) -> QuantizedModule:
    """Test the different model architecture from torch numpy."""

    # Define an input shape (n_examples, n_features)
    n_examples = 500

    # Define the torch model
    torch_model = model_class(
        input_output=input_output_feature, activation_function=activation_function
    )

    num_inputs = len(signature(torch_model.forward).parameters)

    # If no specific input shape is given, use the number of input/output features
    if input_shape is None:
        input_shape = input_output_feature

    # Create random input
    if num_inputs > 1:
        inputset = tuple(
            numpy.random.uniform(-100, 100, size=(n_examples, *to_tuple(input_shape[i])))
            for i in range(num_inputs)
        )
    else:
        inputset = (numpy.random.uniform(-100, 100, size=(n_examples, *to_tuple(input_shape))),)

    # FHE vs Quantized are not done in the test anymore (see issue #177)
    if not simulate:

        n_bits = (
            {"model_inputs": 2, "model_outputs": 2, "op_inputs": 2, "op_weights": 2}
            if qat_bits == 0
            else qat_bits
        )

        if is_onnx:
            output_onnx_file_path = Path(tempfile.mkstemp(suffix=".onnx")[1])

            dummy_input = tuple(torch.from_numpy(val[[0], ::]).float() for val in inputset)
            torch.onnx.export(
                torch_model,
                dummy_input,
                str(output_onnx_file_path),
                opset_version=OPSET_VERSION_FOR_ONNX_EXPORT,
            )
            onnx_model = onnx.load_model(str(output_onnx_file_path))
            onnx.checker.check_model(onnx_model)

            if get_and_compile:
                quantized_numpy_module = get_and_compile_quantized_module(
                    model=onnx_model,
                    inputset=inputset,
                    import_qat=qat_bits != 0,
                    n_bits=n_bits,
                    configuration=default_configuration,
                    verbose=verbose,
                )

            else:
                quantized_numpy_module = compile_onnx_model(
                    onnx_model,
                    inputset,
                    import_qat=qat_bits != 0,
                    configuration=default_configuration,
                    n_bits=n_bits,
                    verbose=verbose,
                )
        else:
            if is_brevitas_qat:
                n_bits = qat_bits

                quantized_numpy_module = compile_brevitas_qat_model(
                    torch_model=torch_model,
                    torch_inputset=inputset,
                    n_bits=n_bits,
                    configuration=default_configuration,
                    verbose=verbose,
                )

            elif get_and_compile:
                quantized_numpy_module = get_and_compile_quantized_module(
                    model=torch_model,
                    inputset=inputset,
                    import_qat=qat_bits != 0,
                    n_bits=n_bits,
                    configuration=default_configuration,
                    verbose=verbose,
                )

            else:
                quantized_numpy_module = compile_torch_model(
                    torch_model,
                    inputset,
                    import_qat=qat_bits != 0,
                    configuration=default_configuration,
                    n_bits=n_bits,
                    verbose=verbose,
                )

        n_examples_test = 1
        # Use some input-set to test the inference.
        # Using the input-set allows to remove any chance of overflow.
        x_test = tuple(inputs[:n_examples_test] for inputs in inputset)

        quantized_numpy_module.check_model_is_compiled()

        # Make sure FHE simulation and quantized module forward give the same output.
        check_is_good_execution_for_cml_vs_circuit(
            x_test, model=quantized_numpy_module, simulate=simulate
        )
    else:
        if is_brevitas_qat:
            n_bits = qat_bits

            quantized_numpy_module = compile_brevitas_qat_model(
                torch_model=torch_model,
                torch_inputset=inputset,
                n_bits=n_bits,
                configuration=default_configuration,
                verbose=verbose,
            )

        else:
            # Compile our network with 16-bits
            # to compare to torch (8b weights + float 32 activations)
            if qat_bits == 0:
                n_bits_w_a = 4
            else:
                n_bits_w_a = qat_bits

            n_bits = {
                "model_inputs": 8,
                "op_weights": n_bits_w_a,
                "op_inputs": n_bits_w_a,
                "model_outputs": 8,
            }

            if get_and_compile:
                quantized_numpy_module = get_and_compile_quantized_module(
                    model=torch_model,
                    inputset=inputset,
                    import_qat=qat_bits != 0,
                    n_bits=n_bits,
                    configuration=default_configuration,
                    verbose=verbose,
                )

            else:
                quantized_numpy_module = compile_torch_model(
                    torch_model,
                    inputset,
                    import_qat=qat_bits != 0,
                    configuration=default_configuration,
                    n_bits=n_bits,
                    verbose=verbose,
                )

        accuracy_test_rounding(
            torch_model,
            quantized_numpy_module,
            inputset,
            import_qat=qat_bits != 0,
            configuration=default_configuration,
            n_bits=n_bits,
            simulate=simulate,
            verbose=verbose,
            check_is_good_execution_for_cml_vs_circuit=check_is_good_execution_for_cml_vs_circuit,
            is_brevitas_qat=is_brevitas_qat,
        )

        if dump_onnx:
            str_model = onnx.helper.printable_graph(quantized_numpy_module.onnx_model.graph)
            print("ONNX model:")
            print(str_model)
            assert str_model == expected_onnx_str

    return quantized_numpy_module


# pylint: disable-next=too-many-arguments,too-many-locals
def accuracy_test_rounding(
    torch_model,
    quantized_numpy_module,
    inputset,
    import_qat,
    configuration,
    n_bits,
    simulate,
    verbose,
    check_is_good_execution_for_cml_vs_circuit,
    is_brevitas_qat=False,
):
    """Check rounding behavior.

    The original quantized_numpy_module, compiled over the torch_model without rounding is
    compared against quantized_numpy_module_round_low_precision, the torch_model compiled with
    a rounding threshold of 2 bits, and quantized_numpy_module_round_high_precision,
    the torch_model compiled with maximum bit-width computed in the quantized_numpy_module - 1.

    The final assertion tests whether the mean absolute error between
    quantized_numpy_module_round_high_precision and quantized_numpy_module is lower than
    quantized_numpy_module_round_low_precision and quantized_numpy_module making sure that the
    rounding feature has the expected behavior on the model accuracy.
    """
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2888
    assert simulate, "Rounding is not available in FHE yet."

    # Check that the maximum_integer_bit_width is at least 4 bits to compare the rounding
    # feature with enough precision.
    assert quantized_numpy_module.fhe_circuit.graph.maximum_integer_bit_width() >= 4

    # Compile with a rounding threshold equal to the maximum bit-width
    # computed in the original quantized_numpy_module
    if is_brevitas_qat:
        quantized_numpy_module_round_high_precision = compile_brevitas_qat_model(
            torch_model,
            inputset,
            n_bits=n_bits,
            configuration=configuration,
            rounding_threshold_bits=(
                quantized_numpy_module.fhe_circuit.graph.maximum_integer_bit_width() - 1
            ),
            verbose=verbose,
        )

        # and another quantized module with a rounding threshold equal to 2 bits
        quantized_numpy_module_round_low_precision = compile_brevitas_qat_model(
            torch_model,
            inputset,
            n_bits=n_bits,
            configuration=configuration,
            rounding_threshold_bits=2,
            verbose=verbose,
        )

    else:
        quantized_numpy_module_round_high_precision = compile_torch_model(
            torch_model,
            inputset,
            import_qat=import_qat,
            configuration=configuration,
            n_bits=n_bits,
            rounding_threshold_bits=(
                quantized_numpy_module.fhe_circuit.graph.maximum_integer_bit_width() - 1
            ),
            verbose=verbose,
        )

        # and another quantized module with a rounding threshold equal to 2 bits
        quantized_numpy_module_round_low_precision = compile_torch_model(
            torch_model,
            inputset,
            import_qat=import_qat,
            configuration=configuration,
            n_bits=n_bits,
            verbose=verbose,
            rounding_threshold_bits=2,
        )

    n_percent_inputset_examples_test = 0.1
    # Using the input-set allows to remove any chance of overflow.
    x_test = create_test_inputset(inputset, n_percent_inputset_examples_test)

    # Make sure the two modules have the same quantization result
    qtest = to_tuple(quantized_numpy_module.quantize_input(*x_test))
    qtest_high = to_tuple(quantized_numpy_module_round_high_precision.quantize_input(*x_test))
    qtest_low = to_tuple(quantized_numpy_module_round_low_precision.quantize_input(*x_test))

    assert all(
        numpy.array_equal(qtest_i, qtest_high_i)
        for (qtest_i, qtest_high_i) in zip(qtest, qtest_high)
    )
    assert all(
        numpy.array_equal(qtest_i, qtest_low_i) for (qtest_i, qtest_low_i) in zip(qtest, qtest_low)
    )

    results = []
    results_high_precision = []
    results_low_precision = []
    for i in range(x_test[0].shape[0]):

        # Extract example i for each tensor in the test tuple with quantized values while
        # keeping the dimension of the original tensors (e.g., if it is a tuple of two (100, 10)
        # tensors, then each quantized value becomes a tuple of two tensors of shape (1, 10).
        q_x = tuple(q[[i]] for q in to_tuple(qtest))

        # encrypt, run, and decrypt with different precision modes
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2888
        q_result = quantized_numpy_module.quantized_forward(*q_x, fhe="simulate")
        q_result_high_precision = quantized_numpy_module_round_high_precision.quantized_forward(
            *q_x,
            fhe="simulate",
        )
        q_result_low_precision = quantized_numpy_module_round_low_precision.quantized_forward(
            *q_x,
            fhe="simulate",
        )

        # de-quantize the results to obtain the actual output values
        result = quantized_numpy_module.dequantize_output(q_result)
        result_high_precision = quantized_numpy_module_round_high_precision.dequantize_output(
            q_result_high_precision
        )
        result_low_precision = quantized_numpy_module_round_low_precision.dequantize_output(
            q_result_low_precision
        )

        # append the results to respective lists
        results.append(result)
        results_high_precision.append(result_high_precision)
        results_low_precision.append(result_low_precision)

    # Check modules predictions FHE simulation vs Concrete ML.
    check_is_good_execution_for_cml_vs_circuit(x_test, quantized_numpy_module, simulate=simulate)
    check_is_good_execution_for_cml_vs_circuit(
        x_test, quantized_numpy_module_round_high_precision, simulate=simulate
    )
    check_is_good_execution_for_cml_vs_circuit(
        x_test, quantized_numpy_module_round_low_precision, simulate=simulate
    )

    # Check that high precision gives a better match than low precision
    # MSE is preferred over MAE here to spot a lack of diversity in the 2 bits rounded model
    # e.g., results_low_precision = mean(results) should impact more MSE than MAE.
    # mse_high_precision = numpy.mean(numpy.square(numpy.subtract(results, results_high_precision)))
    # mse_low_precision = numpy.mean(numpy.square(numpy.subtract(results, results_low_precision)))

    # This assert is too unstable and creates more and more flaky tests, we will investigate a
    # better way to assess the rounding feature's performance
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3662
    # assert mse_high_precision <= mse_low_precision, "Rounding is not working as expected."


@pytest.mark.parametrize(
    "activation_function",
    [
        pytest.param(nn.ReLU, id="relu"),
    ],
)
@pytest.mark.parametrize(
    "model, input_output_feature",
    [
        pytest.param(FCSmall, 5),
        pytest.param(partial(NetWithLoops, n_fc_layers=2), 5),
        pytest.param(BranchingModule, 5),
        pytest.param(BranchingGemmModule, 5),
        pytest.param(MultiInputNN, [5, 5]),
        pytest.param(MultiInputNNDifferentSize, [5, 10]),
        pytest.param(UnivariateModule, 5),
        pytest.param(StepActivationModule, 5),
    ],
)
@pytest.mark.parametrize("simulate", [True, False], ids=["FHE_simulation", "FHE"])
@pytest.mark.parametrize("is_onnx", [True, False], ids=["is_onnx", ""])
@pytest.mark.parametrize("get_and_compile", [True, False], ids=["get_and_compile", "compile"])
def test_compile_torch_or_onnx_networks(
    input_output_feature,
    model,
    activation_function,
    default_configuration,
    simulate,
    is_onnx,
    get_and_compile,
    check_is_good_execution_for_cml_vs_circuit,
    is_weekly_option,
):
    """Test the different model architecture from torch numpy."""

    # Avoid too many tests
    if not simulate and not is_weekly_option:
        if model not in [FCSmall, BranchingModule]:
            pytest.skip("Avoid too many tests")

    # The QAT bits is set to 0 in order to signal that the network is not using QAT
    qat_bits = 0

    compile_and_test_torch_or_onnx(
        input_output_feature=input_output_feature,
        model_class=model,
        activation_function=activation_function,
        qat_bits=qat_bits,
        default_configuration=default_configuration,
        simulate=simulate,
        is_onnx=is_onnx,
        check_is_good_execution_for_cml_vs_circuit=check_is_good_execution_for_cml_vs_circuit,
        verbose=False,
        get_and_compile=get_and_compile,
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
        pytest.param(CNNOther),
        pytest.param(partial(CNNGrouped, groups=3)),
    ],
)
@pytest.mark.parametrize("simulate", [True, False])
@pytest.mark.parametrize("is_onnx", [True, False])
def test_compile_torch_or_onnx_conv_networks(  # pylint: disable=unused-argument
    model,
    activation_function,
    default_configuration,
    simulate,
    is_onnx,
    check_graph_input_has_no_tlu,
    check_graph_output_has_no_tlu,
    check_is_good_execution_for_cml_vs_circuit,
):
    """Test the different model architecture from torch numpy."""

    # The QAT bits is set to 0 in order to signal that the network is not using QAT
    qat_bits = 0

    input_shape = (6, 7, 7)
    input_output = input_shape[0]

    q_module = compile_and_test_torch_or_onnx(
        input_output_feature=input_output,
        model_class=model,
        activation_function=activation_function,
        qat_bits=qat_bits,
        default_configuration=default_configuration,
        simulate=simulate,
        is_onnx=is_onnx,
        check_is_good_execution_for_cml_vs_circuit=check_is_good_execution_for_cml_vs_circuit,
        verbose=False,
        input_shape=input_shape,
    )

    check_graph_input_has_no_tlu(q_module.fhe_circuit.graph)
    check_graph_output_has_no_tlu(q_module.fhe_circuit.graph)


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
@pytest.mark.parametrize(
    "model, input_output_feature",
    [
        pytest.param(FCSmall, 5),
    ],
)
@pytest.mark.parametrize("simulate", [True, False])
@pytest.mark.parametrize("is_onnx", [True, False])
def test_compile_torch_or_onnx_activations(
    input_output_feature,
    model,
    activation_function,
    default_configuration,
    simulate,
    is_onnx,
    check_is_good_execution_for_cml_vs_circuit,
):
    """Test the different model architecture from torch numpy."""

    # The QAT bits is set to 0 in order to signal that the network is not using QAT
    qat_bits = 0

    compile_and_test_torch_or_onnx(
        input_output_feature,
        model,
        activation_function,
        qat_bits,
        default_configuration,
        simulate,
        is_onnx,
        check_is_good_execution_for_cml_vs_circuit,
        verbose=False,
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
@pytest.mark.parametrize("simulate", [True, False])
def test_compile_torch_qat(
    input_output_feature,
    model,
    n_bits,
    default_configuration,
    simulate,
    check_is_good_execution_for_cml_vs_circuit,
):
    """Test the different model architecture from torch numpy."""

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
        simulate,
        is_onnx,
        check_is_good_execution_for_cml_vs_circuit,
        verbose=False,
    )


@pytest.mark.parametrize(
    "model_class, input_output_feature, is_brevitas_qat",
    [pytest.param(partial(MultiInputNNDifferentSize, is_brevitas_qat=True), [5, 10], True)],
)
@pytest.mark.parametrize(
    "n_bits",
    [pytest.param(n_bits) for n_bits in [2]],
)
@pytest.mark.parametrize("simulate", [True, False])
def test_compile_brevitas_qat(
    model_class,
    input_output_feature,
    is_brevitas_qat,
    n_bits,
    simulate,
    default_configuration,
    check_is_good_execution_for_cml_vs_circuit,
):
    """Test compile_brevitas_qat_model."""

    model_class = partial(model_class, n_bits=n_bits)

    # If this is a Brevitas QAT model, use n_bits for QAT bits
    if is_brevitas_qat:
        qat_bits = n_bits

    # The QAT bits is set to 0 in order to signal that the network is not using QAT
    else:
        qat_bits = 0

    compile_and_test_torch_or_onnx(
        input_output_feature=input_output_feature,
        model_class=model_class,
        activation_function=None,
        qat_bits=qat_bits,
        default_configuration=default_configuration,
        simulate=simulate,
        is_onnx=False,
        check_is_good_execution_for_cml_vs_circuit=check_is_good_execution_for_cml_vs_circuit,
        verbose=False,
        is_brevitas_qat=is_brevitas_qat,
    )


@pytest.mark.parametrize(
    "model_class, expected_onnx_str",
    [
        pytest.param(
            FC,
            (
                """graph torch_jit (
  %onnx::Gemm_0[FLOAT, 1x7]
) initializers (
  %fc1.weight[FLOAT, 128x7]
  %fc1.bias[FLOAT, 128]
  %fc2.weight[FLOAT, 64x128]
  %fc2.bias[FLOAT, 64]
  %fc3.weight[FLOAT, 64x64]
  %fc3.bias[FLOAT, 64]
  %fc4.weight[FLOAT, 64x64]
  %fc4.bias[FLOAT, 64]
  %fc5.weight[FLOAT, 10x64]
  %fc5.bias[FLOAT, 10]
) {
  %/fc1/Gemm_output_0 = Gemm[alpha = 1, beta = 1, transB = 1]"""
                """(%onnx::Gemm_0, %fc1.weight, %fc1.bias)
  %/act_1/Relu_output_0 = Relu(%/fc1/Gemm_output_0)
  %/fc2/Gemm_output_0 = Gemm[alpha = 1, beta = 1, transB = 1]"""
                """(%/act_1/Relu_output_0, %fc2.weight, %fc2.bias)
  %/act_2/Relu_output_0 = Relu(%/fc2/Gemm_output_0)
  %/fc3/Gemm_output_0 = Gemm[alpha = 1, beta = 1, transB = 1]"""
                """(%/act_2/Relu_output_0, %fc3.weight, %fc3.bias)
  %/act_3/Relu_output_0 = Relu(%/fc3/Gemm_output_0)
  %/fc4/Gemm_output_0 = Gemm[alpha = 1, beta = 1, transB = 1]"""
                """(%/act_3/Relu_output_0, %fc4.weight, %fc4.bias)
  %/act_4/Relu_output_0 = Relu(%/fc4/Gemm_output_0)
  %19 = Gemm[alpha = 1, beta = 1, transB = 1](%/act_4/Relu_output_0, %fc5.weight, %fc5.bias)
  return %19
}"""
            ),
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
    model_class,
    expected_onnx_str,
    activation_function,
    default_configuration,
    check_is_good_execution_for_cml_vs_circuit,
):
    """This is a test which is equivalent to tests in test_dump_onnx.py, but for torch modules."""
    input_output_feature = 7
    simulate = True
    is_onnx = False
    qat_bits = 0

    compile_and_test_torch_or_onnx(
        input_output_feature,
        model_class,
        activation_function,
        qat_bits,
        default_configuration,
        simulate,
        is_onnx,
        check_is_good_execution_for_cml_vs_circuit,
        dump_onnx=True,
        expected_onnx_str=expected_onnx_str,
        verbose=False,
    )


@pytest.mark.parametrize("verbose", [True, False], ids=["with_verbose", "without_verbose"])
# pylint: disable-next=too-many-locals
def test_pretrained_mnist_qat(
    default_configuration,
    check_accuracy,
    verbose,
    check_graph_output_has_no_tlu,
    check_is_good_execution_for_cml_vs_circuit,
    is_weekly_option,
):
    """Load a QAT MNIST model and confirm we get the same results in FHE simulation as with ONNX."""
    if not is_weekly_option:
        pytest.skip("Tests too long")

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

    # Compile to Concrete ML in FHE simulation mode, with a high bit-width
    n_bits = {
        "model_inputs": 16,
        "op_weights": 2,
        "op_inputs": 2,
        "model_outputs": 16,
    }

    quantized_numpy_module = compile_onnx_model(
        onnx_model,
        inputset,
        import_qat=True,
        configuration=default_configuration,
        n_bits=n_bits,
        verbose=verbose,
    )

    quantized_numpy_module.check_model_is_compiled()

    check_is_good_execution_for_cml_vs_circuit(inputset, quantized_numpy_module, simulate=True)

    # Collect FHE simulation results
    results = []
    for i in range(inputset.shape[0]):

        # Extract example i for each tensor in the tuple input-set
        # while keeping the dimension of the original tensors.
        # e.g., if input-set is a tuple of two (100, 10) tensors
        # then q_x becomes a tuple of two tensors of shape (1, 10).
        x = tuple(input[[i]] for input in to_tuple(inputset))
        result = numpy.argmax(quantized_numpy_module.forward(*x, fhe="simulate"))
        results.append(result)

    # Compare ONNX runtime vs FHE simulation mode
    check_accuracy(onnx_results, results, threshold=0.999)

    # Make sure absolute accuracy is good, this model should have at least 90% accuracy
    check_accuracy(mnist_data["gt"], results, threshold=0.9)

    # Compile to Concrete ML using the FHE simulation mode and compatible bit-width
    n_bits = {
        "model_inputs": 7,
        "op_weights": 2,
        "op_inputs": 2,
        "model_outputs": 7,
    }

    quantized_numpy_module = compile_onnx_model(
        onnx_model,
        inputset,
        import_qat=True,
        configuration=default_configuration,
        n_bits=n_bits,
        verbose=verbose,
    )

    # As this is a custom QAT network, the input goes through multiple univariate
    # ops that form a quantizer. Thus it has input TLUs. But it should not have output TLUs
    check_graph_output_has_no_tlu(quantized_numpy_module.fhe_circuit.graph)

    assert quantized_numpy_module.fhe_circuit.graph.maximum_integer_bit_width() <= 8


def test_qat_import_bits_check(default_configuration):
    """Test that compile_brevitas_qat_model does not need an n_bits config."""

    input_features = 10

    model = SingleMixNet(False, True, 10, 2)

    n_examples = 50

    # All these n_bits configurations should be valid
    # and produce the same result, as the input/output bit-widths for this network
    # are ignored due to the input/output TLU elimination
    n_bits_valid = [
        4,
        2,
        {"model_inputs": 4, "model_outputs": 4},
        {"model_inputs": 2, "model_outputs": 2},
    ]

    # Create random input
    inputset = numpy.random.uniform(-100, 100, size=(n_examples, input_features))

    # Compile with no quantization bit-width, defaults are used
    quantized_numpy_module = compile_brevitas_qat_model(
        model,
        inputset,
        configuration=default_configuration,
    )

    n_percent_inputset_examples_test = 0.1
    # Using the input-set allows to remove any chance of overflow.
    x_test = create_test_inputset(inputset, n_percent_inputset_examples_test)

    # The result of compiling without any n_bits (default)
    predictions = quantized_numpy_module.forward(*x_test, fhe="disable")

    # Compare the results of running with n_bits=None to the results running with
    # all the other n_bits configs. The results should be the same as bit-widths
    # are ignored for this network (they are overridden with Brevitas values stored in ONNX).
    for n_bits in n_bits_valid:
        quantized_numpy_module = compile_brevitas_qat_model(
            model,
            inputset,
            n_bits=n_bits,
            configuration=default_configuration,
        )

        new_predictions = quantized_numpy_module.forward(*x_test, fhe="disable")

        assert numpy.all(predictions == new_predictions)

    n_bits_invalid = [
        {"XYZ": 8, "model_inputs": 8},
        {"XYZ": 8},
    ]

    # Test that giving a dictionary with invalid keys does not work
    for n_bits in n_bits_invalid:
        with pytest.raises(
            AssertionError, match=".*n_bits should only contain the following keys.*"
        ):
            quantized_numpy_module = compile_brevitas_qat_model(
                model,
                inputset,
                n_bits=n_bits,
                configuration=default_configuration,
            )


def test_qat_import_check(default_configuration, check_is_good_execution_for_cml_vs_circuit):
    """Test two cases of custom (non brevitas) NNs where importing as QAT networks should fail."""
    qat_bits = 4

    simulate = True

    error_message_pattern = "Error occurred during quantization aware training.*"

    # This first test is trying to import a network that is QAT (has a quantizer in the graph)
    # but the import bit-width is wrong (mismatch between bit-width specified in training
    # and the bit-width specified during import). For NNs that are not built with Brevitas
    # the bit-width must be manually specified and is used to infer quantization parameters.
    with pytest.raises(ValueError, match=error_message_pattern):
        compile_and_test_torch_or_onnx(
            10,
            partial(SimpleQAT, n_bits=6, disable_bit_check=True),
            nn.ReLU,
            qat_bits,
            default_configuration,
            simulate,
            False,
            check_is_good_execution_for_cml_vs_circuit,
        )

    input_shape = (1, 7, 7)
    input_output = input_shape[0]

    # The second case is a network that is not QAT but is being imported as a QAT network
    with pytest.raises(ValueError, match=error_message_pattern):
        compile_and_test_torch_or_onnx(
            input_output,
            CNNOther,
            nn.ReLU,
            qat_bits,
            default_configuration,
            simulate,
            False,
            check_is_good_execution_for_cml_vs_circuit,
            input_shape=input_shape,
        )

    class AllZeroCNN(CNNOther):
        """A CNN class that has all zero weights and biases."""

        def __init__(self, input_output, activation_function):
            super().__init__(input_output, activation_function)

            for module in self.modules():
                # assert m.bias is not None
                # Disable mypy as it properly detects that module's bias term is None end therefore
                # does not have a `data` attribute but fails to take into consideration the fact
                # that `torch.nn.init.constant_` actually handles such a case
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    torch.nn.init.constant_(module.weight.data, 0)
                    torch.nn.init.constant_(module.bias.data, 0)  # type: ignore[union-attr]

    input_shape = (1, 7, 7)
    input_output = input_shape[0]

    # A network that may look like QAT but it just zeros all inputs
    with pytest.raises(ValueError, match=error_message_pattern):
        compile_and_test_torch_or_onnx(
            input_output,
            AllZeroCNN,
            nn.ReLU,
            qat_bits,
            default_configuration,
            simulate,
            False,
            check_is_good_execution_for_cml_vs_circuit,
            input_shape=input_shape,
        )


@pytest.mark.parametrize("n_bits", [2])
@pytest.mark.parametrize("use_qat", [True, False])
@pytest.mark.parametrize("force_tlu", [True, False])
@pytest.mark.parametrize(
    "module, input_shape, num_inputs, is_fully_leveled",
    [
        (SingleMixNet, (1, 8, 8), 1, True),
        (SingleMixNet, 10, 1, True),
        (MultiInputNNConfigurable, 10, 2, False),
        (MultiInputNNConfigurable, (1, 8, 8), 2, False),
        (DoubleQuantQATMixNet, (1, 8, 8), 1, False),
        (DoubleQuantQATMixNet, 10, 1, False),
    ],
)
def test_net_has_no_tlu(
    module,
    input_shape,
    num_inputs,
    is_fully_leveled,
    use_qat,
    force_tlu,
    n_bits,
    default_configuration,
    check_graph_output_has_no_tlu,
):
    """Tests that there is no TLU in nets with a single conv/linear."""

    # Skip the test if the model is MultiInputNNConfigurable and use_qat is True as the module is
    # not QAT (it has no Brevitas layer)
    if num_inputs > 1 and use_qat:
        return

    use_conv = isinstance(input_shape, tuple) and len(input_shape) > 1

    net = module(use_conv, use_qat, input_shape, n_bits)
    net.eval()

    if not is_fully_leveled:
        # No need to force the presence of a TLU if there are TLUs in the body of the
        # network
        force_tlu = False

    if module is DoubleQuantQATMixNet:
        use_qat = True

    # We have the option to force having a TLU in the net by
    # applying a nonlinear function on the original network's output. Thus
    # we can check that a tlu is indeed present and was not removed by accident in this case
    if force_tlu:

        def relu_adder_decorator(method):
            def decorate_name(self):
                return torch.relu(method(self))

            return decorate_name

        net.forward = relu_adder_decorator(net.forward)

    # Generate the input in both the 2d and 1d cases
    input_shape = to_tuple(input_shape)

    # Create random input
    inputset = tuple(
        numpy.random.uniform(-100, 100, size=(100, *input_shape)) for _ in range(num_inputs)
    )

    if use_qat:
        # Compile with appropriate QAT compilation function, here the zero-points will all be 0
        quantized_numpy_module = compile_brevitas_qat_model(
            net,
            inputset,
            configuration=default_configuration,
        )
    else:
        # Compile with PTQ. Note that this will have zero-point>0
        quantized_numpy_module = compile_torch_model(
            net,
            inputset,
            import_qat=False,
            configuration=default_configuration,
            n_bits=n_bits,
        )

    mlir = quantized_numpy_module.fhe_circuit.mlir

    # Check if a TLU is present or not, depending on whether we force a TLU to be present
    if force_tlu:
        with pytest.raises(AssertionError):
            check_graph_output_has_no_tlu(quantized_numpy_module.fhe_circuit.graph)
        if is_fully_leveled:
            with pytest.raises(AssertionError):
                assert "lookup_table" not in mlir
    else:
        check_graph_output_has_no_tlu(quantized_numpy_module.fhe_circuit.graph)
        if is_fully_leveled:
            assert "lookup_table" not in mlir


@pytest.mark.parametrize("simulate", [True, False])
@pytest.mark.parametrize("is_qat", [True, False])
@pytest.mark.parametrize("n_channels", [2])
def test_shape_operations_net(
    simulate, n_channels, is_qat, default_configuration, check_graph_output_has_no_tlu
):
    """Test a pattern of reshaping, concatenation, chunk extraction."""
    net = ShapeOperationsNet(is_qat)
    inputset = numpy.random.uniform(size=(1, n_channels, 2, 2))

    if is_qat:
        quantized_module = compile_brevitas_qat_model(
            net,
            inputset,
            configuration=default_configuration,
            p_error=0.01,
        )
    else:
        quantized_module = compile_torch_model(
            net,
            inputset,
            configuration=default_configuration,
            n_bits=3,
            p_error=0.01,
        )

    # In QAT quantization options are consistent across all the layers
    # which allows for the elimination of TLUs
    # In PTQ there are TLUs in the graph because Shape/Concat/Transpose
    # must quantize inputs with some default quantization options

    # In QAT testing in FHE is fast since there are no TLUs
    # For PTQ we only test that the model can be compiled and that it can be executed
    if is_qat or simulate:
        fhe_mode = "simulate" if simulate else "execute"

        predictions = quantized_module.forward(inputset, fhe=fhe_mode)

        torch_output = net(torch.tensor(inputset)).detach().numpy()

        # In PTQ the results do not match because of a-priori set quantization options
        # Currently no solution for concat/reshape/transpose correctness in PTQ is proposed.
        if is_qat:
            assert numpy.allclose(torch_output, predictions, atol=0.05, rtol=0)

            # In QAT, since the quantization is defined a-priori, all TLUs will be removed
            # and the input quantizer is moved to the clear. We can thus check there are no TLUs
            # in the graph
            check_graph_output_has_no_tlu(quantized_module.fhe_circuit.graph)
            assert "lookup_table" not in quantized_module.fhe_circuit.mlir


def test_torch_padding(default_configuration, check_circuit_has_no_tlu):
    """Test padding in PyTorch using ONNX pad operators."""
    net = PaddingNet()

    num_batch = 20
    inputset = numpy.random.uniform(size=(num_batch, 1, 2, 2))

    quant_model = compile_brevitas_qat_model(
        net,
        inputset,
        configuration=default_configuration,
        p_error=0.01,
    )

    test_input = numpy.random.uniform(size=(1, 1, 2, 2))

    torch_output = net(torch.tensor(test_input)).detach().numpy()

    cml_output = quant_model.forward(test_input, fhe="disable")

    # We only care about checking that zeros added with padding are in the same positions
    # between the torch output and the Concrete ML output
    torch_output = torch_output > 0
    cml_output = cml_output > 0

    assert numpy.alltrue(torch_output == cml_output)
    check_circuit_has_no_tlu(quant_model.fhe_circuit)


def test_compilation_functions_check_model_types(default_configuration):
    """Check that the compile functions validate the input model types."""

    input_output_feature = 5
    n_examples = 50

    torch_model = FCSmall(input_output_feature, nn.ReLU)

    # Create random input
    inputset = numpy.random.uniform(-100, 100, size=(n_examples, input_output_feature))

    with pytest.raises(
        AssertionError,
        match=".*no Brevitas quantized layers, consider using compile_torch_model instead.*",
    ):
        compile_brevitas_qat_model(
            torch_model,
            inputset,
            configuration=default_configuration,
        )

    torch_model_qat = TinyQATCNN(5, 4, 10, True, False)
    with pytest.raises(
        AssertionError, match=".*must be imported using compile_brevitas_qat_model.*"
    ):
        compile_torch_model(
            torch_model_qat,
            inputset,
            configuration=default_configuration,
        )
