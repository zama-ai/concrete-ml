"""Tests for Keras models (by conversion to ONNX)."""

import numpy
import onnx
import pytest
import torch

from concrete.ml.common.utils import to_tuple
from concrete.ml.onnx.convert import OPSET_VERSION_FOR_ONNX_EXPORT
from concrete.ml.pytest.torch_models import SimpleNet
from concrete.ml.torch.compile import compile_onnx_model

INPUT_OUTPUT_FEATURE = [5, 10]


def compile_and_test_keras(
    input_output_feature,
    default_configuration,
    simulate,
    check_is_good_execution_for_cml_vs_circuit,
):
    """Test the different model architecture from Keras."""

    n_bits = 2
    input_shape = (input_output_feature,)
    n_examples = 50

    # Create random input
    inputset = numpy.random.uniform(-100, 100, size=(n_examples, *input_shape))

    # Convert to ONNX
    onnx_model = onnx.load(f"tests/data/tf_onnx/fc_{input_output_feature}.onnx")
    onnx.checker.check_model(onnx_model)

    # Compile
    quantized_numpy_module = compile_onnx_model(
        onnx_model,
        inputset,
        configuration=default_configuration,
        n_bits=n_bits,
    )

    # Create test data from the same distribution and quantize using
    # learned quantization parameters during compilation
    x_test = tuple(inputs[:1] for inputs in to_tuple(inputset))

    quantized_numpy_module.check_model_is_compiled()

    check_is_good_execution_for_cml_vs_circuit(
        x_test, model=quantized_numpy_module, simulate=simulate
    )


# We should also have some correctness tests
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2749


@pytest.mark.parametrize(
    "input_output_feature",
    [pytest.param(input_output_feature) for input_output_feature in INPUT_OUTPUT_FEATURE],
)
@pytest.mark.parametrize("simulate", [True, False])
def test_compile_keras_networks(
    input_output_feature,
    default_configuration,
    simulate,
    check_is_good_execution_for_cml_vs_circuit,
):
    """Test the different model architecture from Keras."""

    compile_and_test_keras(
        input_output_feature,
        default_configuration,
        simulate,
        check_is_good_execution_for_cml_vs_circuit,
    )


def test_failure_wrong_opset():
    """Test that wrong ONNX opset version are caught."""
    model = SimpleNet()

    with pytest.raises(AssertionError) as excinfo:
        inputset = torch.Tensor([1.0])
        torch.onnx.export(
            model, inputset, "tmp.onnx", opset_version=OPSET_VERSION_FOR_ONNX_EXPORT - 1
        )

        onnx_model = onnx.load("tmp.onnx")

        # Compile
        _ = compile_onnx_model(
            onnx_model,
            inputset,
            n_bits=8,
        )

    expected_string = (
        f"ONNX version must be {OPSET_VERSION_FOR_ONNX_EXPORT} but "
        + f"it is {OPSET_VERSION_FOR_ONNX_EXPORT - 1}"
    )
    assert expected_string == str(excinfo.value)
