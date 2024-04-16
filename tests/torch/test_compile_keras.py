"""Tests for Keras models (by conversion to ONNX)."""

import tempfile
import warnings
from pathlib import Path

import numpy
import onnx
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    import tensorflow
    import tf2onnx

from concrete.ml.common.utils import to_tuple
from concrete.ml.onnx.convert import OPSET_VERSION_FOR_ONNX_EXPORT
from concrete.ml.torch.compile import compile_onnx_model

INPUT_OUTPUT_FEATURE = [5, 10]


class FC(tensorflow.keras.Model):
    """A fully-connected model."""

    def __init__(self):
        super().__init__()
        hidden_layer_size = 13
        output_size = 5

        self.dense1 = tensorflow.keras.layers.Dense(
            hidden_layer_size,
            activation=tensorflow.nn.relu,
        )
        self.dense2 = tensorflow.keras.layers.Dense(output_size, activation=tensorflow.nn.relu6)
        self.flatten = tensorflow.keras.layers.Flatten()

    # pylint: disable-next=unused-argument
    def call(self, inputs, training=None, mask=None):
        """Forward function."""
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


def compile_and_test_keras(
    input_output_feature,
    model,
    opset,
    default_configuration,
    simulate,
    check_is_good_execution_for_cml_vs_circuit,
):
    """Test the different model architecture from Keras."""

    n_bits = 2
    input_shape = (input_output_feature,)
    n_examples = 50

    # Define the Keras model
    keras_model = model()
    keras_model.build((None,) + input_shape)
    keras_model.compute_output_shape(input_shape=(None, input_output_feature))

    # Create random input
    inputset = numpy.random.uniform(-100, 100, size=(n_examples, *input_shape))

    # Convert to ONNX
    output_onnx_file_path = Path(tempfile.mkstemp(suffix=".onnx")[1])
    onnx_model, _ = tf2onnx.convert.from_keras(
        keras_model, opset=opset, output_path=str(output_onnx_file_path)
    )
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
    "model",
    [
        pytest.param(FC),
    ],
)
@pytest.mark.parametrize(
    "input_output_feature",
    [pytest.param(input_output_feature) for input_output_feature in INPUT_OUTPUT_FEATURE],
)
@pytest.mark.parametrize("simulate", [True, False])
def test_compile_keras_networks(
    model,
    input_output_feature,
    default_configuration,
    simulate,
    check_is_good_execution_for_cml_vs_circuit,
):
    """Test the different model architecture from Keras."""

    compile_and_test_keras(
        input_output_feature,
        model,
        OPSET_VERSION_FOR_ONNX_EXPORT,
        default_configuration,
        simulate,
        check_is_good_execution_for_cml_vs_circuit,
    )


def test_failure_wrong_offset(check_is_good_execution_for_cml_vs_circuit):
    """Test that wrong ONNX opset version are caught."""
    input_output_feature = 4
    model = FC

    with pytest.raises(AssertionError) as excinfo:
        compile_and_test_keras(
            input_output_feature,
            model,
            OPSET_VERSION_FOR_ONNX_EXPORT - 1,
            None,
            None,
            check_is_good_execution_for_cml_vs_circuit,
        )

    expected_string = (
        f"ONNX version must be {OPSET_VERSION_FOR_ONNX_EXPORT} but "
        + f"it is {OPSET_VERSION_FOR_ONNX_EXPORT - 1}"
    )
    assert expected_string == str(excinfo.value)
