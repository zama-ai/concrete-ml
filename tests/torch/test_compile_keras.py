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

    def call(self, inputs):
        """Forward function."""
        x = self.flatten(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        return x


def compile_and_test_keras(
    input_output_feature,
    model,
    default_configuration,
    use_virtual_lib,
):
    """Test the different model architecture from Keras."""

    n_bits = 2
    input_shape = (input_output_feature,)
    num_inputs = 1
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
        keras_model, opset=OPSET_VERSION_FOR_ONNX_EXPORT, output_path=str(output_onnx_file_path)
    )
    onnx.checker.check_model(onnx_model)

    # Compile
    quantized_numpy_module = compile_onnx_model(
        onnx_model,
        inputset,
        configuration=default_configuration,
        n_bits=n_bits,
        use_virtual_lib=use_virtual_lib,
    )

    # Create test data from the same distribution and quantize using
    # learned quantization parameters during compilation
    x_test = tuple(
        numpy.random.uniform(-100, 100, size=(1, *input_shape)) for _ in range(num_inputs)
    )
    qtest = quantized_numpy_module.quantize_input(*x_test)

    if not isinstance(qtest, tuple):
        qtest = (qtest,)

    assert quantized_numpy_module.is_compiled
    quantized_numpy_module.forward_fhe.encrypt_run_decrypt(*qtest)

    # FIXME: add tests of results


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
def test_compile_keras_networks(
    input_output_feature,
    model,
    default_configuration,
    use_virtual_lib,
):
    """Test the different model architecture from Keras."""
    compile_and_test_keras(
        input_output_feature,
        model,
        default_configuration,
        use_virtual_lib,
    )
