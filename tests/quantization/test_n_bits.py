"""Tests n_bits values when initializing a ONNXConverter object."""

import pytest
import torch
from torch import nn

from concrete.ml.pytest.torch_models import FCSmall
from concrete.ml.quantization.base_quantized_op import DEFAULT_MODEL_BITS
from concrete.ml.quantization.post_training import ONNXConverter
from concrete.ml.torch.numpy_module import NumpyModule

# Instantiate a simple NumpyModule used for initializing the ONNXConverter. The actual model is not
# tested here.
numpy_module = NumpyModule(
    model=FCSmall(input_output=3, activation_function=nn.ReLU),
    dummy_input=torch.randn(3),
)

n_bits_correct = [
    pytest.param(
        8,
        {
            "model_inputs": 8,
            "model_outputs": 8,
            "op_inputs": 8,
            "op_weights": 8,
        },
        id="n_bits_8_bits",
    ),
    pytest.param(
        3,
        {
            "model_inputs": DEFAULT_MODEL_BITS,
            "model_outputs": DEFAULT_MODEL_BITS,
            "op_inputs": 3,
            "op_weights": 3,
        },
        id="n_bits_3_bits",
    ),
    pytest.param(
        {
            "op_inputs": 2,
            "op_weights": 3,
        },
        {
            "model_inputs": DEFAULT_MODEL_BITS,
            "model_outputs": DEFAULT_MODEL_BITS,
            "op_inputs": 2,
            "op_weights": 3,
        },
        id="n_bits_dict_only_op_under_5_bits",
    ),
    pytest.param(
        {
            "op_inputs": 7,
            "op_weights": 8,
        },
        {
            "model_inputs": DEFAULT_MODEL_BITS,
            "model_outputs": 7,
            "op_inputs": 7,
            "op_weights": 8,
        },
        id="n_bits_dict_only_op_over_5_bits",
    ),
    pytest.param(
        {
            "model_inputs": 2,
            "op_inputs": 3,
            "op_weights": 4,
        },
        {
            "model_inputs": 2,
            "model_outputs": DEFAULT_MODEL_BITS,
            "op_inputs": 3,
            "op_weights": 4,
        },
        id="n_bits_dict_no_model_outputs",
    ),
    pytest.param(
        {
            "model_outputs": 7,
            "op_inputs": 6,
            "op_weights": 8,
        },
        {
            "model_inputs": DEFAULT_MODEL_BITS,
            "model_outputs": 7,
            "op_inputs": 6,
            "op_weights": 8,
        },
        id="n_bits_dict_no_model_inputs",
    ),
    pytest.param(
        {
            "model_inputs": 7,
            "model_outputs": 6,
            "op_inputs": 5,
            "op_weights": 8,
        },
        {
            "model_inputs": 7,
            "model_outputs": 6,
            "op_inputs": 5,
            "op_weights": 8,
        },
        id="n_bits_dict_all_over_5_bits",
    ),
    pytest.param(
        {
            "model_inputs": 1,
            "model_outputs": 3,
            "op_inputs": 2,
            "op_weights": 4,
        },
        {
            "model_inputs": 1,
            "model_outputs": 3,
            "op_inputs": 2,
            "op_weights": 4,
        },
        id="n_bits_dict_all_under_5_bits",
    ),
]


@pytest.mark.parametrize("n_bits_test, n_bits_true", n_bits_correct)
def test_n_bits(n_bits_test, n_bits_true):
    """Tests different possible values for n_bits when initializing a ONNXConverter object."""
    onnx_converter = ONNXConverter(n_bits=n_bits_test, numpy_model=numpy_module)

    assert (
        onnx_converter.n_bits_model_outputs == n_bits_true["model_outputs"]
        and onnx_converter.n_bits_model_inputs == n_bits_true["model_inputs"]
        and onnx_converter.n_bits_op_inputs == n_bits_true["op_inputs"]
        and onnx_converter.n_bits_op_weights == n_bits_true["op_weights"]
    )


n_bits_only_op = [
    pytest.param(
        {},
        id="n_bits_empty_dict",
    ),
    pytest.param(
        {
            "op_inputs": 8,
        },
        id="n_bits_only_op_inputs",
    ),
    pytest.param(
        {
            "op_weights": 8,
        },
        id="n_bits_only_op_weights",
    ),
    pytest.param(
        {
            "net": 8,
            "model_outputs": 8,
            "op_inputs": 8,
            "op_weights": 8,
        },
        id="n_bits_wrong_key",
    ),
]


@pytest.mark.parametrize("n_bits", n_bits_only_op)
def test_n_bits_only_op(n_bits):
    """Tests forbidden values for n_bits when initializing a ONNXConverter object."""
    with pytest.raises(
        AssertionError,
        match=r"Invalid n_bits, either pass an integer or a dictionary containing integer values",
    ):
        ONNXConverter(n_bits=n_bits, numpy_model=numpy_module)


n_bits_wrong_key = [
    pytest.param(
        {
            "model_inputs": 8,
            "model_outputs": 3,
            "op_inputs": 8,
            "op_weights": 8,
        },
        id="n_bits_larger_op_inputs_than_model_outputs",
    ),
]


@pytest.mark.parametrize("n_bits", n_bits_wrong_key)
def test_n_bits_wrong_key(n_bits):
    """Tests forbidden values for n_bits when initializing a ONNXConverter object."""
    with pytest.raises(
        AssertionError,
        match=(
            r"Using fewer bits to represent the model_outputs than the op inputs is not "
            r"recommended."
        ),
    ):
        ONNXConverter(n_bits=n_bits, numpy_model=numpy_module)
