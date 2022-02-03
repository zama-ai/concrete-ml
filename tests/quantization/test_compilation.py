"""Test Neural Networks compilations"""
import numpy
import onnx
import pytest
import torch
from onnx import helper, numpy_helper
from torch import nn

from concrete.ml.quantization import QuantizedArray, QuantizedGemm
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
@pytest.mark.parametrize(
    "activation",
    [
        nn.ReLU6,
        nn.Sigmoid,
    ],
)
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


class NumpyModuleTest(NumpyModule):
    """Test class to build NumpyModule in an alternative way."""

    def __init__(self, onnx_model: onnx.ModelProto):  # pylint: disable=super-init-not-called
        self.numpy_forward = lambda x: x
        self.onnx_model = onnx_model


def test_post_training_quantization_constant_folding():
    """Test to check that constant folding works properly."""

    # First add a few initializers
    f_one = 1.0
    f_zero = 0.0

    f_ones_init = numpy_helper.from_array(numpy.ones((10, 10), dtype=numpy.float32), "f_ones")
    f_zeros_init = numpy_helper.from_array(numpy.zeros((10, 10), dtype=numpy.float32), "f_zeros")

    constant_f_one = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["constant_f_one"],
        name="constant_f_one_node",
        **{"value_float": f_one},
    )
    constant_f_zero = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["constant_f_zero"],
        name="constant_f_zero_node",
        **{"value_float": f_zero},
    )

    add_one_and_zero = helper.make_node(
        "Add",
        inputs=["constant_f_one", "constant_f_zero"],
        outputs=["add_one_and_zero"],
        name="add_one_and_zero_node",
    )

    sub_zeros_and_ones = helper.make_node(
        "Sub",
        inputs=["f_zeros", "f_ones"],
        outputs=["sub_zeros_and_ones"],
        name="sub_zeros_and_ones_node",
    )

    model_input_name = "x_input"
    x_input = helper.make_tensor_value_info(model_input_name, onnx.TensorProto.FLOAT, [10, 10])
    model_output_name = "y_output"
    y_output = helper.make_tensor_value_info(model_output_name, onnx.TensorProto.FLOAT, [10, 10])

    add_input_f_one = helper.make_node(
        "Add",
        inputs=["constant_f_zero", "constant_f_one"],
        outputs=["input_plus_one"],
        name="input_plus_one_node",
    )

    exp_f_zero = helper.make_node(
        "Exp",
        inputs=["constant_f_zero"],
        outputs=["exp_f_zero"],
        name="exp_f_zero_node",
    )

    add_input_f_one_exp = helper.make_node(
        "Add",
        inputs=["input_plus_one", "exp_f_zero"],
        outputs=["input_plus_two"],
        name="input_plus_two_node",
    )

    negative_input = helper.make_node(
        "Mul",
        inputs=["input_plus_two", "sub_zeros_and_ones"],
        outputs=["negative_input"],
        name="negative_input_node",
    )

    negative_input_plus_one = helper.make_node(
        "Add",
        inputs=["negative_input", "add_one_and_zero"],
        outputs=["negative_input_plus_one"],
        name="negative_input_plus_one_node",
    )

    gemm = helper.make_node(
        "Gemm",
        inputs=["x_input", "negative_input_plus_one"],
        outputs=[model_output_name],
        name="matmul_with_folded_cst",
        **{"transA": 0, "transB": 0, "alpha": 1.0, "beta": 1.0},
    )

    graph_def = helper.make_graph(
        nodes=[
            constant_f_one,
            constant_f_zero,
            add_one_and_zero,
            sub_zeros_and_ones,
            add_input_f_one,
            exp_f_zero,
            add_input_f_one_exp,
            negative_input,
            negative_input_plus_one,
            gemm,
        ],
        name="test_constant_folding",
        inputs=[x_input],
        outputs=[y_output],
        initializer=[f_ones_init, f_zeros_init],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name="onnx-example")
    model_def.opset_import[0].version = 14

    model_def = onnx.shape_inference.infer_shapes(model_def)

    onnx.checker.check_model(model_def)

    numpy_model = NumpyModuleTest(model_def)
    # Quantize with post-training static method
    post_training_quant = PostTrainingAffineQuantization(7, numpy_model)

    numpy_input = numpy.random.random(size=(10, 10))

    post_training_quant.quantize_module(numpy_input)

    expected_constant = (0.0 + 1.0) + (
        (numpy.zeros((10, 10), dtype=numpy.float32) - numpy.ones((10, 10), dtype=numpy.float32))
        * ((1.0 + 0.0) + numpy.exp(0.0))
    )

    # Check we have only the gemm node in the quant_ops_dict after quantization as it's the only one
    # that depends on a variable input
    assert len(post_training_quant.quant_ops_dict) == 1
    assert model_output_name in post_training_quant.quant_ops_dict
    assert isinstance(
        q_gemm := post_training_quant.quant_ops_dict[model_output_name], QuantizedGemm
    )
    assert numpy.array_equal(q_gemm.constant_inputs[1].values, expected_constant)
