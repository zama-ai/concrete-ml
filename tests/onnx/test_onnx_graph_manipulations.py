"""Test file for onnx graph manipulations."""

import onnx
from onnx import helper

from concrete.ml.onnx.convert import OPSET_VERSION_FOR_ONNX_EXPORT
from concrete.ml.onnx.onnx_model_manipulations import remove_unused_constant_nodes


def test_remove_unused_constant_nodes():
    """Test remove_unused_constant_nodes"""

    unused_constant = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["unused_constant"],
        name="unused_constant_node",
        value_float=1.0,
    )

    used_constant = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["used_constant"],
        name="used_constant_node",
        value_float=0.0,
    )

    other_used_constant = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["other_used_constant"],
        name="other_used_constant_node",
        value_float=0.0,
    )

    identity_node = helper.make_node(
        "Identity",
        inputs=["used_constant"],
        outputs=["identity"],
        name="indentity_node",
    )

    output = helper.make_tensor_value_info("used_constant", onnx.TensorProto.FLOAT, ())
    output_2 = helper.make_tensor_value_info("other_used_constant", onnx.TensorProto.FLOAT, ())
    identity_output = helper.make_tensor_value_info("indentity", onnx.TensorProto.FLOAT, ())

    graph_def = helper.make_graph(
        nodes=[unused_constant, used_constant, other_used_constant, identity_node],
        name="test_remove_constants",
        inputs=[],
        outputs=[output, output_2, identity_output],
        initializer=[],
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name="onnx-example")
    model_def.opset_import[0].version = OPSET_VERSION_FOR_ONNX_EXPORT

    model_def = onnx.shape_inference.infer_shapes(model_def)

    onnx.checker.check_model(model_def)

    # Check the used constant is seen as a graph output and the unused constant is not
    assert "used_constant" in set(out.name for out in model_def.graph.output)
    assert "unused_constant" not in set(out.name for out in model_def.graph.output)

    # Check both constants are in the graph nodes
    assert "used_constant" in set(node.output[0] for node in model_def.graph.node)
    assert "unused_constant" in set(node.output[0] for node in model_def.graph.node)

    remove_unused_constant_nodes(model_def)

    onnx.checker.check_model(model_def)

    # Check the used constant is seen as a graph output and the unused constant is not
    assert "used_constant" in set(out.name for out in model_def.graph.output)
    assert "unused_constant" not in set(out.name for out in model_def.graph.output)

    # Check that used_constant is still in the graph while unused_constant has been removed
    assert "used_constant" in set(node.output[0] for node in model_def.graph.node)
    assert "unused_constant" not in set(node.output[0] for node in model_def.graph.node)
