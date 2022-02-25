"""Some code to manipulate models."""

from copy import deepcopy
from typing import Iterable

import onnx

from ..common.debugging import assert_true


def simplify_onnx_model(onnx_model: onnx.ModelProto):
    """Simplify an ONNX model, removes unused Constant nodes and Identity nodes.

    Args:
        onnx_model (onnx.ModelProto): the model to simplify.
    """
    remove_unused_constant_nodes(onnx_model)
    remove_identity_nodes(onnx_model)


def remove_unused_constant_nodes(onnx_model: onnx.ModelProto):
    """Remove unused Constant nodes in the provided onnx model.

    Args:
        onnx_model (onnx.ModelProto): the model for which we want to remove unused Constant nodes.
    """

    constants_to_remove = {}

    for node in onnx_model.graph.node:
        if node.op_type == "Constant":
            # Initially we don't know if a constant is used, so it's to be removed by default
            constants_to_remove[node.output[0]] = node
            continue

        for input_ in node.input:
            if input_ in constants_to_remove:
                # If we find a constant that is used, then it is not a constant to remove anymore
                constants_to_remove.pop(input_)

    for graph_output in onnx_model.graph.output:
        if (graph_output_name := graph_output.name) in constants_to_remove:
            # If we find a constant that is used, then it is not a constant to remove anymore
            constants_to_remove.pop(graph_output_name)

    for node in constants_to_remove.values():
        onnx_model.graph.node.remove(node)


# TODO: https://github.com/zama-ai/concrete-ml-internal/issues/410
# Improve that algorithm which is O(N^2) for now
def remove_identity_nodes(onnx_model: onnx.ModelProto):
    """Remove identity nodes from a model.

    Args:
        onnx_model (onnx.ModelProto): the model for which we want to remove Identity nodes.
    """

    # This is avery sub-optimal O(N^2) implementation that needs to be improved
    node_idx = 0
    while node_idx < len(onnx_model.graph.node):
        node = onnx_model.graph.node[node_idx]
        if node.op_type == "Identity":
            identity_input = node.input[0]
            identity_output = node.output[0]
            # We can only look at the end of the graph as nodes are in topological order
            # flake8 has this unresolved issue: https://github.com/PyCQA/pycodestyle/issues/373
            for next_nodes in onnx_model.graph.node[node_idx + 1 :]:  # noqa: E203
                for input_idx, input_ in enumerate(next_nodes.input):
                    if input_ == identity_output:
                        next_nodes.input[input_idx] = identity_input

            for output in onnx_model.graph.output:
                if output.name == identity_output:
                    output.name = identity_input

            onnx_model.graph.node.pop(node_idx)
        else:
            node_idx += 1


def keep_following_outputs_discard_others(
    onnx_model: onnx.ModelProto, outputs_to_keep: Iterable[str]
):
    """Keep the outputs given in outputs_to_keep and remove the others from the model.

    Args:
        onnx_model (onnx.ModelProto): the ONNX model to modify.
        outputs_to_keep (Iterable[str]): the outputs to keep by name.
    """

    outputs_to_keep_set = set(outputs_to_keep)

    graph_outputs = onnx_model.graph.output
    outputs_value_infos_to_keep = []
    for output in graph_outputs:
        if output.name in outputs_to_keep_set:
            outputs_value_infos_to_keep.append(deepcopy(output))

    n_outputs_to_keep = len(outputs_value_infos_to_keep)
    assert_true(n_outputs_to_keep > 0)

    # Put the outputs to keep in the first cells, pop all the others repeatedly
    for idx in range(n_outputs_to_keep):
        graph_outputs[idx].CopyFrom(outputs_value_infos_to_keep[idx])

    while len(graph_outputs) > n_outputs_to_keep:
        graph_outputs.pop(n_outputs_to_keep)

    assert_true(set(output.name for output in graph_outputs) == outputs_to_keep_set)


def replace_uncessary_nodes_by_identity(onnx_model: onnx.ModelProto, op_type_to_replace: list):
    """Replace unecessary nodes by Identity nodes.

    Args:
        onnx_model (onnx.ModelProto): the ONNX model to modify.
        op_type_to_replace (list): the op_type of the nodes to be replaced by Identity nodes.

    Raises:
        ValueError: Wrong replacement by an Identity node.
    """

    op_type_inputs = {}
    # Replace not needed ops by Identity
    for node_index, node in enumerate(onnx_model.graph.node):
        # Save op_type for each node
        for output in node.output:
            op_type_inputs[output] = node.op_type
        if node.op_type in op_type_to_replace:
            # Check that node.input[0] is not a constant
            if node.input[0] != "input_0" and op_type_inputs[node.input[0]] == "Constant":
                raise ValueError(
                    f"Trying to apply identity over a constant input." f"Node: {node.op_type}"
                )  # pragma: no cover
            # Create a Identity node
            new_node = onnx.helper.make_node(
                "Identity",
                inputs=[str(node.input[0])],
                outputs=node.output,
            )
            # Update current node with new_node
            onnx_model.graph.node[node_index].CopyFrom(new_node)


def cut_onnx_graph_after_node_name(onnx_model: onnx.ModelProto, node_name: str) -> str:
    """Cut the graph after the node with the given name.

    Args:
        onnx_model (onnx.ModelProto): the ONNX model to modify.
        node_name (str): the name of the node after which the graph will be cut.
            (node_name is included in the new graph)

    Returns:
        str: the name of the output to keep
    """
    nodes_to_remove = []
    cut_node_reached = False
    for node in onnx_model.graph.node:
        if cut_node_reached:
            nodes_to_remove.append(node)
        if node.name == node_name:
            cut_node_reached = True
            # Create output node
            onnx_model.graph.output[0].CopyFrom(
                onnx.helper.make_tensor_value_info(node.output[0], onnx.TensorProto.FLOAT, [2])
            )
            output_to_follow = onnx_model.graph.output[0].name

    # Remove nodes
    for node in nodes_to_remove:
        onnx_model.graph.node.remove(node)

    return output_to_follow


def remove_transpose_in_first_gemm_node(onnx_model):
    """Find the first Gemm node and remove the transpose option.

    FIXME remove this function once #292 is fixed

    Args:
        onnx_model (onnx.ModelProto): the ONNX model to modify.
    """
    # Find the Gemm node
    for node_index, node in enumerate(onnx_model.graph.node):
        if node.op_type == "Gemm":
            gemm_node_index = node_index
            break

    gemm_node = onnx_model.graph.node[gemm_node_index]
    new_node = onnx.numpy_helper.helper.make_node(
        name=gemm_node.name,
        op_type=gemm_node.op_type,
        inputs=gemm_node.input,
        outputs=gemm_node.output,
        alpha=1.0,
        beta=0.0,
    )
    onnx_model.graph.node[gemm_node_index].CopyFrom(new_node)
