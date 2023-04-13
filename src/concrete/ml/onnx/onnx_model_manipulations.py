"""Some code to manipulate models."""

from copy import deepcopy
from typing import Iterable, List

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
            # Initially we don't know if a constant is used, so it is to be removed by default
            constants_to_remove[node.output[0]] = node
            continue

        for input_ in node.input:
            if input_ in constants_to_remove:
                # If we find a constant that is used, then it is not a constant to remove anymore
                constants_to_remove.pop(input_)

    for graph_output in onnx_model.graph.output:
        graph_output_name = graph_output.name
        if graph_output_name in constants_to_remove:
            # If we find a constant that is used, then it is not a constant to remove anymore
            constants_to_remove.pop(graph_output_name)

    for node in constants_to_remove.values():
        onnx_model.graph.node.remove(node)


# This algorithm needs to be improved, currently it runs in O(N^2)
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/410
def remove_identity_nodes(onnx_model: onnx.ModelProto):
    """Remove identity nodes from a model.

    Args:
        onnx_model (onnx.ModelProto): the model for which we want to remove Identity nodes.
    """

    # This is a very sub-optimal O(N^2) implementation that needs to be improved
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


def remove_node_types(onnx_model: onnx.ModelProto, op_types_to_remove: List[str]):
    """Remove unnecessary nodes from the ONNX graph.

    Args:
        onnx_model (onnx.ModelProto): The ONNX model to modify.
        op_types_to_remove (List[str]): The node types to remove from the graph.

    Raises:
        ValueError: Wrong replacement by an Identity node.
    """

    op_type_inputs = {}
    op_type_inputs["input_0"] = "Input"

    # Reference initializer as Constant
    for initializer in onnx_model.graph.initializer:
        op_type_inputs[initializer.name] = "Constant"

    # Replace the nodes to remove by Identity nodes
    for node_index, node in enumerate(onnx_model.graph.node):
        # Save op_type for each node
        for output in node.output:
            op_type_inputs[output] = node.op_type

        # If the current node type needs to be removed
        if node.op_type in op_types_to_remove:

            # Find the non-constant input
            non_constant_input = None
            for input_ in node.input:
                if op_type_inputs[input_] != "Constant":
                    non_constant_input = input_
                    break

            # Check that node.input[0] is not a constant
            if non_constant_input is None:
                raise ValueError(
                    f"Trying to apply identity over a constant input." f"Node: {node.op_type}"
                )  # pragma: no cover

            # Create an Identity node
            new_node = onnx.helper.make_node(
                "Identity",
                inputs=[str(non_constant_input)],
                outputs=node.output,
            )

            # Update the current node with the new Identity node
            onnx_model.graph.node[node_index].CopyFrom(new_node)

    # Remove Constant and Identity nodes from the graph
    simplify_onnx_model(onnx_model)


def clean_graph_at_node_op_type(
    onnx_model: onnx.ModelProto, node_op_type: str, fail_if_not_found: bool = True
):
    """Clean the graph of the onnx model by removing nodes at the given node type.

    Note: the specified node_type is also removed.

    Args:
        onnx_model (onnx.ModelProto): The onnx model.
        node_op_type (str): The node's op_type whose following nodes will be removed.
        fail_if_not_found (bool): If true, abort if the node op_type is not found

    Raises:
        ValueError: if fail_if_not_found is set
    """

    target_node_list = [node for node in onnx_model.graph.node if node.op_type == node_op_type]
    assert_true(
        len(target_node_list) == 1,
        "Multiple ReduceSum nodes found which is unexpected in tree-based models",
    )

    target_node = target_node_list[0]

    # Get the input to ReduceSum where index 0 is the data
    input_name = target_node.input[0]

    # Find the node that has the input_name as output name
    node_to_cut = next(node for node in onnx_model.graph.node if node.output[0] == input_name)
    node_name = node_to_cut.name

    nodes_to_remove = []
    output_to_follow = "variable"
    op_reached = False

    # Find nodes to remove
    for node in onnx_model.graph.node:

        # If the operator has previously been reached, store the node
        if op_reached:
            nodes_to_remove.append(node)

        # ELse, if the current node represents the operator, retrieve its output node
        elif node.name == node_name:
            op_reached = True

            # Create output node
            onnx_model.graph.output[0].CopyFrom(
                onnx.helper.make_tensor_value_info(node.output[0], onnx.TensorProto.FLOAT, [2])
            )
            output_to_follow = node.output[0]

    # Once the graph has been covered and a operator was found, remove its its following nodes
    if op_reached:
        for node in nodes_to_remove:
            onnx_model.graph.node.remove(node)
    elif fail_if_not_found:  # pragma: no cover
        raise ValueError(f"Error, can't find {node_name}")  # pragma: no cover

    # Keep the output node
    keep_following_outputs_discard_others(onnx_model, [output_to_follow])


def clean_graph_after_node_op_type(
    onnx_model: onnx.ModelProto, node_op_type: str, fail_if_not_found: bool = True
):
    """Clean the graph of the onnx model by removing nodes after the given node type.

    Args:
        onnx_model (onnx.ModelProto): The onnx model.
        node_op_type (str): The node's op_type whose following nodes will be removed.
        fail_if_not_found (bool): If true, abort if the node op_type is not found

    Raises:
        ValueError: if the node op_type is not found and if fail_if_not_found is set

    """
    nodes_to_remove = []
    output_to_follow = "variable"
    op_reached = False

    # Find nodes to remove
    for node in onnx_model.graph.node:

        # If the operator has previously been reached, store the node
        if op_reached:
            nodes_to_remove.append(node)

        # Else, if the current node represents the operator, retrieve its output node
        elif node.op_type == node_op_type:
            op_reached = True

            # Create output node
            onnx_model.graph.output[0].CopyFrom(
                onnx.helper.make_tensor_value_info(node.output[0], onnx.TensorProto.FLOAT, [2])
            )
            output_to_follow = node.output[0]

    # Once the graph has been covered and a operator was found, remove its its following nodes
    if op_reached:
        for node in nodes_to_remove:
            onnx_model.graph.node.remove(node)
    elif fail_if_not_found:  # pragma: no cover
        raise ValueError(f"Error, can't find {node_op_type}")  # pragma: no cover

    # Keep the output node
    keep_following_outputs_discard_others(onnx_model, [output_to_follow])
