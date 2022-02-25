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
