"""Some code to manipulate models."""

from copy import deepcopy
from typing import Iterable, List, Optional, Tuple

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
    """Remove the first node matching node_op_type and its following nodes from the ONNX graph.

    Args:
        onnx_model (onnx.ModelProto): The onnx model.
        node_op_type (str): The node's op_type whose following nodes as well as itself will be
            removed.
        fail_if_not_found (bool): If true, raise an error if no node matched the given op_type.

    Raises:
        ValueError: If no node matched the given op_type and fail_if_not_found is set to True
    """
    # Get the list of node that matches the given operator type
    nodes_of_type = [node for node in onnx_model.graph.node if node.op_type == node_op_type]

    if nodes_of_type:

        # Get the (first) input's same of the first node that matches the given operator type
        input_name = nodes_of_type[0].input[0]

        # Find the node that has the input_name as output name (i.e., the node tha precedes the
        # selected node) and retrieve its name
        node_to_cut = next(node for node in onnx_model.graph.node if node.output[0] == input_name)
        node_to_cut_name = node_to_cut.name

        _clean_graph_at_node_name(
            onnx_model, node_name=node_to_cut_name, fail_if_not_found=fail_if_not_found
        )

    elif fail_if_not_found:  # pragma: no cover
        raise ValueError(  # pragma: no cover
            f"Node of type '{node_op_type}' could not be found within the ONNX graph"
        )


def clean_graph_after_node_op_type(
    onnx_model: onnx.ModelProto, node_op_type: str, fail_if_not_found: bool = True
):
    """Remove the nodes following first node matching node_op_type from the ONNX graph.

    Args:
        onnx_model (onnx.ModelProto): The onnx model.
        node_op_type (str): The node's op_type whose following nodes will be removed.
        fail_if_not_found (bool): If true, raise an error if no node matched the given op_type.

    Raises:
        ValueError: If no node matched the given op_type and fail_if_not_found is set to True
    """
    # Get the list of node that matches the given operator type
    nodes_of_type = [node for node in onnx_model.graph.node if node.op_type == node_op_type]

    if node_op_type:

        # Get the first node that matches the given operator type and retrieve its name
        node_to_cut_name = nodes_of_type[0].name

        _clean_graph_at_node_name(
            onnx_model, node_name=node_to_cut_name, fail_if_not_found=fail_if_not_found
        )

    elif fail_if_not_found:  # pragma: no cover
        raise ValueError(  # pragma: no cover
            f"Node of type '{node_op_type}' could not be found within the ONNX graph"
        )


def _clean_graph_at_node_name(
    onnx_model: onnx.ModelProto, node_name: str, fail_if_not_found: bool = True
):
    """Remove the first node matching node_name and its following nodes from the ONNX graph.

    Args:
        onnx_model (onnx.ModelProto): The onnx model.
        node_name (str): The node's name whose following nodes as well as itself will be removed.
        fail_if_not_found (bool): If true, raise an error if no node matched the given name.

    Raises:
        ValueError: If no node matched the given name and fail_if_not_found is set to True
    """
    # Build a dictionary associating all initializers with their names
    initializers_name_to_instance = {
        initializer.name: initializer for initializer in onnx_model.graph.initializer
    }

    nodes_to_remove = []
    initializers_to_remove = []
    output_to_follow = "variable"
    op_reached = False

    # Find nodes to remove
    for node in onnx_model.graph.node:

        # If the operator has previously been reached, store the node
        if op_reached:
            nodes_to_remove.append(node)

            # If one of the node's inputs is an initializer, store the initializer
            for node_input_name in node.input:
                if node_input_name in initializers_name_to_instance:
                    initializers_to_remove.append(initializers_name_to_instance[node_input_name])

        # Else, if the current node represents the operator, retrieve its output node
        elif node.name == node_name:
            op_reached = True

            # Create output node
            onnx_model.graph.output[0].CopyFrom(
                onnx.helper.make_tensor_value_info(node.output[0], onnx.TensorProto.FLOAT, [2])
            )
            output_to_follow = node.output[0]

    # If the graph has been covered and an operator to remove has been found, remove it and its
    # following nodes as well as the initializers they used as inputs
    if op_reached:
        for node in nodes_to_remove:
            onnx_model.graph.node.remove(node)

        for initializer in initializers_to_remove:
            onnx_model.graph.initializer.remove(initializer)

    elif fail_if_not_found:  # pragma: no cover
        raise ValueError(  # pragma: no cover
            f"Node '{node_name}' could not be found within the ONNX graph"
        )

    # Keep the output node
    keep_following_outputs_discard_others(onnx_model, [output_to_follow])


# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4532
# Function to convert the first Gather nodes
# to matrix multiplications with one-hot encoding as a pre-processing step
def convert_first_gather_to_matmul(
    onnx_model: onnx.ModelProto,
) -> Tuple[Optional[onnx.ModelProto], onnx.ModelProto]:
    """Convert the first Gather node to a matrix multiplication node.

    In FHE, Gather is a costly operation since it can involve many PBS.
    When it appears first in the onnx model, we can remove it and replace it by a matrix
    multiplication node by converting the indices to a one-hot encoding.

    Args:
        onnx_model (onnx.ModelProto): The onnx model.

    Returns:
        Tuple[Optional[onnx.ModelProto], onnx.ModelProto]: The pre-processing model and the modified
            onnx model.
    """
    pre_processing_nodes = []
    modified = False
    depth_tensors = []
    gather_depths = {}

    for node in onnx_model.graph.node:
        if node.op_type == "Gather" and node.input[1] in [
            input.name for input in onnx_model.graph.input
        ]:
            # Extract the inputs and output of the Gather node
            data_input = node.input[0]
            indices_input = node.input[1]
            gather_output = node.output[0]

            # Find the shape of the data_input (embedding matrix)
            data_shape_initializer = next(
                (init for init in onnx_model.graph.initializer if init.name == data_input), None
            )
            assert data_shape_initializer is not None, f"Shape of {data_input} not found"

            # Extract the depth arg for the OneHot node using the embedding matrix
            depth = data_shape_initializer.dims[0]
            depth_name = f"depth_{node.name}"
            gather_depths[node.name] = depth

            # Create a node for OneHot operation
            pre_processed_x = f"pre_processed_{indices_input}"
            pre_processing_nodes.append(
                onnx.helper.make_node(
                    "OneHot",
                    inputs=[indices_input, depth_name, "values"],
                    outputs=[pre_processed_x],
                )
            )

            # Store the depth tensor for this Gather node
            depth_tensor = onnx.helper.make_tensor(depth_name, onnx.TensorProto.INT64, [1], [depth])
            depth_tensors.append(depth_tensor)

            # Replace Gather node with MatMul node
            matmul_node = onnx.helper.make_node(
                "MatMul", inputs=[indices_input, data_input], outputs=[gather_output]
            )
            onnx_model.graph.node.remove(node)

            # Insert the new node at the beginning of the graph
            onnx_model.graph.node.insert(0, matmul_node)
            modified = True

    if not modified:
        return None, onnx_model

    # Create a tensor for the values of the OneHot node
    values_tensor = onnx.helper.make_tensor("values", onnx.TensorProto.FLOAT, [2], [0.0, 1.0])

    # Create pre-processing graph
    pre_processing_graph = onnx.helper.make_graph(
        pre_processing_nodes,
        "pre_processing_graph",
        [
            onnx.helper.make_tensor_value_info(node.input[0], onnx.TensorProto.INT64, [None])
            for node in pre_processing_nodes
        ],
        [
            onnx.helper.make_tensor_value_info(
                node.output[0], onnx.TensorProto.FLOAT, [None, depth]
            )
            for node, depth in zip(pre_processing_nodes, gather_depths.values())
        ],
        depth_tensors + [values_tensor],
    )

    pre_processing_onnx = onnx.helper.make_model(
        pre_processing_graph, opset_imports=[onnx.helper.make_opsetid("", 14)]
    )

    # Check the pre-processing onnx
    onnx.checker.check_model(pre_processing_onnx)

    return pre_processing_onnx, onnx_model
