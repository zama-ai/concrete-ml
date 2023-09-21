"""ONNX conversion related code."""

import tempfile
from pathlib import Path
from typing import Callable, Tuple, Union

import numpy
import onnx
import onnxoptimizer
import torch
from onnx import checker, helper

from .onnx_utils import IMPLEMENTED_ONNX_OPS, execute_onnx_with_numpy, get_op_type

OPSET_VERSION_FOR_ONNX_EXPORT = 14


# pylint: disable=too-many-branches
def fuse_matmul_bias_to_gemm(onnx_model: onnx.ModelProto):
    """Fuse sequence of matmul -> add into a gemm node.

    Args:
        onnx_model (onnx.ModelProto): A onnx model to optimize using Mat-Mult + Add -> Gemm

    Returns:
        onnx.ModelProto: the optimized onnx model

    """
    # Convert nodes to list to avoid modifying iterable during iteration
    nodes_list = list(onnx_model.graph.node)

    # Iterate through graph nodes
    for matmul_node in nodes_list:
        # Run only if the node is a MatMul node
        if matmul_node.op_type != "MatMul":
            continue
        # Store MatMul node output name
        matmul_node_output_name = matmul_node.output[0]
        assert len(matmul_node.output) == 1

        # Make sure that only one node uses the output of the mat-mult
        mat_mult_output_use_node = []
        for other_node in onnx_model.graph.node:
            if other_node is matmul_node:
                continue
            if matmul_node_output_name in other_node.input:
                mat_mult_output_use_node.append(other_node)
        if len(mat_mult_output_use_node) != 1:
            continue

        # Check that following node is Add
        add_node = mat_mult_output_use_node[0]
        if add_node.op_type != "Add":
            continue
        assert len(add_node.output) == 1

        # Find other Add input
        bias_other_input_node_name = None
        for input_name in add_node.input:
            if input_name != matmul_node_output_name:
                bias_other_input_node_name = input_name
        assert bias_other_input_node_name is not None

        # Only merge if the input of the add node is an initializer
        # otherwise there might be some scaling issues
        initializer_names = [elt.name for elt in onnx_model.graph.initializer]
        if bias_other_input_node_name not in initializer_names:
            continue

        # Create a GEMM node which combines the MatMul and Add operations
        gemm_node = helper.make_node(
            "Gemm",  # op_type
            [matmul_node.input[0], matmul_node.input[1], bias_other_input_node_name],  # inputs
            [add_node.output[0]],  # outputs
            name="Gemm_Node",
            alpha=1.0,
            beta=1.0,  # attributes
        )
        assert len(gemm_node.output) == 1

        # Replace the MatMul and Add nodes with the GEMM node
        # The graph needs to keep being topologically sorted
        mat_mult_node_index = list(onnx_model.graph.node).index(matmul_node)
        add_node_index = list(onnx_model.graph.node).index(add_node)
        gemm_node_index = max(mat_mult_node_index, add_node_index)

        onnx_model.graph.node.insert(gemm_node_index, gemm_node)
        onnx_model.graph.node.remove(add_node)
        onnx_model.graph.node.remove(matmul_node)

        # Update connections in the graph
        for potential_next_node in onnx_model.graph.node:
            # Check if this node was connected to the add_node
            if add_node.output[0] not in potential_next_node.input:
                continue

            # replace the reference to the old add_node output with the gemm_node output
            for idx, potential_next_node_input in enumerate(potential_next_node.input):
                if potential_next_node_input == add_node.output[0]:
                    potential_next_node.input[idx] = gemm_node.output[0]

        # Update the model's output if necessary
        for model_output in onnx_model.graph.output:
            if model_output.name == add_node.output[0]:
                model_output.name = gemm_node.output[0]

    return onnx_model


def get_equivalent_numpy_forward_from_torch(
    torch_module: torch.nn.Module,
    dummy_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    output_onnx_file: Union[None, Path, str] = None,
) -> Tuple[Callable[..., Tuple[numpy.ndarray, ...]], onnx.ModelProto]:
    """Get the numpy equivalent forward of the provided torch Module.

    Args:
        torch_module (torch.nn.Module): the torch Module for which to get the equivalent numpy
            forward.
        dummy_input (Union[torch.Tensor, Tuple[torch.Tensor, ...]]): dummy inputs for ONNX export.
        output_onnx_file (Optional[Union[Path, str]]): Path to save the ONNX file to. Will
            use a temp file if not provided.
            Defaults to None.

    Returns:
        Tuple[Callable[..., Tuple[numpy.ndarray, ...]], onnx.GraphProto]: The function that will
            execute the equivalent numpy code to the passed torch_module and the generated ONNX
            model.
    """
    output_onnx_file_path = Path(
        tempfile.mkstemp(suffix=".onnx")[1] if output_onnx_file is None else output_onnx_file
    )
    use_tempfile: bool = output_onnx_file is None

    # Export to ONNX
    torch.onnx.export(
        torch_module,
        dummy_input,
        str(output_onnx_file_path),
        opset_version=OPSET_VERSION_FOR_ONNX_EXPORT,
    )
    equivalent_onnx_model = onnx.load_model(str(output_onnx_file_path))
    # Remove the tempfile if we used one
    if use_tempfile:
        output_onnx_file_path.unlink()

    equivalent_numpy_forward, equivalent_onnx_model = get_equivalent_numpy_forward_from_onnx(
        equivalent_onnx_model, check_model=True
    )
    with output_onnx_file_path.open("wb") as file:
        file.write(equivalent_onnx_model.SerializeToString())

    return (
        equivalent_numpy_forward,
        equivalent_onnx_model,
    )


def get_equivalent_numpy_forward_from_onnx(
    onnx_model: onnx.ModelProto,
    check_model: bool = True,
) -> Tuple[Callable[..., Tuple[numpy.ndarray, ...]], onnx.ModelProto]:
    """Get the numpy equivalent forward of the provided ONNX model.

    Args:
        onnx_model (onnx.ModelProto): the ONNX model for which to get the equivalent numpy
            forward.
        check_model (bool): set to True to run the onnx checker on the model.
            Defaults to True.

    Raises:
        ValueError: Raised if there is an unsupported ONNX operator required to convert the torch
            model to numpy.

    Returns:
        Callable[..., Tuple[numpy.ndarray, ...]]: The function that will execute
            the equivalent numpy function.
    """
    if check_model:
        checker.check_model(onnx_model)
    checker.check_model(onnx_model)

    # Optimize ONNX graph
    # List of all currently supported onnx optimizer passes
    # From https://github.com/onnx/optimizer/blob/master/onnxoptimizer/pass_registry.h
    onnx_passes = [
        "fuse_matmul_add_bias_into_gemm",
        "eliminate_nop_pad",
        "fuse_pad_into_conv",
        "extract_constant_to_initializer",
        "eliminate_unused_initializer",
    ]
    equivalent_onnx_model = onnxoptimizer.optimize(onnx_model, onnx_passes)
    checker.check_model(equivalent_onnx_model)
    # Custom optimization
    # ONNX optimizer does not optimize Mat-Mult + Bias pattern into GEMM if the input isn't a matrix
    # We manually do the optimization for this case
    equivalent_onnx_model = fuse_matmul_bias_to_gemm(equivalent_onnx_model)
    checker.check_model(equivalent_onnx_model)

    # Check supported operators
    required_onnx_operators = set(get_op_type(node) for node in equivalent_onnx_model.graph.node)
    unsupported_operators = required_onnx_operators - IMPLEMENTED_ONNX_OPS
    if len(unsupported_operators) > 0:
        raise ValueError(
            "The following ONNX operators are required to convert the torch model to numpy but are "
            f"not currently implemented: {', '.join(sorted(unsupported_operators))}.\n"
            f"Available ONNX operators: {', '.join(sorted(IMPLEMENTED_ONNX_OPS))}"
        )

    # Return lambda of numpy equivalent of onnx execution
    return (
        lambda *args: execute_onnx_with_numpy(equivalent_onnx_model.graph, *args)
    ), equivalent_onnx_model
