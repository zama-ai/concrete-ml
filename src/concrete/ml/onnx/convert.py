"""ONNX conversion related code."""

import inspect
import tempfile
import warnings
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy
import onnx
import onnxoptimizer
import torch
from onnx import helper
from typing_extensions import TypeAlias

from ..common.debugging import assert_true
from ..onnx.onnx_model_manipulations import convert_first_gather_to_matmul
from .onnx_utils import (
    IMPLEMENTED_ONNX_OPS,
    check_onnx_model,
    execute_onnx_with_numpy,
    execute_onnx_with_numpy_trees,
    get_op_type,
)

NumpyForwardCallable: TypeAlias = Callable[..., Tuple[numpy.ndarray, ...]]
ONNXAndNumpyForwards: TypeAlias = Tuple[
    NumpyForwardCallable, Optional[onnx.ModelProto], NumpyForwardCallable, onnx.ModelProto
]

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
) -> ONNXAndNumpyForwards:
    """Get the numpy equivalent forward of the provided torch Module.

    Args:
        torch_module (torch.nn.Module): the torch Module for which to get the equivalent numpy
            forward.
        dummy_input (Union[torch.Tensor, Tuple[torch.Tensor, ...]]): dummy inputs for ONNX export.
        output_onnx_file (Optional[Union[Path, str]]): Path to save the ONNX file to. Will
            use a temp file if not provided.
            Defaults to None.

    Returns:
        ONNXAndNumpyForwards: The function that will execute the equivalent numpy code to the
            passed torch_module and the generated ONNX model.
    """
    output_onnx_file_path = Path(
        tempfile.mkstemp(suffix=".onnx")[1] if output_onnx_file is None else output_onnx_file
    )
    use_tempfile: bool = output_onnx_file is None

    arguments = list(inspect.signature(torch_module.forward).parameters)

    # Export to ONNX
    torch.onnx.export(
        torch_module,
        dummy_input,
        str(output_onnx_file_path),
        opset_version=OPSET_VERSION_FOR_ONNX_EXPORT,
        input_names=arguments,
    )
    equivalent_onnx_model = onnx.load_model(str(output_onnx_file_path))

    # Check if the inputs are present in the model's graph
    for input_name in arguments:
        assert_true(
            any(input_name == node.name for node in equivalent_onnx_model.graph.input),
            f"Input '{input_name}' is missing in the ONNX graph after export. "
            "Verify the forward pass for issues.",
        )

    # Remove the tempfile if we used one
    if use_tempfile:
        output_onnx_file_path.unlink()

    numpy_preprocessing, onnx_preprocessing, equivalent_numpy_forward, equivalent_onnx_model = (
        get_equivalent_numpy_forward_from_onnx(equivalent_onnx_model)
    )
    with output_onnx_file_path.open("wb") as file:
        file.write(equivalent_onnx_model.SerializeToString())

    return (
        numpy_preprocessing,
        onnx_preprocessing,
        equivalent_numpy_forward,
        equivalent_onnx_model,
    )


def preprocess_onnx_model(
    onnx_model: onnx.ModelProto, check_model: bool
) -> Tuple[Optional[onnx.ModelProto], onnx.ModelProto]:
    """Preprocess the ONNX model to be used for numpy execution.

    Args:
        onnx_model (onnx.ModelProto): the ONNX model for which to get the equivalent numpy
            forward.
        check_model (bool): set to True to run the onnx checker on the model.
            Defaults to True.

    Raises:
        ValueError: Raised if there is an unsupported ONNX operator required to convert the torch
            model to numpy.

    Returns:
        Tuple[Optional[onnx.ModelProto], onnx.ModelProto]: The preprocessing ONNX model and
            preprocessed ONNX model. The preprocessing model is None if there is no preprocessing
            required.
    """

    # All onnx models should be checked, "check_model" parameter must be removed
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4157
    if not check_model:  # pragma: no cover
        warnings.simplefilter("always")
        warnings.warn(
            "`check_model` parameter should always be set to True, to ensure proper onnx model "
            "verification and avoid bypassing essential onnx model validation checks.",
            category=UserWarning,
            stacklevel=2,
        )

    check_onnx_model(onnx_model)

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
    check_onnx_model(equivalent_onnx_model)

    # Custom optimization
    # ONNX optimizer does not optimize Mat-Mult + Bias pattern into GEMM if the input isn't a matrix
    # We manually do the optimization for this case
    equivalent_onnx_model = fuse_matmul_bias_to_gemm(equivalent_onnx_model)
    check_onnx_model(equivalent_onnx_model)

    # Check supported operators
    required_onnx_operators = set(get_op_type(node) for node in equivalent_onnx_model.graph.node)
    unsupported_operators = required_onnx_operators - IMPLEMENTED_ONNX_OPS
    if len(unsupported_operators) > 0:
        raise ValueError(
            "The following ONNX operators are required to convert the torch model to numpy but are "
            f"not currently implemented: {', '.join(sorted(unsupported_operators))}.\n"
            f"Available ONNX operators: {', '.join(sorted(IMPLEMENTED_ONNX_OPS))}"
        )

    # Convert the first Gather node to a matrix multiplication with one-hot encoding
    # In FHE, embedding is either a TLU or a matmul with a one-hot.
    # The second case allows for leveled operation thus much faster.
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4532
    onnx_preprocessing, equivalent_onnx_model = convert_first_gather_to_matmul(
        equivalent_onnx_model
    )

    return onnx_preprocessing, equivalent_onnx_model


def get_equivalent_numpy_forward_from_onnx(
    onnx_model: onnx.ModelProto,
    check_model: bool = True,
) -> ONNXAndNumpyForwards:
    """Get the numpy equivalent forward of the provided ONNX model.

    Args:
        onnx_model (onnx.ModelProto): the ONNX model for which to get the equivalent numpy
            forward.
        check_model (bool): set to True to run the onnx checker on the model.
            Defaults to True.

    Returns:
        ONNXAndNumpyForwards: The function that will execute the equivalent numpy function.
    """

    onnx_preprocessing, equivalent_onnx_model = preprocess_onnx_model(onnx_model, check_model)

    def create_numpy_forward(model: Optional[onnx.ModelProto]) -> NumpyForwardCallable:
        """Create numpy forward function.

        Args:
            model (onnx.ModelProto): The ONNX model to execute.

        Returns:
            NumpyForwardCallable: The numpy equivalent of the ONNX model.
        """
        if model is None:
            # Return the inputs as is
            return lambda *args: args
        return lambda *args: execute_onnx_with_numpy(model.graph, *args)

    # Return lambda of numpy equivalent of onnx execution
    return (
        create_numpy_forward(onnx_preprocessing),
        onnx_preprocessing,
        create_numpy_forward(equivalent_onnx_model),
        equivalent_onnx_model,
    )


def get_equivalent_numpy_forward_from_onnx_tree(
    onnx_model: onnx.ModelProto,
    check_model: bool = True,
    lsbs_to_remove_for_trees: Optional[Tuple[int, int]] = None,
) -> Tuple[NumpyForwardCallable, onnx.ModelProto]:
    """Get the numpy equivalent forward of the provided ONNX model for tree-based models only.

    Args:
        onnx_model (onnx.ModelProto): the ONNX model for which to get the equivalent numpy
            forward.
        check_model (bool): set to True to run the onnx checker on the model.
            Defaults to True.
        lsbs_to_remove_for_trees (Optional[Tuple[int, int]]): This parameter is exclusively used for
            optimizing tree-based models. It contains the values of the least significant bits to
            remove during the tree traversal, where the first value refers to the first comparison
            (either "less" or "less_or_equal"), while the second value refers to the "Equal"
            comparison operation. Default to None, as it is not applicable to other types of models.

    Returns:
        Tuple[NumpyForwardCallable, onnx.ModelProto]: The function that will
            execute the equivalent numpy function.
    """

    _, equivalent_onnx_model = preprocess_onnx_model(onnx_model, check_model)

    # Return lambda of numpy equivalent of onnx execution
    return (
        lambda *args: execute_onnx_with_numpy_trees(
            equivalent_onnx_model.graph, lsbs_to_remove_for_trees, *args
        )
    ), equivalent_onnx_model
