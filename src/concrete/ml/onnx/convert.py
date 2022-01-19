"""ONNX conversion related code."""

import tempfile
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy
import onnx
import torch
from onnx import checker

from .onnx_utils import IMPLEMENTED_ONNX_OPS, execute_onnx_with_numpy


def get_equivalent_numpy_forward(
    torch_module: torch.nn.Module,
    dummy_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
    output_onnx_file: Optional[Union[Path, str]] = None,
) -> Callable[..., Tuple[numpy.ndarray, ...]]:
    """Get the numpy equivalent forward of the provided torch Module.

    Args:
        torch_module (torch.nn.Module): the torch Module for which to get the equivalent numpy
            forward.
        dummy_input (Union[torch.Tensor, Tuple[torch.Tensor, ...]]): dummy inputs for ONNX export.
        output_onnx_file (Optional[Union[Path, str]], optional): Path to save the ONNX file to. Will
            use a temp file if not provided.
            Defaults to None.

    Raises:
        ValueError: Raised if there is an unsupported ONNX operator required to convert the torch
            model to numpy.

    Returns:
        Callable[..., Tuple[numpy.ndarray, ...]]: The function that will execute the equivalent
            numpy code to the passed torch_module.
    """

    output_onnx_file_path = Path(
        tempfile.mkstemp(suffix=".onnx")[1]
        if (use_tempfile := (output_onnx_file is None))
        else output_onnx_file
    )

    torch.onnx.export(torch_module, dummy_input, str(output_onnx_file_path), opset_version=14)
    equivalent_onnx_model = onnx.load_model(output_onnx_file_path)
    checker.check_model(equivalent_onnx_model)

    # Remove the tempfile if we used one
    if use_tempfile:
        output_onnx_file_path.unlink(missing_ok=True)

    required_onnx_operators = set(node.op_type for node in equivalent_onnx_model.graph.node)
    unsupported_operators = required_onnx_operators - IMPLEMENTED_ONNX_OPS

    if len(unsupported_operators) > 0:
        raise ValueError(
            "The following ONNX operators are required to convert the torch model to numpy but are "
            f"not currently implemented: {', '.join(sorted(unsupported_operators))}.\n"
            f"Available ONNX operators: {', '.join(sorted(IMPLEMENTED_ONNX_OPS))}"
        )

    return lambda *args: execute_onnx_with_numpy(equivalent_onnx_model.graph, *args)
