"""A torch to numpy module."""
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy
import torch
from torch import nn

from ..onnx.convert import get_equivalent_numpy_forward_and_onnx_model


class NumpyModule:
    """General interface to transform a torch.nn.Module to numpy module.

    Args:
        torch_model (nn.Module): A fully trained, torch model alond with its parameters.
        dummy_input (Union[torch.Tensor, Tuple[torch.Tensor, ...]]): Sample tensors for all the
            module inputs, used in the ONNX export to get a simple to manipulate nn representation.
        debug_onnx_output_file_path: (Optional[Union[Path, str]], optional): An optional path to
            indicate where to save the ONNX file exported by torch for debug.
            Defaults to None.
    """

    def __init__(
        self,
        torch_model: nn.Module,
        dummy_input: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        debug_onnx_output_file_path: Optional[Union[Path, str]] = None,
    ):
        self.numpy_forward, self.onnx_model = get_equivalent_numpy_forward_and_onnx_model(
            torch_model, dummy_input, debug_onnx_output_file_path
        )

    def __call__(self, *args: numpy.ndarray) -> Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]:
        return self.forward(*args)

    def forward(self, *args: numpy.ndarray) -> Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]:
        """Apply a forward pass on args with the equivalent numpy function only.

        Args:
            *args: the inputs of the forward function

        Returns:
            Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]: result of the forward on the given
                inputs
        """
        outputs = self.numpy_forward(*args)
        return outputs[0] if len(outputs) == 1 else outputs
