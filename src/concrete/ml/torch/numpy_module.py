"""A torch to numpy module."""
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy
import onnx
import torch
from torch import nn

from ..common.debugging import assert_true
from ..common.utils import get_onnx_opset_version
from ..onnx.convert import (
    OPSET_VERSION_FOR_ONNX_EXPORT,
    get_equivalent_numpy_forward,
    get_equivalent_numpy_forward_and_onnx_model,
)


class NumpyModule:
    """General interface to transform a torch.nn.Module to numpy module.

    Args:
        torch_model (Union[nn.Module, onnx.ModelProto]): A fully trained, torch model along with
            its parameters or the onnx graph of the model.
        dummy_input (Union[torch.Tensor, Tuple[torch.Tensor, ...]]): Sample tensors for all the
            module inputs, used in the ONNX export to get a simple to manipulate nn representation.
        debug_onnx_output_file_path: (Optional[Union[Path, str]], optional): An optional path to
            indicate where to save the ONNX file exported by torch for debug.
            Defaults to None.
    """

    def __init__(
        self,
        model: Union[nn.Module, onnx.ModelProto],
        dummy_input: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
        debug_onnx_output_file_path: Optional[Union[Path, str]] = None,
    ):
        if isinstance(model, nn.Module):

            # mypy
            assert (
                dummy_input is not None
            ), "dummy_input must be provided if model is a torch.nn.Module"

            self.numpy_forward, self._onnx_model = get_equivalent_numpy_forward_and_onnx_model(
                model, dummy_input, debug_onnx_output_file_path
            )

        elif isinstance(model, onnx.ModelProto):

            onnx_model_opset_version = get_onnx_opset_version(model)
            assert_true(
                onnx_model_opset_version == OPSET_VERSION_FOR_ONNX_EXPORT,
                f"ONNX version must be {OPSET_VERSION_FOR_ONNX_EXPORT} "
                + f"but it is {onnx_model_opset_version}",
            )

            self._onnx_model = model
            self.numpy_forward = get_equivalent_numpy_forward(model)
        else:
            raise ValueError(
                f"model must be a torch.nn.Module or an onnx.ModelProto, got {type(model).__name__}"
            )

    @property
    def onnx_model(self):
        """Get the ONNX model.

        .. # noqa: DAR201

        Returns:
           _onnx_model (onnx.ModelProto): the ONNX model
        """
        return self._onnx_model

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
