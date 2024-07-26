"""A torch to numpy module."""

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy
import onnx
import torch
from torch import nn

from ..common.debugging import assert_true
from ..common.utils import get_onnx_opset_version, to_tuple
from ..onnx.convert import (
    OPSET_VERSION_FOR_ONNX_EXPORT,
    get_equivalent_numpy_forward_from_onnx,
    get_equivalent_numpy_forward_from_torch,
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

            (
                self._numpy_preprocessing,
                self._onnx_preprocessing,
                self.numpy_forward,
                self._onnx_model,
            ) = get_equivalent_numpy_forward_from_torch(
                model, dummy_input, debug_onnx_output_file_path
            )

        elif isinstance(model, onnx.ModelProto):

            onnx_model_opset_version = get_onnx_opset_version(model)
            assert_true(
                onnx_model_opset_version == OPSET_VERSION_FOR_ONNX_EXPORT,
                f"ONNX version must be {OPSET_VERSION_FOR_ONNX_EXPORT} "
                + f"but it is {onnx_model_opset_version}",
            )

            (
                self._numpy_preprocessing,
                self._onnx_preprocessing,
                self.numpy_forward,
                self._onnx_model,
            ) = get_equivalent_numpy_forward_from_onnx(model)
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

    @property
    def onnx_preprocessing(self):
        """Get the ONNX preprocessing.

        .. # noqa: DAR201

        Returns:
           _onnx_preprocessing (onnx.ModelProto): the ONNX preprocessing
        """
        return self._onnx_preprocessing

    def __call__(self, *args: numpy.ndarray) -> Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]:
        return self.forward(*args)

    def pre_processing(self, *args: numpy.ndarray) -> Tuple[numpy.ndarray, ...]:
        """Apply a preprocessing pass on args with the equivalent numpy function only.

        Args:
            *args: the inputs of the preprocessing function

        Returns:
            Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]: result of the preprocessing on the
                given inputs or the original inputs if no preprocessing function is defined
        """
        return (
            args
            if self._numpy_preprocessing is None
            else to_tuple(self._numpy_preprocessing(*args))
        )

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
