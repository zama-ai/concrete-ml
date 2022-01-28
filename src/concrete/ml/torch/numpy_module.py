"""A torch to numpy module."""
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy
import torch
from torch import nn

from ..common.debugging import assert_not_reached
from ..onnx.convert import get_equivalent_numpy_forward_and_onnx_model


class NumpyModule:
    """General interface to transform a torch.nn.Module to numpy module."""

    IMPLEMENTED_MODULES = {nn.Linear, nn.Sigmoid, nn.ReLU, nn.ReLU6, nn.Tanh}

    def __init__(self, torch_model: nn.Module):
        """Initialize our numpy module.

        Current constraint:    All objects used in the forward have to be defined in the
                               __init__() of torch.nn.Module and follow the exact same order.
                               (i.e. each linear layer must have one variable defined in the
                               right order). This constraint will disappear when
                               TorchScript is in place. (issue #818)

        Args:
            torch_model (nn.Module): A fully trained, torch model alond with its parameters.
        """
        self.torch_model = torch_model
        self.check_compatibility()
        self.convert_to_numpy()

    def check_compatibility(self):
        """Check the compatibility of all layers in the torch model.

        Raises:
            ValueError: Raises an error if a layer is not implemented

        """

        for _, layer in self.torch_model.named_children():
            if (layer_type := type(layer)) not in self.IMPLEMENTED_MODULES:
                raise ValueError(
                    f"The following module is currently not implemented: {layer_type.__name__}. "
                    f"Please stick to the available torch modules: "
                    f"{', '.join(sorted(module.__name__ for module in self.IMPLEMENTED_MODULES))}."
                )

    def convert_to_numpy(self):
        """Transform all parameters from torch tensor to numpy arrays."""
        self.numpy_module_dict = {}

        for name, weights in self.torch_model.state_dict().items():
            params = weights.detach().numpy()
            self.numpy_module_dict[name] = params.T if "weight" in name else params

    def __call__(self, x: numpy.ndarray) -> numpy.ndarray:
        """Apply the forward pass.

        Args:
            x (numpy.ndarray): The data on which to apply the forward

        Returns:
            numpy.array: Processed input.
        """
        return self.forward(x)

    def forward(self, x: numpy.ndarray) -> numpy.ndarray:
        """Apply a forward pass with numpy function only.

        Args:
            x (numpy.array): Input to be processed in the forward pass.

        Returns:
            x (numpy.array): Processed input.
        """

        for name, layer in self.torch_model.named_children():

            if isinstance(layer, nn.Linear):
                # Apply a matmul product and add the bias.
                x = (
                    x @ self.numpy_module_dict[f"{name}.weight"]
                    + self.numpy_module_dict[f"{name}.bias"]
                )
            elif isinstance(layer, nn.Sigmoid):
                x = 1 / (1 + numpy.exp(-x))
            elif isinstance(layer, nn.ReLU):
                x = numpy.maximum(0, x)
            elif isinstance(layer, nn.ReLU6):
                x = numpy.minimum(numpy.maximum(0, x), 6)
            elif isinstance(layer, nn.Tanh):
                x = numpy.tanh(x)
            else:
                assert_not_reached("missing activation")  # pragma: no cover
        return x


class NewNumpyModule:
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
