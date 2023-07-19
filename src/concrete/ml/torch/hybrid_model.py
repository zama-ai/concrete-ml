"""Implement the conversion of a torch model to a hybrid fhe/torch inference."""

import uuid
from pathlib import Path
from typing import List, Optional, Union

import torch
from concrete.fhe import Configuration
from torch import nn
from transformers import Conv1D

from ..deployment.fhe_client_server import FHEModelClient, FHEModelDev
from .compile import compile_torch_model


# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3858
def convert_conv1d_to_linear(layer_or_module):
    """Convert all Conv1D layers in a module or a Conv1D layer itself to nn.Linear.

    Args:
        layer_or_module (nn.Module or Conv1D): The module which will be recursively searched
            for Conv1D layers, or a Conv1D layer itself.

    Returns:
        nn.Module or nn.Linear: The updated module with Conv1D layers converted to Linear layers,
            or the Conv1D layer converted to a Linear layer.
    """
    if isinstance(layer_or_module, Conv1D):
        # Get the weight size
        weight_size = layer_or_module.weight.size()

        # Create a new linear layer, and copy the weights
        linear_layer = nn.Linear(weight_size[1], weight_size[0])
        linear_layer.weight.data = layer_or_module.weight.T
        if layer_or_module.bias is not None:
            linear_layer.bias.data = layer_or_module.bias
        return linear_layer

    for name, child in layer_or_module.named_children():
        if isinstance(child, Conv1D):
            # Get the weight size
            weight_size = child.weight.size()

            # Create a new linear layer, and copy the weights
            linear_layer = nn.Linear(weight_size[1], weight_size[0])
            linear_layer.weight.data = child.weight.T
            if child.bias is not None:
                linear_layer.bias.data = child.bias
            setattr(layer_or_module, name, linear_layer)
        else:
            convert_conv1d_to_linear(child)

    return layer_or_module


class RemoteModule(nn.Module):
    """A wrapper class for the modules to be done remotely with FHE."""

    def __init__(
        self,
        module=None,
        server_remote_address=None,
    ):
        super().__init__()
        self.private_module = module
        self.server_remote_address = server_remote_address
        self.calibration_data = []
        self.uid = str(uuid.uuid4())
        self.private_q_module = None
        self.fhe_local_mode = "disable"
        self.client: Optional[FHEModelClient] = None
        self.path_to_keys = None
        self.path_to_client = None

    def init_fhe_client(self, path_to_client: str, path_to_keys: str):
        """Set the clients keys.

        Args:
            path_to_client (str): Path where the client.zip is located.
            path_to_keys (str): Path where keys are located.
        """
        # TODO: here we need to load fhe client.zip with FHEModelClient.
        # Either by getting it from the server with the self.uid or
        # directly getting it when downloading the model from HF.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the remote module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor.
        """
        if self.fhe_local_mode != "disable":
            # for mypy
            assert self.private_module is not None
            assert self.private_q_module is not None
            y = self.private_q_module.forward(x.detach().numpy(), fhe=self.fhe_local_mode)
            y = torch.Tensor(y)
        elif self.private_module is not None:
            if isinstance(x, torch.Tensor):
                self.calibration_data.append(x.detach())
            y = self.private_module(x)
        # TODO: https://github.com/zama-ai/concrete-ml-internal/issues/3869
        # else:
        #     y = self.remote_call(x)
        return y

    def remote_call(self, x: torch.Tensor):
        """Call the remote server to get the private module inference.

        Args:
            x (torch.Tensor): The input tensor.
        """
        # TODO: https://github.com/zama-ai/concrete-ml-internal/issues/3869
        # implement server call and client initialization


class HybridFHEModel:
    """Convert a model to a hybrid model."""

    def __init__(
        self,
        model: nn.Module,
        module_names: Union[str, List[str]],
        server_remote_address=None,
    ):
        self.model = model
        self.module_names = [module_names] if isinstance(module_names, str) else module_names
        self.server_remote_address = server_remote_address
        self.private_modules = {
            name: self._get_module_by_name(self.model, name) for name in self.module_names
        }
        self.remote_modules: dict = {}
        self.private_q_modules: dict = {}
        self.configuration: Configuration = None
        self._replace_modules()

    def _replace_modules(self):
        """Replace the private modules in the model with remote layers."""

        for name in self.module_names:

            # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3858
            # Conv1d introduce reshaping operations which adds more TLU
            self.private_modules[name] = convert_conv1d_to_linear(self.private_modules[name])

            remote_module = RemoteModule(self.private_modules[name], self.server_remote_address)

            self.remote_modules[name] = remote_module

            # Now we need to replace the module in its parent module.
            *path, last = name.split(".")
            parent_module = (
                self._get_module_by_name(self.model, ".".join(path)) if path else self.model
            )
            setattr(parent_module, last, remote_module)

    def __call__(self, x: torch.Tensor, fhe: str = "disable") -> torch.Tensor:
        """Call method to run the model locally with a fhe mode.

        Args:
            x (torch.Tensor): The input tensor.
            fhe (str): The Fully Homomorphic Encryption (FHE) mode (default is "disable").

        Returns:
            (torch.Tensor): The output tensor.
        """

        # Set the fhe mode in each remote module
        for module in self.remote_modules.values():
            module.fhe_local_mode = fhe
        x = self.model(x)
        return x

    @staticmethod
    def _get_module_by_name(model: nn.Module, name: str) -> Union[RemoteModule, nn.Module]:
        """Retrieve the module from the model by its name.

        Args:
            model (nn.Module): The model where the module will be searched.
            name (str): The name of the module to be searched.

        Returns:
            (nn.Module): The found module.

        Raises:
            ValueError: If no module found for the given name.
        """
        for module_name, module in model.named_modules():
            if module_name == name:
                return module
        raise ValueError(f"No module found for name {name}")

    def init_client(self, path_to_client: str, path_to_keys: str):
        """Initialize client for all remote modules.

        Args:
            path_to_client (str): Path to the client.zip files.
            path_to_keys (str): Path to the keys folder.
        """
        # TODO: https://github.com/zama-ai/concrete-ml-internal/issues/3869
        # implement client initialization

    def compile_model(
        self,
        x: torch.Tensor,
        n_bits: int = 8,
        rounding_threshold_bits: int = 8,
        p_error=0.01,
        configuration: Configuration = None,
    ):
        """Compiles the specific layers to FHE.

        Args:
            x (torch.Tensor): The input tensor for the model. This is used to run the model
                once for calibration.
            n_bits (int): The bit precision for quantization during FHE model compilation.
                Default is 8.
            rounding_threshold_bits (int): The number of bits to use for rounding threshold during
                FHE model compilation. Default is 8.
            p_error (float): Error allowed for each table look-up in the circuit.
            configuration (Configuration): A concrete Configuration object specifying the FHE
                encryption parameters. If not specified, a default configuration is used.
        """
        self.model(x)
        self.configuration = configuration

        for name in self.module_names:
            remote_module = self._get_module_by_name(self.model, name)
            assert isinstance(remote_module, RemoteModule)

            calibration_data_tensor = torch.cat(remote_module.calibration_data, dim=0)

            self.private_q_modules[name] = compile_torch_model(
                self.private_modules[name],
                calibration_data_tensor,
                n_bits=n_bits,
                rounding_threshold_bits=rounding_threshold_bits,
                configuration=configuration,
                p_error=p_error,
            )

            self.remote_modules[name].private_q_module = self.private_q_modules[name]

    def _save_fhe_circuit(self, path: Path):
        """Private method that saves the FHE circuits.

        Args:
            path (Path): The directory where the FHE circuit will be saved.
        """

        path = Path(path)
        for name in self.module_names:
            model_dev = FHEModelDev(
                str(path.resolve()) + f"/{name}_fhe_circuit",
                self.private_q_modules[name],
            )
            model_dev.save()

    def save_and_clear_private_info(self, path: Path):
        """Save the PyTorch model to the provided path and also saves the corresponding FHE circuit.

        Args:
            path (Path): The directory where the model and the FHE circuit will be saved.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for name in self.module_names:
            module = self._get_module_by_name(self.model, name)

            # Remove private information
            for attr in ["private_module", "calibration_data", "private_q_module"]:
                if hasattr(module, attr):
                    setattr(module, attr, None)

        # Save the model with a specific filename
        model_path = path / "model.pth"
        torch.save(self.model, model_path.resolve())

        # Save the FHE circuit in the same directory
        self._save_fhe_circuit(path)

    def publish_to_hub(self):
        """Allow the user to push the model and FHE required files to HF Hub."""
        # TODO: implement HuggingFace model hub integration
