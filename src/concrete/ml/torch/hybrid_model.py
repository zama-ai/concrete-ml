"""Implement the conversion of a torch model to a hybrid fhe/torch inference."""

import ast
import enum
import io
import sys
import time
import uuid
from abc import abstractmethod
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy
import requests
import torch
from brevitas.quant_tensor import QuantTensor
from concrete.fhe import Configuration
from torch import nn
from transformers import Conv1D

from ..common.utils import MAX_BITWIDTH_BACKWARD_COMPATIBLE
from ..deployment.fhe_client_server import FHEModelClient, FHEModelDev, FHEModelServer
from .compile import (
    QuantizedModule,
    compile_brevitas_qat_model,
    compile_torch_model,
    has_any_qnn_layers,
)


class HybridFHEMode(enum.Enum):
    """Simple enum for different modes of execution of HybridModel."""

    DISABLE = "disable"  # Use torch weights
    REMOTE = "remote"  # Use remote FHE server
    SIMULATE = "simulate"  # Use FHE simulation
    CALIBRATE = "calibrate"  # Use calibration (to run before FHE compilation)
    EXECUTE = "execute"  # Use FHE execution


def tuple_to_underscore_str(tup: Tuple) -> str:
    """Convert a tuple to a string representation.

    Args:
        tup (Tuple): a tuple to change into string representation

    Returns:
        str: a string representing the tuple
    """
    return repr(tup).replace("(", "po_").replace(")", "_pc").replace(", ", "_")


def underscore_str_to_tuple(tup: str) -> Tuple:
    """Convert a a string representation of a tuple to a tuple.

    Args:
        tup (str): a string representing the tuple

    Returns:
        Tuple: a tuple to change into string representation
    """
    return ast.literal_eval(tup.replace("po_", "(").replace("_pc", ")").replace("_", ", "))


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


# pylint: disable-next=too-many-instance-attributes
class RemoteModule(nn.Module):
    """A wrapper class for the modules to be evaluated remotely with FHE."""

    def __init__(
        self,
        module: Optional[nn.Module] = None,
        server_remote_address: Optional[str] = None,
        module_name: Optional[str] = None,
        model_name: Optional[str] = None,
        verbose: int = 0,
    ):
        super().__init__()
        self.private_module: Optional[nn.Module] = module
        self.server_remote_address: Optional[str] = server_remote_address
        self.calibration_data: List = []
        self.uid = str(uuid.uuid4())
        self.private_q_module: Optional[QuantizedModule] = None
        self.fhe_local_mode: HybridFHEMode = HybridFHEMode.CALIBRATE
        self.clients: Dict[str, Tuple[str, FHEModelClient]] = {}
        self.path_to_keys: Optional[Path] = None
        self.path_to_clients: Optional[Path] = None
        self.module_name: Optional[str] = module_name
        self.model_name: Optional[str] = model_name
        self.verbose = verbose

    def init_fhe_client(
        self, path_to_client: Optional[Path] = None, path_to_keys: Optional[Path] = None
    ):  # pragma:no cover
        """Set the clients keys.

        Args:
            path_to_client (str): Path where the client.zip is located.
            path_to_keys (str): Path where keys are located.

        Raises:
            ValueError: if anything goes wrong with the server.
        """
        # Handle paths
        self.path_to_clients = path_to_client
        if self.path_to_clients is None:
            self.path_to_clients = Path() / "clients"
        self.path_to_clients.mkdir(exist_ok=True)
        self.path_to_keys = path_to_keys
        if self.path_to_keys is None:
            self.path_to_keys = Path() / "keys"
        self.path_to_keys.mkdir(exist_ok=True)

        # List all shapes supported by the server
        # This is needed until we have generic shape support in Concrete Python
        assert self.module_name is not None
        shapes_response = requests.get(
            f"{self.server_remote_address}/list_shapes",
            data={"module_name": self.module_name, "model_name": self.model_name},
        )
        if shapes_response.status_code != 200:
            # Add link to request content
            raise ValueError(
                f"Couldn't get shapes from server:\n{shapes_response.content.decode('utf-8')}"
            )

        # For all supported shape we need to get the FHE client from the server
        shapes = shapes_response.json()
        for shape in shapes:
            client_response = requests.get(
                f"{self.server_remote_address}/get_client",
                data={
                    "module_name": self.module_name,
                    "model_name": self.model_name,
                    "input_shape": shape,
                },
            )
            if client_response.status_code != 200:
                # Add link to request content
                raise ValueError(
                    f"Couldn't get client from server:\n{client_response.content.decode('utf-8')}"
                )
            path_to_client = self.path_to_clients / tuple_to_underscore_str(ast.literal_eval(shape))
            path_to_client.mkdir(exist_ok=True)
            with open(path_to_client / "client.zip", "wb") as file:
                file.write(client_response.content)
            # Create the client
            client = FHEModelClient(
                path_dir=str(path_to_client.resolve()), key_dir=str(self.path_to_keys.resolve())
            )
            # The client first need to create the private and evaluation keys.
            serialized_evaluation_keys = client.get_serialized_evaluation_keys()

            if self.verbose:
                print(f"Evaluation keys size: {len(serialized_evaluation_keys) / (10**6):.2f} MB")
            assert isinstance(serialized_evaluation_keys, bytes)
            assert self.module_name is not None
            # Upload the key to the server
            response = requests.post(
                f"{self.server_remote_address}/add_key",
                data={
                    "module_name": self.module_name,
                    "model_name": self.model_name,
                    "input_shape": shape,
                },
                files={"key": io.BytesIO(initial_bytes=serialized_evaluation_keys)},
            )
            assert response.status_code == 200, response.content.decode("utf-8")
            uid = response.json()["uid"]
            # We store the key id and the client in the object
            # If we observe memory issues due to this we can always move
            # towards client lazy loading with caching as done on the server.
            self.clients[shape] = (uid, client)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, QuantTensor]:
        """Forward pass of the remote module.

        To change the behavior of this forward function one must change the fhe_local_mode
        attribute. Choices are:
        - disable: forward using torch module
        - remote: forward with fhe client-server
        - simulate: forward with local fhe simulation
        - calibrate: forward for calibration

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor.

        Raises:
            ValueError: if local_fhe_mode is not supported
        """
        # - disable: torch module
        # - remote: client-server
        # - simulate: compiled simulation
        # - calibrate: calibration

        if self.fhe_local_mode not in {
            HybridFHEMode.DISABLE,
            HybridFHEMode.CALIBRATE,
            HybridFHEMode.REMOTE,
            None,
        }:
            # Using quantized module
            assert self.private_q_module is not None
            y = torch.Tensor(
                self.private_q_module.forward(x.detach().numpy(), fhe=self.fhe_local_mode.value)
            )

        elif self.fhe_local_mode == HybridFHEMode.DISABLE:
            # Calling torch
            assert self.private_module is not None
            y = self.private_module.forward(
                x.detach(),
            )
            assert isinstance(y, (QuantTensor, torch.Tensor))

        elif self.fhe_local_mode == HybridFHEMode.CALIBRATE:
            # Calling torch + gathering calibration data
            assert self.private_module is not None
            self.calibration_data.append(x.detach())
            y = self.private_module(x)
            assert isinstance(y, (QuantTensor, torch.Tensor))

        elif self.fhe_local_mode == HybridFHEMode.REMOTE:  # pragma:no cover
            # Remote call
            y = self.remote_call(x)

        else:  # pragma:no cover
            # Shouldn't happen
            raise ValueError(f"{self.fhe_local_mode} is not recognized")

        return y

    def remote_call(self, x: torch.Tensor) -> torch.Tensor:  # pragma:no cover
        """Call the remote server to get the private module inference.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The result of the FHE computation
        """
        # Store tensor device and move to CPU for FHE encryption
        base_device = x.device
        x = x.to(device="cpu")

        # We need to iterate over elements in the batch since
        # we don't support batch inference
        inferences: List[numpy.ndarray] = []
        for index in range(len(x)):
            # Manage tensor, tensor shape, and encrypt tensor
            clear_input = x[[index], :].detach().numpy()
            input_shape = (1,) + tuple(clear_input.shape)
            repr_input_shape = str(input_shape[1:])
            assert isinstance(clear_input, numpy.ndarray)
            assert repr_input_shape in self.clients
            key_id, client = self.clients[repr_input_shape]
            assert client is not None
            encrypted_input = client.quantize_encrypt_serialize(clear_input)
            assert isinstance(encrypted_input, bytes)
            if self.verbose:
                print(
                    f"Encrypted input size: {sys.getsizeof(encrypted_input) / 1024 / 1024:.2f} MB"
                )
            start = time.time()
            assert self.module_name is not None
            if self.verbose:
                print("Infering ...")
            # Inference using FHE server
            inference_query = requests.post(
                f"{self.server_remote_address}/compute",
                files={
                    "model_input": io.BytesIO(encrypted_input),
                },
                data={
                    "uid": key_id,
                    "module_name": self.module_name,
                    "model_name": self.model_name,
                    "input_shape": repr_input_shape,
                },
                stream=True,
            )
            end = time.time()
            if self.verbose:
                print(f"Inference done in {end - start} seconds")
            # Deserialize and decrypt the result
            assert inference_query.status_code == 200, inference_query.content.decode("utf-8")
            encrypted_result = inference_query.content
            decrypted_prediction = client.deserialize_decrypt_dequantize(encrypted_result)[0]
            inferences.append(decrypted_prediction)

        # Concatenate results and move them back to proper device
        return torch.Tensor(numpy.array(inferences)).to(device=base_device)


class HybridFHEModel:
    """Convert a model to a hybrid model.

    This is done by converting targeted modules by RemoteModules.
    This will modify the model in place.

    Args:
        model (nn.Module): The model to modify (in-place modification)
        module_names (Union[str, List[str]]): The module name(s) to replace with FHE server.
        server_remote_address): The remote address of the FHE server
        model_name (str): Model name identifier
        verbose (int): If logs should be printed when interacting with FHE server
    """

    def __init__(
        self,
        model: nn.Module,
        module_names: Union[str, List[str]],
        server_remote_address=None,
        model_name: str = "model",
        verbose: int = 0,
    ):
        self.model = model
        self.module_names = [module_names] if isinstance(module_names, str) else module_names
        self.server_remote_address = server_remote_address
        self.private_modules: Dict[str, nn.Module] = {
            name: self._get_module_by_name(self.model, name) for name in self.module_names
        }
        self.remote_modules: Dict[str, RemoteModule] = {}
        self.private_q_modules: dict = {}
        self.configuration: Optional[Configuration] = None
        self.model_name = model_name
        self.verbose = verbose
        self._replace_modules()

    def _replace_modules(self):
        """Replace the private modules in the model with remote layers."""

        for module_name in self.module_names:

            # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3858
            # Conv1d introduce reshaping operations which adds more TLU
            self.private_modules[module_name] = convert_conv1d_to_linear(
                self.private_modules[module_name]
            )

            remote_module = RemoteModule(
                module=self.private_modules[module_name],
                server_remote_address=self.server_remote_address,
                module_name=module_name,
                model_name=self.model_name,
                verbose=self.verbose,
            )

            self.remote_modules[module_name] = remote_module

            # Now we need to replace the module in its parent module.
            *path, last = module_name.split(".")
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
            module.fhe_local_mode = HybridFHEMode(fhe)
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
        # FIXME: Shouldn't this search recursively in name modules of name modules?
        for module_name, module in model.named_modules():
            if module_name == name:
                return module
        raise ValueError(f"No module found for name {name} in {list(model.named_modules())}")

    def init_client(
        self, path_to_clients: Optional[Path] = None, path_to_keys: Optional[Path] = None
    ):  # pragma:no cover
        """Initialize client for all remote modules.

        Args:
            path_to_clients (Optional[Path]): Path to the client.zip files.
            path_to_keys (Optional[Path]): Path to the keys folder.
        """
        if path_to_clients is None:
            path_to_clients = Path("clients")
        path_to_clients.mkdir(exist_ok=True)
        for module_name, module in self.remote_modules.items():
            path_to_client = path_to_clients / module_name
            path_to_client.mkdir(exist_ok=True)
            module.init_fhe_client(path_to_client=path_to_client, path_to_keys=path_to_keys)

    def compile_model(
        self,
        x: torch.Tensor,
        n_bits: Union[int, Dict[str, int]] = MAX_BITWIDTH_BACKWARD_COMPATIBLE,
        rounding_threshold_bits: Optional[int] = None,
        p_error: Optional[float] = None,
        configuration: Optional[Configuration] = None,
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
        # We do a forward pass where we accumulate inputs to use for compilation
        for name in self.module_names:
            # default is "calibrate"
            self.remote_modules[name].fhe_local_mode = HybridFHEMode.CALIBRATE
        self.model(x)

        self.configuration = configuration

        for name in self.module_names:
            remote_module = self._get_module_by_name(self.model, name)
            assert isinstance(remote_module, RemoteModule)

            calibration_data_tensor = torch.cat(remote_module.calibration_data, dim=0)

            if has_any_qnn_layers(self.private_modules[name]):
                self.private_q_modules[name] = compile_brevitas_qat_model(
                    self.private_modules[name],
                    calibration_data_tensor,
                    n_bits=n_bits,
                    rounding_threshold_bits=rounding_threshold_bits,
                    configuration=configuration,
                    p_error=p_error,
                )
            else:
                self.private_q_modules[name] = compile_torch_model(
                    self.private_modules[name],
                    calibration_data_tensor,
                    n_bits=n_bits,
                    rounding_threshold_bits=rounding_threshold_bits,
                    configuration=configuration,
                    p_error=p_error,
                )

            self.remote_modules[name].private_q_module = self.private_q_modules[name]

    def _save_fhe_circuit(self, path: Path, via_mlir=False):
        """Private method that saves the FHE circuits.

        Args:
            path (Path): The directory where the FHE circuit will be saved.
            via_mlir (bool): if fhe circuits should be serialized using via_mlir option
                useful for cross-platform (compile on one architecture and run on another)
        """

        model_path = Path(path)
        for module_name in self.module_names:
            input_shapes = [
                tuple(elt.dim_value for elt in onnx_input.type.tensor_type.shape.dim)
                for onnx_input in self.private_q_modules[  # pylint: disable=protected-access
                    self.module_names[0]
                ]._onnx_model.graph.input
            ]
            assert len(input_shapes) == 1, "Multi-input circuits not supported yet"
            model_module_path = model_path.resolve() / module_name
            model_module_path.mkdir(exist_ok=True)
            model_module_shape_path = model_module_path / tuple_to_underscore_str(input_shapes[0])
            model_dev = FHEModelDev(
                str(model_module_shape_path.resolve()),
                self.private_q_modules[module_name],
            )
            model_dev.save(via_mlir=via_mlir)

    def save_and_clear_private_info(self, path: Path, via_mlir=False):
        """Save the PyTorch model to the provided path and also saves the corresponding FHE circuit.

        Args:
            path (Path): The directory where the model and the FHE circuit will be saved.
            via_mlir (bool): if fhe circuits should be serialized using via_mlir option
                useful for cross-platform (compile on one architecture and run on another)
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
        self._save_fhe_circuit(path, via_mlir=via_mlir)

    def publish_to_hub(self):
        """Allow the user to push the model and FHE required files to HF Hub."""
        # FIXME: implement HuggingFace model hub integration

    def set_fhe_mode(self, hybrid_fhe_mode: Union[str, HybridFHEMode]):
        """Set Hybrid FHE mode for all remote modules.

        Args:
            hybrid_fhe_mode (Union[str, HybridFHEMode]): Hybrid FHE mode to set to all
                remote modules.
        """
        for module in self.remote_modules.values():
            module.fhe_local_mode = HybridFHEMode(hybrid_fhe_mode)


class LoggerStub:  # pragma:no cover
    """Placeholder type for a typical logger like the one from loguru."""

    @abstractmethod
    def info(self, msg: str):
        """Placholder function for logger.info.

        Args:
            msg (str): the message to output
        """


@lru_cache(maxsize=None)  # noqa: B019
def _load_key(key_path: Path, uid: Union[str, uuid.UUID]) -> bytes:  # pragma:no cover
    """Load a public key from the file system.

    Args:
        key_path (Path): key path
        uid (Union[str, uuid.UUID]): uid of the public key to load

    Returns:
        bytes: the bytes of the public key
    """
    with open(key_path / str(uid), "rb") as file:
        return file.read()


@lru_cache(maxsize=None)  # noqa: B019,W1517
def _get_circuit(path: str) -> FHEModelServer:  # pragma:no cover
    """Get circuit based on model name, module name and input shape.

    Args:
        path (str): path to the model server

    Returns:
        FHEModelServer: a fhe model server of the given module of the given model
            for the given shape

    """
    return FHEModelServer(path)


class HybridFHEModelServer:  # pragma:no cover
    """Hybrid FHE Model Server.

    This is a class object to server FHE models serialized using HybridFHEModel.
    """

    def __init__(self, key_path: Path, model_dir: Path, logger: Optional[LoggerStub]):
        self.logger = logger
        self.key_path = key_path
        self.key_path.mkdir(exist_ok=True)
        self.model_dir = model_dir
        self.modules: Dict[str, Dict[str, Dict[str, Dict]]] = defaultdict(dict)

        # Populate modules at the beginning
        # this could also be done dynamically on each query if needed
        # We build the following mapping:
        # model_name -> module_name -> input_shape -> some information
        for model_path in self.model_dir.iterdir():  # Model
            if not model_path.is_dir():
                continue
            model_name = model_path.name
            self.modules[model_name] = defaultdict(dict)
            for module_path in model_path.iterdir():  # Module
                if not module_path.is_dir():
                    continue
                module_name = module_path.name
                self.modules[model_name][module_name] = defaultdict(dict)
                for input_shape_path in module_path.iterdir():
                    if not input_shape_path.is_dir():
                        continue
                    input_shape = str(underscore_str_to_tuple(input_shape_path.name))
                    self.modules[model_name][module_name][input_shape] = {
                        "path": input_shape_path.resolve(),
                        "module_name": module_name,
                        "model_name": model_name,
                        "shape": input_shape,
                    }

    def load_key(self, uid: Union[str, uuid.UUID]) -> bytes:
        """Load a public key from the key path in the file system.

        Args:
            uid (Union[str, uuid.UUID]): uid of the public key to load

        Returns:
            bytes: the bytes of the public key
        """
        return _load_key(self.key_path, uid)

    def dump_key(self, key_bytes: bytes, uid: Union[uuid.UUID, str]) -> None:
        """Dump a public key to a stream.

        Args:
            key_bytes (bytes): stream to dump the public serialized key to
            uid (Union[str, uuid.UUID]): uid of the public key to dump
        """
        with open(self.key_path / str(uid), "wb") as file:
            file.write(key_bytes)

    def get_circuit(self, model_name, module_name, input_shape):
        """Get circuit based on model name, module name and input shape.

        Args:
            model_name (str): name of the model
            module_name (str): name of the module in the model
            input_shape (str): input shape of the module

        Returns:
            FHEModelServer: a fhe model server of the given module of the given model
                for the given shape

        """
        path = str(self.modules[model_name][module_name][input_shape]["path"])
        return _get_circuit(path)

    def check_inputs(self, model_name: str, module_name: Optional[str], input_shape: Optional[str]):
        """Check that the given configuration exist in the compiled models folder.

        Args:
            model_name (str): name of the model
            module_name (Optional[str]): name of the module in the model
            input_shape (Optional[str]): input shape of the module

        Raises:
            ValueError: if the given configuration does not exist.
        """
        if model_name not in self.modules:
            raise ValueError(
                f"provided names '{model_name}' does not match any known name",
            )
        if module_name is not None and module_name not in self.modules[model_name]:
            raise ValueError(
                f"provided names '{module_name}' does not match any known name"
                f"{list(self.modules[model_name].keys())}",
            )
        if (
            model_name is not None
            and module_name is not None
            and input_shape is not None
            and input_shape not in self.modules[model_name][module_name]
        ):
            raise ValueError(
                f"provided names '{module_name}' does not match any known name"
                f"{list(self.modules[model_name][module_name].keys())}",
            )

    def list_modules(self, model_name: str):
        """List all modules in a model.

        Args:
            model_name (str): name of the model

        Returns:
            Dict[str, Dict[str, Dict]]
        """
        self.check_inputs(model_name, None, None)
        return self.modules[model_name]

    def list_shapes(self, model_name: str, module_name: str):
        """List all modules in a model.

        Args:
            model_name (str): name of the model
            module_name (str): name of the module in the model

        Returns:
            Dict[str, Dict]
        """
        self.check_inputs(model_name, module_name, None)
        return self.modules[model_name][module_name]

    def get_client(self, model_name: str, module_name: str, input_shape: str):
        """Get client.

        Args:
            model_name (str): name of the model
            module_name (str): name of the module in the model
            input_shape (str): input shape of the module

        Returns:
            Path: the path to the correct client

        Raises:
            ValueError: if client couldn't be found
        """
        self.check_inputs(model_name, module_name, input_shape)
        path_to_client = (
            self.modules[model_name][module_name][str(input_shape)]["path"] / "client.zip"
        ).resolve()
        if not path_to_client.exists():
            raise ValueError("Could not find client.")
        return path_to_client

    def add_key(
        self,
        key: bytes,
        model_name: str,
        module_name: str,
        input_shape: str,
    ):
        """Add public key.

        Arguments:
            key (bytes): public key
            model_name (str): model name
            module_name (str): name of the module in the model
            input_shape (str): input shape of said module

        Returns:
            Dict[str, str]
                - uid: uid a personal uid
        """
        self.check_inputs(model_name, module_name, input_shape)
        uid = str(uuid.uuid4())
        self.dump_key(key, uid)
        return {"uid": uid}

    def compute(
        self,
        model_input: bytes,
        uid: str,
        model_name: str,
        module_name: str,
        input_shape: str,
    ):  # noqa: B008
        """Compute the circuit over encrypted input.

        Arguments:
            model_input (bytes): input of the circuit
            uid (str): uid of the public key to use
            model_name (str): model name
            module_name (str): name of the module in the model
            input_shape (str): input shape of said module

        Returns:
            bytes: the result of the circuit
        """
        self.check_inputs(model_name, module_name, input_shape)
        start = time.time()
        key_bytes = self.load_key(uid)
        end = time.time()
        if self.logger is not None:
            self.logger.info(f"It took {end - start} seconds to load the key")

        start = time.time()
        fhe = self.get_circuit(model_name, module_name, input_shape)
        end = time.time()
        if self.logger is not None:
            self.logger.info(f"It took {end - start} seconds to load the circuit")

        start = time.time()
        encrypted_results = fhe.run(
            serialized_encrypted_quantized_data=model_input,
            serialized_evaluation_keys=key_bytes,
        )
        end = time.time()

        if self.logger is not None:
            self.logger.info(f"fhe inference of input of shape {input_shape} took {end - start}")
            self.logger.info(f"Results size is {len(encrypted_results)/(1024**2)} Mb")
        start = time.time()
        return encrypted_results
