"""Implement the conversion of a torch model to a hybrid fhe/torch inference."""

# pylint: disable=too-many-lines,too-many-locals
import ast
import csv
import io
import json
import logging
import sys
import time
import uuid
from abc import abstractmethod
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import concrete_ml_extensions as fhext
import numpy
import requests
import torch
from brevitas.quant_tensor import QuantTensor
from concrete.fhe import Configuration
from torch import nn
from tqdm.autonotebook import tqdm

from ..common.utils import MAX_BITWIDTH_BACKWARD_COMPATIBLE, HybridFHEMode
from ..deployment.fhe_client_server import FHEModelClient, FHEModelDev
from ..quantization.linear_op_glwe_backend import (
    GLWELinearLayerExecutor,
    _add_bias,
    _apply_correction_and_dequantize,
    _dynamic_input_quantization,
    has_glwe_backend,
)
from .compile import (
    QuantizedModule,
    build_quantized_module,
    compile_brevitas_qat_model,
    compile_torch_model,
    has_any_qnn_layers,
)
from .hybrid_backprop_linear import BackwardModuleLinear, ForwardModuleLinear


def tuple_to_underscore_str(tup: Tuple) -> str:
    """Convert a tuple to a string representation.

    Examples:
        >>> tuple_to_underscore_str((1, 2, 3))
        'po_1_2_3_pc'

    Args:
        tup (Tuple): a tuple to change into string representation

    Returns:
        str: a string representing the tuple
    """
    return repr(tup).replace("(", "po_").replace(")", "_pc").replace(", ", "_")


def underscore_str_to_tuple(tup: str) -> Tuple:
    """Convert a a string representation of a tuple to a tuple.

    Examples:
        >>> underscore_str_to_tuple("po_1_2_3_pc")
        (1, 2, 3)

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
    try:
        from transformers.modeling_utils import Conv1D  # pylint: disable=import-outside-toplevel
    except ImportError:  # pragma: no cover
        return layer_or_module

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


def fetch_remote_weights(
    layer_dir: Union[str, Path],
    filename_weight_format: str = "remote_weights",
    filename_weight_extension: str = "npy",
) -> Path:
    """Fetch and return the unique remote weight file in the given directory.

    This function searches for a file in `layer_dir` whose name matches the pattern
    `<filename_weight_format>*.<filename_weight_extension>`.

    Args:
        layer_dir (Union[str, Path]): Path to the directory where the weight file is stored.
        filename_weight_format (str): Prefix pattern used to identify the weight file.
                                                Defaults to "remote_weights".
        filename_weight_extension (str): File extension of the weight file.
                                                   Defaults to "npy".

    Returns:
        (Path): The path to the unique weight file that matches the pattern.

    Raises:
        FileNotFoundError: If no weight file matches the pattern in the directory.
        RuntimeError: If multiple files match the pattern, leading to ambiguity.
    """
    layer_dir = Path(layer_dir)

    pattern = f"{filename_weight_format}*.{filename_weight_extension}"
    candidates = list(layer_dir.glob(pattern))

    if not candidates:
        raise FileNotFoundError(f"No weight file matching pattern '{pattern}' in `{layer_dir}`")

    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple weight files matching pattern '{pattern}' in `{layer_dir}`: "
            f"{[str(p) for p in candidates]}"
        )

    return candidates[0]


class BenchmarkLogger:
    """A simple class to save timing and metadata into a CSV file."""

    def __init__(
        self,
        file_path: Path,
        columns: List[str],
        delimiter: str = ";",
        logger=None,
        reset: bool = False,
    ):
        """Initialize the BenchmarkLogger.

        Args:
            file_path (Path): Path to the CSV file.
            columns (List[str]): The column headers.
            delimiter (str): CSV delimiter.
            logger: Optional logger to use for messages.
            reset (bool): If True, overwrite the file.
        """
        self.file_path = file_path
        self.columns = columns
        self.delimiter = delimiter
        self.logger = logger

        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        if reset and self.file_path.exists():
            self.file_path.unlink()
            msg = "Benchmark file reset"

        elif not self.file_path.exists():
            with self.file_path.open("w", newline="") as csvfile:
                writer = csv.writer(csvfile, delimiter=self.delimiter)
                writer.writerow(self.columns)
            msg = "Benchmark file created"
        else:
            msg = "Benchmark file already created"
        if self.logger:
            self.logger.info("%s: '%s'", msg, self.file_path.resolve())

    def append(self, data: Dict):
        """Append a row to the CSV file.

        Args:
            data (Dict): A dict with keys matching the columns.

        Raises:
            ValueError: If `data` contains keys that are not present in `self.columns`.
        """
        invalid_keys = set(data.keys()) - set(self.columns)
        if invalid_keys:
            raise ValueError(
                f"Invalid keys in benchmark data: {invalid_keys}\n" f"Allowed keys: {self.columns}"
            )

        row = [data.get(col, "") for col in self.columns]

        with self.file_path.open("a", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=self.delimiter)
            writer.writerow(row)

        if self.logger:
            self.logger.debug("Benchmark row added: %s", row)


# pylint: disable-next=too-many-instance-attributes
class RemoteModule(nn.Module):
    """A wrapper class for the modules to be evaluated remotely with FHE.

    Args:
        module (Optional[nn.Module]): The PyTorch module to be wrapped.
        server_remote_address (Optional[str]): The URL of the remote server.
        module_name (Optional[str]): Name of the module used by the server.
        model_name (Optional[str]): Name of the model used by the server.
        optimized_linear_execution (bool): Whether to use an optimized GLWE.

        The server must has the following endpoints:
        - `/list_shapes`: List all shapes supported by the server.
        - `/get_client`: Get the FHE client for a specific shape.
        - `/add_key`: Add the evaluation key for a specific shape.
        - `/compute`: Perform the FHE computation with the encrypted input.

    """

    def __init__(
        self,
        module: Optional[nn.Module] = None,
        server_remote_address: Optional[str] = None,
        module_name: Optional[str] = None,
        model_name: Optional[str] = None,
        optimized_linear_execution: bool = False,
        benchmark_logger: Optional[BenchmarkLogger] = None,
        logger: Optional[logging.Logger] = None,
        logger_level=logging.INFO,
        machine_type: str = "",
    ):
        super().__init__()

        self.private_module: Optional[nn.Module] = module
        self.server_remote_address: Optional[str] = server_remote_address
        self.calibration_data: Optional[List] = []
        self.uid = str(uuid.uuid4())
        self.private_q_module: Optional[QuantizedModule] = None
        self.fhe_local_mode: HybridFHEMode = HybridFHEMode.CALIBRATE
        self.clients: Dict[str, Tuple[str, FHEModelClient]] = {}
        self.path_to_keys: Optional[Path] = None
        self.path_to_clients: Optional[Path] = None
        self.module_name: Optional[str] = module_name
        self.model_name: Optional[str] = model_name
        self.optimized_linear_execution = optimized_linear_execution
        self.executor: Optional[GLWELinearLayerExecutor] = None
        self.progress_callback: Optional[Callable[[], None]] = None
        self.private_remote_weights_path: Optional[Path] = None
        self.machine_type = machine_type
        self.benchmark_logger = benchmark_logger
        self.logger = logger or logging.getLogger(f"HybridFHEModel.RemoteModule.{module_name}")
        self.logger.setLevel(logger_level)

    def init_fhe_client(
        self,
        path_to_client: Optional[Path] = None,
        path_to_keys: Optional[Path] = None,
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
        shapes = [tuple(shape) for shape in shapes]
        for shape in shapes:
            client_response = requests.get(
                f"{self.server_remote_address}/get_client",
                data={
                    "module_name": self.module_name,
                    "model_name": self.model_name,
                    "input_shape": str(shape),
                },
            )
            if client_response.status_code != 200:
                # Add link to request content
                raise ValueError(
                    f"Couldn't get client from server:\n{client_response.content.decode('utf-8')}"
                )

            path_to_client = self.path_to_clients / tuple_to_underscore_str(shape)
            path_to_client.mkdir(exist_ok=True)
            # with open(path_to_client / "client.zip", "wb") as file:
            #     file.write(client_response.content)
            # Create the client
            client = FHEModelClient(
                path_dir=str(path_to_client.resolve()), key_dir=str(self.path_to_keys.resolve())
            )
            # The client first need to create the private and evaluation keys.
            serialized_evaluation_keys = client.get_serialized_evaluation_keys()

            if self.logger:
                self.logger.info(
                    "Evaluation keys size: '%.2f' MB", len(serialized_evaluation_keys) / 1e6
                )

            assert isinstance(serialized_evaluation_keys, bytes)
            assert self.module_name is not None
            # Upload the key to the server
            response = requests.post(
                f"{self.server_remote_address}/add_key",
                data={
                    "module_name": self.module_name,
                    "model_name": self.model_name,
                    "input_shape": str(shape),
                },
                files={"key": io.BytesIO(initial_bytes=serialized_evaluation_keys)},
            )
            assert (
                response.status_code == 200
            ), f'Got code=`{response.status_code}` - {response.content.decode("utf-8")}'

            uid = response.json()["uid"]
            # We store the key id and the client in the object
            # If we observe memory issues due to this we can always move
            # towards client lazy loading with caching as done on the server.
            self.clients[str(shape)] = (uid, client)

    def _apply(self, fn, recurse=True):
        """Prevent remote modules moving private debug weights to GPU.

        .. # noqa: DAR101
        .. # noqa: DAR201

        """
        return self

    def _ensure_module_on_device(self, x: torch.Tensor) -> None:
        """Ensure the private module is on the same device as the input tensor.

        Args:
            x (torch.Tensor): The input tensor to match device with.
        """
        assert self.private_module is not None

        # Check if any parameter is not on the same device as the input tensor
        if any(
            param.device != x.device for param in self.private_module.parameters()
        ):  # pragma: no cover
            self.private_module = self.private_module.to(x.device)  # pragma: no cover

    def forward(
        self, x: torch.Tensor, device: Union[str, torch.device] = "cpu"
    ) -> Union[torch.Tensor, QuantTensor]:
        """Forward pass of the remote module.

        To change the behavior of this forward function one must change the fhe_local_mode
        attribute. Choices are:
        - disable: forward using torch module
        - remote: forward with fhe client-server
        - simulate: forward with local fhe simulation
        - calibrate: forward for calibration

        Args:
            x (torch.Tensor): The input tensor.
            device (Union[str, torch.device]): The target device.

        Returns:
            (torch.Tensor): The output tensor.

        Raises:
            ValueError: if local_fhe_mode is not supported
        """
        # - disable: quantized module
        # - remote: client-server
        # - simulate: compiled simulation
        # - calibrate: calibration

        if self.fhe_local_mode not in {
            HybridFHEMode.CALIBRATE,
            HybridFHEMode.REMOTE,
            HybridFHEMode.TORCH,
            None,
        }:
            assert self.private_q_module is not None

            if self.executor:
                # Delegate to the optimized GLWE executor
                y = self.executor.forward(x.detach(), self.private_q_module, self.fhe_local_mode)
            else:
                device = x.device
                # Delegate to the quantized module for all fhe modes
                y = torch.Tensor(
                    self.private_q_module.forward(
                        x.cpu().detach().numpy(), fhe=self.fhe_local_mode.value
                    )
                ).to(device)

        elif self.fhe_local_mode == HybridFHEMode.CALIBRATE:
            # Calling torch + gathering calibration data
            assert self.private_module is not None
            assert self.calibration_data is not None
            self.calibration_data.append(x.detach())
            self._ensure_module_on_device(x)
            y = self.private_module(x)
            assert isinstance(y, (QuantTensor, torch.Tensor))

        elif self.fhe_local_mode == HybridFHEMode.REMOTE:  # pragma:no cover
            if self.executor:
                y = self.remote_glwe_call(x, x.device)
            else:
                y = self.remote_call(x, x.device)
        elif self.fhe_local_mode == HybridFHEMode.TORCH:
            # Using torch layers
            assert self.private_module is not None
            # Move private module parameters to same device as input if needed
            self._ensure_module_on_device(x)
            y = self.private_module(x)
        else:  # pragma:no cover
            # Shouldn't happen
            raise ValueError(f"{self.fhe_local_mode} is not recognized")

        # Call progress callback if set
        if self.progress_callback is not None:
            self.progress_callback()

        return y

    def remote_call(
        self, x: torch.Tensor, device: Union[str, torch.device] = "cpu"
    ) -> torch.Tensor:  # pragma:no cover
        """Call the remote server to get the private module inference.

        Args:
            x (torch.Tensor): The input tensor.
            device (Union[str, torch.device]): The device.

        Returns:
            torch.Tensor: The result of the FHE computation
        """
        # Store tensor device and move to CPU for FHE encryption
        x = x.to(device=device)

        # We need to iterate over elements in the batch since
        # we don't support batch inference
        inferences: List[numpy.ndarray] = []

        for index in range(len(x)):
            # Manage tensor, tensor shape, and encrypt tensor
            clear_input = x[[index], :].detach().numpy()
            input_shape = (1,) + tuple(clear_input.shape)
            repr_input_shape = str(input_shape[1:])
            assert isinstance(clear_input, numpy.ndarray)
            assert repr_input_shape in self.clients, (
                f"Client with input shape `{repr_input_shape}` not found in `self.clients`. "
                f"Available keys: `{list(self.clients.keys())}`. "
            )
            key_id, client = self.clients[repr_input_shape]
            assert client is not None
            encrypted_input = client.quantize_encrypt_serialize(clear_input)
            assert isinstance(encrypted_input, bytes)

            if self.logger:
                self.logger.info(
                    "Encrypted input size: '%.2f' MB", sys.getsizeof(encrypted_input) / 1024 / 1024
                )
            assert self.module_name is not None

            # Inference using FHE server
            start = time.time()
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

            if self.logger:
                self.logger.info("Inference done in '%.2f' s.", time.time() - start)

            # Deserialize and decrypt the result
            assert inference_query.status_code == 200, inference_query.content.decode("utf-8")
            encrypted_result = inference_query.content
            decrypted_prediction = client.deserialize_decrypt_dequantize(encrypted_result)[0]
            inferences.append(decrypted_prediction)

        # Concatenate results and move them back to proper device
        return torch.Tensor(numpy.array(inferences)).to(device=device)

    # pylint: disable=too-many-statements
    def remote_glwe_call(
        self, x: torch.Tensor, device: Union[str, torch.device] = "cpu"
    ) -> torch.Tensor:  # pragma:no cover
        """Call the remote server to get the private module inference.

        Args:
            x (torch.Tensor): The input tensor.
            device (Union[str, torch.device]): The device.

        Returns:
            torch.Tensor: The result of the FHE computation
        """

        start = time.time()
        # Store tensor device and move to CPU for FHE encryption
        x = x.to(device=device)
        inferences: List[torch.Tensor] = []

        # Iterate over each element in the batch
        # x.shape -> 1, 64, 2048 (batch_size, sequence_length (nb token), hidden_size)
        for index in range(len(x)):
            clear_input = x[[index], :]

            # Dynamic input quantization
            x_q, x_scale, x_zp, original_shape = _dynamic_input_quantization(clear_input, n_bits=7)
            # Convert quantized data to numpy arrays for encryption.
            x_q_int = x_q.long().cpu().numpy().astype(numpy.int64).astype(numpy.uint64)

            assert self.private_remote_weights_path is not None
            path_client = Path(self.private_remote_weights_path).parent / "client"
            path_client.mkdir(parents=True, exist_ok=True)
            numpy.save(path_client / "quantized_input.npy", x_q_int)

            # Encrypt the input
            assert self.executor is not None, "Executor must be initialized for GLWE execution."
            assert self.executor.private_key is not None
            s = time.time()
            ciphertext = fhext.encrypt_matrix(  # pylint: disable=no-member
                pkey=self.executor.private_key,
                crypto_params=self.executor.glwe_crypto_params,
                data=x_q_int,
            )
            time_encryption_input = time.time() - s

            s = time.time()
            ciphertext_serialized = ciphertext.serialize()
            time_serialization_input = time.time() - s

            # if self.logger:
            #     self.logger.info("Starting inference for module name: '%s'...", self.module_name)

            output_path = f"{self.private_remote_weights_path}/encrypted_output_from_server.bin"

            s = time.time()
            response = requests.post(
                url=f"{self.server_remote_address}/compute",
                files={
                    "encrypted_input": io.BytesIO(ciphertext_serialized),
                },
                data={
                    "uid": str(self.uid),
                    "shape": x_q_int.shape,
                    "linear_layer_name_path": str(self.private_remote_weights_path),
                },
                stream=True,
            )
            total_compute_func = time.time() - s

            assert response.status_code == 200

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=4096):
                    if chunk:
                        f.write(chunk)

            bundle = numpy.load(output_path)
            encrypted_output = bundle["encrypted_output"].tobytes()
            weight_scale = torch.tensor(bundle["weight_scale"], dtype=torch.float32, device=device)
            weight_zp = torch.tensor(bundle["weight_zp"], dtype=torch.float32, device=device)
            sum_w = torch.tensor(bundle["sum_w"], dtype=torch.float32, device=device)
            weight_shape = tuple(bundle["weight_shape"])
            bias = torch.tensor(bundle["bias"], device=device) if "bias" in bundle else None
            input_n_bits = int(bundle["input_n_bits"])

            assert input_n_bits == 7, "Only 7-bits is supported."

            num_valid_glwe_values_in_last_ciphertext = (
                weight_shape[1] % self.executor.poly_size or self.executor.poly_size
            )
            s = time.time()
            # pylint: disable=no-member
            encrypted_deserialized_output = fhext.CompressedResultEncryptedMatrix.deserialize(
                encrypted_output
            )
            time_deserialization_output = time.time() - s

            s = time.time()
            decrypted_deserialize_output = fhext.decrypt_matrix(  # pylint: disable=no-member
                encrypted_deserialized_output,
                self.executor.private_key,
                self.executor.glwe_crypto_params,
                num_valid_glwe_values_in_last_ciphertext,
            ).astype(numpy.int64)

            time_decryption_output = time.time() - s

            q_result = decrypted_deserialize_output
            result_tensor = torch.tensor(q_result, device=device, dtype=torch.long)

            s = time.time()
            out_tensor = _apply_correction_and_dequantize(
                result_tensor,
                x_q,
                x_zp,
                weight_zp,
                sum_w,
                weight_shape[0],
                x_scale,
                weight_scale,
            )
            time_dequantization_output = time.time() - s
            out_tensor = (
                out_tensor.view(*original_shape[:-1], -1) if original_shape[:-1] else out_tensor
            )
            assert (
                original_shape[:-1] == out_tensor.shape[:-1]
            ), "Original shape and output shape do not match"

            out_tensor = _add_bias(out_tensor, bias, device)
            inferences.append(out_tensor)

            total_timing = time.time() - start

            assert self.benchmark_logger is not None

            self.benchmark_logger.append(
                {
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "device": str(device),
                    "machine": self.machine_type,
                    "mode": "Remote",
                    "uid": self.uid,
                    "server_remote_address": self.server_remote_address,
                    "layer_name": str(
                        fetch_remote_weights(self.private_remote_weights_path)
                    ).rsplit("/", maxsplit=1)[-1],
                    "input_shape": str(x_q_int.shape),
                    "remote_weight_shape": str(weight_shape),
                    "time_encryption_input": time_encryption_input,
                    "time_serialization_input": time_serialization_input,
                    "time_deserialization_output": time_deserialization_output,
                    "time_decryption_output": time_decryption_output,
                    "time_dequantization_output": time_dequantization_output,
                    "total_compute_func": total_compute_func,
                    "total_timing": total_timing,
                }
            )

        return torch.stack(inferences, dim=0).to(device)[0]


# pylint: disable-next=too-many-instance-attributes
class HybridFHEModel:
    """Convert a model to a hybrid model.

    This is done by converting targeted modules by RemoteModules.
    This will modify the model in place.

    Args:
        model (nn.Module): The model to modify (in-place modification).
        module_names (Union[str, List[str]]): The module name(s) to replace with FHE server.
        server_remote_address (str): The remote address of the FHE server.
        model_name (str): Model name identifier.
        optimized_linear_execution (bool): Whether to enable the GLWE backend.
        Defaults to True. Enabling this feature is highly recommended for optimal performance.

    Raises:
        TypeError: If the provided model is not an instance of torch.nn.Module.
    """

    def __init__(
        self,
        model: nn.Module,
        module_names: Union[str, List[str]],
        server_remote_address: Optional[str] = None,
        model_name: str = "model",
        optimized_linear_execution: bool = True,
        machine_type: str = "",
        logger: Optional[logging.Logger] = None,
        benchmark_file = None,
    ):
        if not isinstance(model, torch.nn.Module):
            raise TypeError("The model must be a PyTorch or Brevitas model.")

        self.logger = logger or logging.getLogger("HybridFHEModel")
        benchmark_file = Path(benchmark_file or "client_benchmarks.csv")
        self.benchmark_logger = BenchmarkLogger(
            file_path=benchmark_file,
            columns=[
                "date",
                "device",
                "machine",
                "mode",
                "uid",
                "server_remote_address",
                "layer_name",
                "input_shape",
                "remote_weight_shape",
                "time_encryption_input",
                "time_serialization_input",
                "time_deserialization_output",
                "time_decryption_output",
                "time_dequantization_output",
                "total_compute_func",
                "total_timing",
            ],
            logger=self.logger,
            reset=False,
        )

        self.optimized_linear_execution = optimized_linear_execution
        self.model = model
        self.module_names = [module_names] if isinstance(module_names, str) else module_names
        self.server_remote_address = server_remote_address
        self.private_modules: Dict[str, nn.Module] = {
            name: self._get_module_by_name(self.model, name) for name in self.module_names
        }
        self.remote_modules: Dict[str, RemoteModule] = {}
        self.private_q_modules: Dict[str, QuantizedModule] = {}
        self.configuration: Optional[Configuration] = None
        self.model_name = model_name
        self.executor: Optional[GLWELinearLayerExecutor] = None
        self.machine_type = machine_type
        self.client_model_state_dict = None

        self._replace_modules()

    def _replace_modules(self):
        """Replace the private modules in the model with remote layers."""
        self._has_only_large_linear_layers = True
        for module_name in self.module_names:
            # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3858
            # Conv1d introduce reshaping operations which adds more TLU
            self.private_modules[module_name] = convert_conv1d_to_linear(
                self.private_modules[module_name]
            )

            # Determine if this remote module is a pure linear one
            # that is supported for compressed encrypted matmul
            # Conv1D will have been converted to Linear by the line above
            is_pure_linear_layer = isinstance(
                self.private_modules[module_name],
                (nn.Linear, ForwardModuleLinear, BackwardModuleLinear),
            )

            # Check input dimensions for linear layers
            # If the input dimension is less than 512 we do not use the GLWE optimization.
            # Optimal input dimension is 2048, below 512 the performance are too low.
            if is_pure_linear_layer:
                module = self.private_modules[module_name]
                # Use weight shape instead of in/out_features
                input_dim, output_dim = (
                    (
                        module.weight.shape[1],
                        module.weight.shape[0],
                    )
                    if hasattr(module, "weight")
                    else (0, 0)
                )

                is_pure_linear_layer = (
                    is_pure_linear_layer and input_dim >= 512 and output_dim >= 512
                )

            if not is_pure_linear_layer:
                self._has_only_large_linear_layers = False

        for module_name in self.module_names:
            # Create the optimized glwe linear layer executor if needed
            remote_module = RemoteModule(
                module=self.private_modules[module_name],
                server_remote_address=self.server_remote_address,
                module_name=module_name,
                model_name=self.model_name,
                optimized_linear_execution=self._has_only_large_linear_layers,
                benchmark_logger=self.benchmark_logger,
                logger=self.logger,
                machine_type=self.machine_type,
            )

            self.remote_modules[module_name] = remote_module

            # Now we need to replace the module in its parent module.
            *path, last = module_name.split(".")
            parent_module = (
                self._get_module_by_name(self.model, ".".join(path)) if path else self.model
            )
            setattr(parent_module, last, remote_module)

    def forward(self, x: torch.Tensor, fhe: str = "disable") -> torch.Tensor:
        """Forward pass of the hybrid model.

        Args:
            x (torch.Tensor): The input tensor.
            fhe (str): The Fully Homomorphic Encryption (FHE) mode (default is "disable").

        Returns:
            torch.Tensor: The output tensor.

        Raises:
            AssertionError: if the execution mode is not supported
        """
        self.set_fhe_mode(fhe)

        # Validate the FHE mode
        fhe_mode = HybridFHEMode(fhe)

        if (
            has_glwe_backend()
            and self._has_only_large_linear_layers
            and self.optimized_linear_execution
        ):
            if fhe_mode == HybridFHEMode.SIMULATE:
                raise AssertionError(
                    "When the HybridFHEModel is instantiated with only "
                    "linear remote layers, fhe=simulate is not supported for now.",
                )

            if fhe_mode in (HybridFHEMode.EXECUTE, HybridFHEMode.REMOTE, HybridFHEMode.DISABLE):
                # Initialize executor only if not already done
                self.executor = self.executor or GLWELinearLayerExecutor()

                # Generate keys only if needed and not already done
                if fhe_mode != HybridFHEMode.DISABLE and self.executor.private_key is None:
                    self.executor.keygen()

        # Update executor for all remote modules
        for module in self.remote_modules.values():
            module.executor = self.executor

        # Show progress bar for execute mode
        if fhe_mode == HybridFHEMode.EXECUTE:
            # Initialize the progress bar based on the number of remote modules
            num_remote_modules = len(self.remote_modules)
            with tqdm(total=num_remote_modules, desc="FHE Modules Inference") as pbar:
                # Set each remote module's progress_callback to update the progress bar
                for remote_module in self.remote_modules.values():
                    remote_module.progress_callback = lambda: pbar.update(1)

                # Call the model forward pass which, in turn, will trigger each remote module
                result = self.model(x)
        else:
            # For other modes, just run the model without progress tracking
            for remote_module in self.remote_modules.values():
                remote_module.progress_callback = None
            result = self.model(x)

        return result

    def __call__(self, x: torch.Tensor, fhe: str = "disable") -> torch.Tensor:
        """Call method to run the model locally with a fhe mode.

        Args:
            x (torch.Tensor): The input tensor.
            fhe (str): The Fully Homomorphic Encryption (FHE) mode (default is "disable").

        Returns:
            (torch.Tensor): The output tensor.
        """
        return self.forward(x, fhe)

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

        if self.executor is not None:
            self.client_model_state_dict = torch.load(Path("client") / "client_model.pth")
            self.executor = self.executor or GLWELinearLayerExecutor()
            if self.executor.private_key is None:
                self.logger.info("Generating keys...")
                self.executor.keygen()
                self.logger.info("Keys generated...")
                ckey = self.executor.compression_key
                assert ckey is not None
                assert hasattr(ckey, "serialize")
                serialized_ckey = ckey.serialize()
                assert isinstance(serialized_ckey, bytes)
                # Save the keys
                with (path_to_clients / "public_evaluation_key.serverKey").open(
                    "wb"
                ) as binary_file:
                    binary_file.write(serialized_ckey)
                self.logger.info("Saving the public evaluation key at %s...", (path_to_clients / "public_evaluation_key.serverKey"))

                response = requests.post(
                    f"{self.server_remote_address}/add_key",
                    files={"key": ("key", io.BytesIO(serialized_ckey))},
                )
                assert (
                    response.status_code == 200
                ), f"Request failed: `{response.status_code}`\nResponse:\n`{response.text}`"
                self.logger.info("The key has been sent to the server...")

                uid = response.json()["uid"]
                for _, module in self.remote_modules.items():
                    module.uid = uid
        else:
            for module_name, module in self.remote_modules.items():
                path_to_client = path_to_clients / module_name
                path_to_client.mkdir(exist_ok=True)
                module.init_fhe_client(path_to_client=path_to_client, path_to_keys=path_to_keys)

        self.logger.info("All remote modules initialized.")

    def compile_model(
        self,
        x: torch.Tensor,
        n_bits: Union[int, Dict[str, int]] = MAX_BITWIDTH_BACKWARD_COMPATIBLE,
        rounding_threshold_bits: Optional[int] = None,
        p_error: Optional[float] = None,
        device: str = "cpu",
        configuration: Optional[Configuration] = None,
        use_dynamic_quantization: bool = False,
    ):
        """Compiles the specific layers to FHE.

        Args:
            x (torch.Tensor): The input tensor for the model. This is used to run the model
                once for calibration.
            n_bits (int): The bit precision for quantization during FHE model compilation.
                Default is 8.
            rounding_threshold_bits (Optional[int]): The number of bits to use for rounding
                threshold during FHE model compilation. Default to None.
            p_error (Optional[float]): Error allowed for each table look-up in the circuit.
            device (str): FHE compilation device, can be either 'cpu' or 'cuda'.
            configuration (Optional[Configuration]): A concrete Configuration object specifying the
                FHE encryption parameters. If not specified, a default configuration is used.
            use_dynamic_quantization (bool): If True, use dynamic quantization;
                otherwise, use static quantization. (only for GLWE backend)
        """

        assert (
            has_glwe_backend() or not use_dynamic_quantization
        ), "Dynamic quantization requires GLWE backend"

        # We do a forward pass where we accumulate inputs to use for compilation
        self.set_fhe_mode(HybridFHEMode.CALIBRATE)

        # Set correct device
        if isinstance(x, torch.Tensor):
            x = x.to(device)
        self.model = self.model.to(device)

        # Run the model to get the calibration data
        self.model(x)

        self.configuration = configuration

        for name in tqdm(self.module_names, desc="Compiling FHE layers"):

            remote_module = self._get_module_by_name(self.model, name)
            assert isinstance(remote_module, RemoteModule)

            assert remote_module.calibration_data is not None
            calibration_data_tensor = torch.cat(remote_module.calibration_data, dim=0).to(device)
            self.private_modules[name] = self.private_modules[name].to(device)

            if has_any_qnn_layers(self.private_modules[name]):
                self.private_q_modules[name] = compile_brevitas_qat_model(
                    self.private_modules[name],
                    calibration_data_tensor,
                    n_bits=n_bits,
                    rounding_threshold_bits=rounding_threshold_bits,
                    configuration=configuration,
                    p_error=p_error,
                    device=device,
                )
            else:
                # If all layers are linear and the GLWE backend is available
                # then simply quantize the model without compiling with
                # Concrete Python.
                if self._has_only_large_linear_layers and has_glwe_backend():
                    self.executor = GLWELinearLayerExecutor(
                        use_dynamic_quantization=use_dynamic_quantization
                    )
                    self.private_q_modules[name] = build_quantized_module(
                        self.private_modules[name],
                        calibration_data_tensor,
                        n_bits=n_bits,
                        rounding_threshold_bits=rounding_threshold_bits,
                        keep_onnx=True,
                        device=device,
                    )

                    # Update executor for all remote modules
                    for module in self.remote_modules.values():
                        module.executor = self.executor

                    vals = self.private_q_modules[name].quant_layers_dict.values()
                    _, q_op = next(iter(vals))
                    const_inp = q_op.constant_inputs[1]  # Get the weights, the bias is in [2]

                    if not use_dynamic_quantization:
                        const_inp.values = const_inp.qvalues.astype(numpy.float32)
                    const_inp.qvalues = const_inp.qvalues.astype(numpy.int16)
                else:
                    self.private_q_modules[name] = compile_torch_model(
                        self.private_modules[name],
                        calibration_data_tensor,
                        n_bits=n_bits,
                        rounding_threshold_bits=rounding_threshold_bits,
                        configuration=configuration,
                        p_error=p_error,
                        device=device,
                    )

            self.remote_modules[name].private_q_module = self.private_q_modules[name]

            remote_module.calibration_data = None

    def _save_fhe_circuit(self, path: Path, via_mlir: bool = False):
        """Private method that saves the FHE circuits.

        Args:
            path (Path): The directory where the FHE circuit will be saved.
            via_mlir (bool): if fhe circuits should be serialized using via_mlir option
                useful for cross-platform (compile on one architecture and run on another).
        """
        for i, module_name in enumerate(self.module_names):
            onnx_model = self.private_q_modules[module_name].onnx_model
            private_q_module = self.private_q_modules[module_name]

            if onnx_model is not None:
                input_shapes = [
                    tuple(elt.dim_value for elt in onnx_input.type.tensor_type.shape.dim)
                    for onnx_input in onnx_model.graph.input
                ]
                assert len(input_shapes) == 1, "Multi-input circuits not supported yet"

                shape_str = tuple_to_underscore_str(input_shapes[0])
                module_path = path / module_name / shape_str
                module_path.mkdir(parents=True, exist_ok=True)

                if self.executor is not None:
                    # Extract and save private weights
                    prefix = f"{module_name}.private_module"
                    matching_keys = [
                        k for k in self.model.state_dict().keys() if k.startswith(prefix)
                    ]
                    assert (
                        len(matching_keys) == 1
                    ), f"Expected 1 match for `{prefix}`, found `{len(matching_keys)}`"
                    private_remote_weights = self.model.state_dict()[matching_keys[0]]

                    # Ensure target directories exist
                    server_path = module_path / "server"
                    server_path.mkdir(parents=True, exist_ok=True)
                    self.remote_modules[module_name].private_remote_weights_path = server_path
                    array_to_save = private_remote_weights.cpu().numpy().astype(numpy.float64)
                    numpy.save(server_path / f"remote_weights_layer{i}.npy", array_to_save)

                    # Extract quantized layer
                    layers_in_module = list(private_q_module.quant_layers_dict.values())
                    assert (
                        len(layers_in_module) == 1
                    ), "Expected exactly one linear layer in `QuantizedModule`"
                    quantized_linear_op = layers_in_module[0][1]
                    assert quantized_linear_op.supported_by_linear_backend()
                    _, quantized_layer = next(iter(private_q_module.quant_layers_dict.items()))

                    # Save bias if present
                    has_bias = len(quantized_layer[1].constant_inputs) > 1
                    if has_bias:
                        bias = list(quantized_layer[1].constant_inputs.values())[1].values
                        bias = bias.cpu().numpy().astype(numpy.float64)
                        numpy.save(server_path / f"remote_bias{i}.npy", bias)

                    # Save GLWE metadata
                    info = {
                        "module_name": str(module_name),
                        "shape": private_remote_weights.shape,
                        "transpose_inputs1": quantized_linear_op.attrs.get("transA", False),
                        "transpose_inputs2": quantized_linear_op.attrs.get("transB", False),
                        "bias": has_bias,
                        "input_n_bits": private_q_module.input_quantizers[0].quant_options.n_bits,
                    }

                    with open(server_path / "information.json", "w", encoding="utf-8") as f:
                        json.dump(info, f)

                else:
                    model_dev = FHEModelDev(str(module_path), private_q_module)
                    model_dev.save(via_mlir=via_mlir)

    def save_and_clear_private_info(self, path: Path, via_mlir: bool = True):
        """Save the PyTorch model to the provided path and also saves the corresponding FHE circuit.

        Args:
            path (Path): The directory where the model and the FHE circuit will be saved.
            via_mlir (bool): if fhe circuits should be serialized using via_mlir option
                useful for cross-platform (compile on one architecture and run on another)
        """

        def clear_private_info(module):
            # Remove private information
            for attr in [
                "private_module",
                "calibration_data",
                "private_q_module",
                "private_key",
                "compression_key",
            ]:
                if hasattr(module, attr):
                    setattr(module, attr, None)

            for child in module.children():
                clear_private_info(child)

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save the FHE circuit in the same directory
        self._save_fhe_circuit(path, via_mlir=via_mlir)

        # Developer-side: Save the complete model, including private info
        dev_model_path = path.parent.parent / "dev" / "full_model.pth"
        dev_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), dev_model_path.resolve())

        # Client-side: Save the model, excluding remote module information
        # Save the model state dict, instead of the entire model structure, due to a Brevitas issue
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4572

        # Clear private info from the full model before saving
        clear_private_info(self.model)
        client_model_path = path.parent.parent / "client" / "client_model.pth"
        client_model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), client_model_path.resolve())

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
