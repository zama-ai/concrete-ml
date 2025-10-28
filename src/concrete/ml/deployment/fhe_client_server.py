"""APIs for FHE deployment."""

import json
import sys
import zipfile
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import concrete_ml_extensions as fhext
import numpy
from concrete.fhe import tfhers

from concrete import fhe

from ..common.debugging.custom_assert import assert_true
from ..common.serialization.dumpers import dump
from ..common.serialization.loaders import load
from ..common.utils import CiphertextFormat, to_tuple
from ..quantization import QuantizedModule
from ..torch.numpy_module import NumpyModule
from ..version import __version__ as CML_VERSION
from ._utils import deserialize_encrypted_values, serialize_encrypted_values

try:
    # 3.8 and above
    # pylint: disable-next=no-name-in-module
    from importlib.metadata import version
except ImportError:  # pragma: no cover
    # 3.7 and below
    # pylint: disable-next=no-name-in-module
    from importlib_metadata import version


class DeploymentMode(Enum):
    """Mode for the FHE API."""

    INFERENCE = "inference"
    TRAINING = "training"

    @staticmethod
    def is_valid(mode: Union["DeploymentMode", str]) -> bool:
        """Indicate if the given name is a supported mode.

        Args:
            mode (Union[Mode, str]): The mode to check.

        Returns:
            bool: Whether the mode is supported or not.
        """
        return mode in {member.value for member in DeploymentMode.__members__.values()}


def check_concrete_versions(zip_path: Path):
    """Check that current versions match the ones used in development.

    This function loads the version JSON file found in client.zip or server.zip files and then
    checks that current package versions (Concrete Python, Concrete ML) as well as the Python
    current version all match the ones that are currently installed.

    Args:
        zip_path (Path): The path to the client or server zip file that contains the version.json
            file to check.

    Raises:
        ValueError: If at least one version mismatch is found.
    """

    with zipfile.ZipFile(zip_path) as zip_file:
        is_mlir = "circuit.mlir" in zip_file.namelist()
        with zip_file.open("versions.json", mode="r") as file:
            versions = json.load(file)

    # Check for package coherence
    packages_to_check = {"concrete-python", "concrete-ml"}

    errors = []
    for package_name, package_version in versions.items():
        if package_name not in packages_to_check:
            continue

        if package_name == "concrete-ml":
            current_version = CML_VERSION

        else:
            current_version = version(package_name)

        if package_version != current_version:  # pragma: no cover
            errors.append((package_name, package_version, current_version))

    # Raise an error if at least one package version did not match the one currently installed
    if errors:  # pragma: no cover
        raise ValueError(
            "Version mismatch for packages: \n"
            + "\n".join(f"{error[0]}: {error[1]} != {error[2]}" for error in errors)
        )

    if not is_mlir:
        # Raise an error if the Python version do not match the one currently installed
        if not versions["python"].startswith(
            f"{sys.version_info.major}.{sys.version_info.minor}"
        ):  # pragma: no cover
            raise ValueError(
                "Not the same Python version between the zip file "
                "({zip_path}) and the current machine: "
                f"{versions['python']} != {sys.version_info.major}.{sys.version_info.minor}"
            )


class FHEModelServer:
    """Server API to load and run the FHE circuit."""

    server: fhe.Server
    _tfhers_bridge: Optional[tfhers.bridge.Bridge] = None

    def __init__(self, path_dir: str):
        """Initialize the FHE API.

        Args:
            path_dir (str): the path to the directory where the circuit is saved
        """

        self.path_dir = path_dir

        # Load the FHE circuit
        self.load()
        self._tfhers_bridge = None

    def load(self):
        """Load the circuit."""
        server_zip_path = Path(self.path_dir).joinpath("server.zip")

        check_concrete_versions(server_zip_path)

        self.server = fhe.Server.load(Path(self.path_dir).joinpath("server.zip"))

        specs = self.server.client_specs

        if specs.tfhers_specs is not None and len(specs.tfhers_specs.output_types_per_func) > 0:
            lst_keys = list(specs.tfhers_specs.output_types_per_func.keys())
            assert (
                len(lst_keys) == 1
            ), "The FHE model server cannot be instantiated on a multi-function circuit"
            func_name = lst_keys[0]

            func_specs = specs.tfhers_specs.output_types_per_func[func_name]
            if func_specs is not None and func_specs[0] is not None:
                server_client = fhe.Client(self.server.client_specs)
                self._tfhers_bridge = tfhers.new_bridge(server_client)

    # We should make 'serialized_encrypted_quantized_data' handle unpacked inputs, as Concrete does,
    # instead of tuples
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4477
    # We should also rename the input arguments to remove the `serialized` part, as we now accept
    # both serialized and deserialized input values
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4476
    def run(
        self,
        serialized_encrypted_quantized_data: Union[
            bytes, fhe.Value, Tuple[bytes, ...], Tuple[fhe.Value, ...]
        ],
        serialized_evaluation_keys: bytes,
    ) -> Union[bytes, fhe.Value, Tuple[bytes, ...], Tuple[fhe.Value, ...]]:
        """Run the model on the server over encrypted data.

        Args:
            serialized_encrypted_quantized_data (Union[bytes, fhe.Value, Tuple[bytes, ...], \
                Tuple[fhe.Value, ...]]): The encrypted and quantized values to consider. If these
                values are serialized (in bytes), they are first deserialized.
            serialized_evaluation_keys (bytes): The evaluation keys. If they are serialized (in
                bytes), they are first deserialized.

        Returns:
            Union[bytes, fhe.Value, Tuple[bytes, ...], Tuple[fhe.Value, ...]]: The model's encrypted
                and quantized results. If the inputs were initially serialized, the outputs are also
                serialized.
        """

        # TODO: make desr / ser optional
        assert_true(self.server is not None, "Model has not been loaded.")

        input_quant_encrypted = to_tuple(serialized_encrypted_quantized_data)

        # Make sure no inputs are None, to avoid any crash in Concrete
        assert not any(x is None for x in input_quant_encrypted), "No input values should be None"

        inputs_are_serialized = all(isinstance(x, bytes) for x in input_quant_encrypted)
        inputs_are_encrypted_values = all(isinstance(x, fhe.Value) for x in input_quant_encrypted)

        # Make sure inputs are either only serialized values or encrypted values
        assert (
            inputs_are_serialized ^ inputs_are_encrypted_values
        ), "Inputs must be all of the same types, either 'bytes' or 'concrete.fhe.Value'"

        # Deserialize the values if they are all serialized
        if inputs_are_serialized:
            if self._tfhers_bridge is not None:
                input_quant_encrypted = tuple(
                    [
                        self._tfhers_bridge.import_value(x, input_idx)
                        for input_idx, x in enumerate(input_quant_encrypted)
                    ]
                )
            else:
                input_quant_encrypted = to_tuple(
                    deserialize_encrypted_values(*input_quant_encrypted)
                )

        # Deserialize the evaluation keys if they are serialized
        evaluation_keys = serialized_evaluation_keys
        if isinstance(evaluation_keys, bytes):
            evaluation_keys = fhe.EvaluationKeys.deserialize(evaluation_keys)

        result_quant_encrypted = self.server.run(
            *input_quant_encrypted, evaluation_keys=evaluation_keys
        )

        # If inputs were serialized, return serialized values as well
        if inputs_are_serialized:
            if self._tfhers_bridge is not None:
                result_quant_encrypted = tuple(
                    [
                        self._tfhers_bridge.export_value(x, output_idx=idx)
                        for idx, x in enumerate(
                            to_tuple(result_quant_encrypted)
                        )  # pylint: disable=no-member
                    ]
                )
            else:
                result_quant_encrypted = serialize_encrypted_values(
                    *to_tuple(result_quant_encrypted)
                )

        # Mypy complains because the outputs of `serialize_encrypted_values` can be None, but here
        # we already made sure this is not the case
        return result_quant_encrypted  # type: ignore[return-value]


class FHEModelDev:
    """Dev API to save the model and then load and run the FHE circuit."""

    model: Any = None

    def __init__(self, path_dir: str, model: Any = None):
        """Initialize the FHE API.

        Args:
            path_dir (str): the path to the directory where the circuit is saved
            model (Any): the model to use for the FHE API
        """

        self.path_dir = path_dir
        self.model = model

        Path(self.path_dir).mkdir(parents=True, exist_ok=True)

    def _export_model_to_json(self, is_training: bool = False) -> Path:
        """Export the quantizers to a json file.

        Args:
            is_training (bool): If True, we export the training circuit.

        Returns:
            Path: the path to the json file
        """
        module_to_export = self.model.training_quantized_module if is_training else self.model
        serialized_processing = {
            "model_type": module_to_export.__class__,
            "model_post_processing_params": module_to_export.post_processing_params,
            "input_quantizers": module_to_export.input_quantizers,
            "output_quantizers": module_to_export.output_quantizers,
            "is_training": is_training,
            "ciphertext_format": module_to_export.ciphertext_format,
            "onnx_preprocessing": None,
        }

        preprocessing_module = getattr(module_to_export, "_preprocessing_module", None)
        if preprocessing_module is not None:
            serialized_processing["onnx_preprocessing"] = preprocessing_module.onnx_model

        # Export the `is_fitted` attribute for built-in models
        if hasattr(self.model, "is_fitted"):
            serialized_processing["is_fitted"] = self.model.is_fitted

        # Dump json
        json_path = Path(self.path_dir).joinpath("serialized_processing.json")
        with open(json_path, "w", encoding="utf-8") as file:
            dump(serialized_processing, file)

        return json_path

    def save(self, mode: DeploymentMode = DeploymentMode.INFERENCE, via_mlir: bool = True):
        """Export all needed artifacts for the client and server.

        Arguments:
            mode (DeploymentMode): the mode to save the FHE circuit,
                either "inference" or "training".
            via_mlir (bool): serialize with `via_mlir` option from Concrete-Python.

        Raises:
            Exception: path_dir is not empty or training module does not exist
            ValueError: if mode is not "inference" or "training"
        """

        if isinstance(mode, str):
            mode_lower = mode.lower()
            if not DeploymentMode.is_valid(mode_lower):
                raise ValueError("Mode must be either 'inference' or 'training'")
            mode = DeploymentMode(mode_lower)

        # Get fhe_circuit based on the mode
        if mode == DeploymentMode.TRAINING:

            # Check that training FHE circuit exists
            assert_true(
                hasattr(self.model, "training_quantized_module")
                and (self.model.training_quantized_module),
                "Training FHE circuit does not exist.",
            )
            self.model.training_quantized_module.check_model_is_compiled()
            fhe_circuit = self.model.training_quantized_module.fhe_circuit
        else:
            self.model.check_model_is_compiled()
            fhe_circuit = self.model.fhe_circuit

        # Check if the path_dir is empty with pathlib
        listdir = list(Path(self.path_dir).glob("**/*"))
        if len(listdir) > 0:
            raise Exception(
                f"path_dir: {self.path_dir} is not empty. "
                "Please delete it before saving a new model."
            )

        # Export the quantizers
        json_path = self._export_model_to_json(is_training=(mode == DeploymentMode.TRAINING))

        # Save the circuit for the server
        path_circuit_server = Path(self.path_dir).joinpath("server.zip")
        fhe_circuit.server.save(path_circuit_server, via_mlir=via_mlir)

        # Save the circuit for the client
        path_circuit_client = Path(self.path_dir).joinpath("client.zip")
        fhe_circuit.client.save(path_circuit_client)

        with zipfile.ZipFile(path_circuit_client, "a") as zip_file:
            zip_file.write(filename=json_path, arcname="serialized_processing.json")

        # Add versions
        versions_path = Path(self.path_dir).joinpath("versions.json")
        versions = {
            "concrete-python": version("concrete-python"),
            "concrete-ml": CML_VERSION,
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        }
        with open(versions_path, "w", encoding="utf-8") as file:
            json.dump(fp=file, obj=versions)

        with zipfile.ZipFile(path_circuit_server, "a") as zip_file:
            zip_file.write(filename=versions_path, arcname="versions.json")

        with zipfile.ZipFile(path_circuit_client, "a") as zip_file:
            zip_file.write(filename=versions_path, arcname="versions.json")

        versions_path.unlink()
        json_path.unlink()


class FHEModelClient:
    """Client API to encrypt and decrypt FHE data."""

    client: fhe.Client
    _tfhers_bridge: Optional[tfhers.bridge.Bridge] = None
    _tfhers_sk: Optional[bytes] = None
    _tfhers_pk: Optional[bytes] = None

    def __init__(self, path_dir: str, key_dir: Optional[str] = None):
        """Initialize the FHE API.

        Args:
            path_dir (str): the path to the directory where the circuit is saved
            key_dir (str): the path to the directory where the keys are stored
        """
        self.path_dir = path_dir
        self.key_dir = key_dir

        # If path_dir does not exist raise
        assert_true(
            Path(path_dir).exists(), f"{path_dir} does not exist. Please specify a valid path."
        )

        # Load
        self.load()

    def load(self):  # pylint: disable=no-value-for-parameter
        """Load the quantizers along with the FHE specs."""
        client_zip_path = Path(self.path_dir).joinpath("client.zip")

        self.client = fhe.Client.load(client_zip_path, self.key_dir)

        # Load the quantizers
        with zipfile.ZipFile(client_zip_path) as client_zip:
            with client_zip.open("serialized_processing.json", mode="r") as file:
                serialized_processing = load(file)

        # Load and check versions
        check_concrete_versions(client_zip_path)

        # Initialize the model
        self.model = serialized_processing["model_type"]()

        # Load the quantizers
        self.model.input_quantizers = serialized_processing["input_quantizers"]
        self.model.output_quantizers = serialized_processing["output_quantizers"]

        # Load the `_is_fitted` private attribute for built-in models
        if "is_fitted" in serialized_processing:
            # This private access should be temporary as the Client-Server interface could benefit
            # from built-in serialization load/dump methods
            # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3243
            # pylint: disable-next=protected-access
            self.model._is_fitted = serialized_processing["is_fitted"]

        self.model.ciphertext_format = serialized_processing["ciphertext_format"]

        if hasattr(self.model, "_preprocessing_module"):
            onnx_preprocessing = serialized_processing.get("onnx_preprocessing")
            # pylint: disable-next=protected-access
            self.model._preprocessing_module = (
                NumpyModule(onnx_preprocessing) if onnx_preprocessing is not None else None
            )

        # Load model parameters
        # Add some checks on post-processing-params
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3131
        self.model.post_processing_params = serialized_processing["model_post_processing_params"]

    def generate_private_and_evaluation_keys(self, force=False):
        """Generate the private and evaluation keys.

        Args:
            force (bool): if True, regenerate the keys even if they already exist
        """

        if self.model.ciphertext_format == CiphertextFormat.TFHE_RS:
            sk, pk, lwe_sk = fhext.keygen_radix()  # pylint: disable=no-member
            self._tfhers_bridge = tfhers.new_bridge(self.client)

            input_idx_to_key = {0: lwe_sk}
            self._tfhers_bridge.keygen_with_initial_keys(  # pylint: disable=no-member
                input_idx_to_key_buffer=input_idx_to_key
            )

            self._tfhers_sk = sk  # Public server key
            self._tfhers_pk = pk  # Private client key
        else:
            self.client.keygen(force)

    def get_serialized_evaluation_keys(
        self, include_tfhers_key: bool = False
    ) -> Union[bytes, Tuple[bytes, Optional[bytes]]]:
        """Get the serialized evaluation keys.

        Args:
            include_tfhers_key (bool): Whether to include the serialized TFHE-rs public key.

        Returns:
            bytes: the evaluation keys
        """
        # Generate private and evaluation keys if not already generated
        self.generate_private_and_evaluation_keys(force=False)

        if include_tfhers_key and self.model.ciphertext_format == CiphertextFormat.TFHE_RS:
            # should be generated by the call above
            assert self._tfhers_pk is not None
            return (self.client.evaluation_keys.serialize(), self._tfhers_pk)

        return self.client.evaluation_keys.serialize()

    def get_tfhers_secret_key(self) -> Optional[bytes]:
        """Get the secret key that decrypts TFHE-rs ciphertexts.

        This secret key use used by the client object to encrypt TFHE-rs ciphertexts
        that are sent to the server, but also to decrypt outputs produced by
        a server-side computation, that can be a Concrete ML model or a TFHE-rs program.

        Returns:
            bytes: the serialized TFHE-rs secret key
        """
        return self._tfhers_sk

    def quantize_encrypt_serialize(
        self, *x: Optional[numpy.ndarray]
    ) -> Union[Optional[bytes], Tuple[Optional[bytes], ...]]:
        """Quantize, encrypt and serialize the values.

        Args:
            x (Optional[numpy.ndarray]): The values to quantize, encrypt and serialize.

        Returns:
            Union[bytes, Tuple[bytes, ...]]: The quantized, encrypted and serialized values.
        """

        # Apply the same preprocessing as during standard forward passes when available so that
        # inputs expected by the FHE circuit (e.g., one-hot vectors for optimized embeddings) are
        # generated here too.
        if hasattr(self.model, "pre_processing"):
            x = to_tuple(self.model.pre_processing(*x))
        else:
            x = to_tuple(x)

        # Quantize the values
        x_quant = to_tuple(self.model.quantize_input(*x))

        if self.model.ciphertext_format == CiphertextFormat.TFHE_RS:
            input_is_signed = self.model.input_quantizers[0].is_signed

            encrypt_dtype = numpy.int8 if input_is_signed else numpy.uint8

            return tuple(
                [
                    fhext.encrypt_radix(x_quant[idx].astype(encrypt_dtype), self._tfhers_sk)
                    for idx in range(len(x_quant))
                ]
            )

        # Encrypt the values
        x_quant_encrypted = to_tuple(self.client.encrypt(*x_quant))

        # Serialize the encrypted values to be sent to the server
        return serialize_encrypted_values(*x_quant_encrypted)

    # We should find a better name for `serialized_encrypted_quantized_result`
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4476
    def deserialize_decrypt(
        self, *serialized_encrypted_quantized_result: Optional[bytes]
    ) -> Union[Any, Tuple[Any, ...]]:
        """Deserialize and decrypt the values.

        Args:
            serialized_encrypted_quantized_result (Optional[bytes]): The serialized, encrypted and
                quantized values.

        Returns:
            Union[Any, Tuple[Any, ...]]: The decrypted and deserialized values.
        """

        if self.model.ciphertext_format == CiphertextFormat.TFHE_RS:
            specs = self.client.specs.tfhers_specs
            func_name = list(specs.output_types_per_func.keys())[0]

            result_quant = tuple(
                [
                    fhext.decrypt_radix(
                        serialized_encrypted_quantized_result[idx],
                        specs.output_shapes_per_func[func_name][idx],
                        specs.output_types_per_func[func_name][idx].bit_width,
                        specs.output_types_per_func[func_name][idx].is_signed,
                        self._tfhers_sk,
                    )
                    for idx in range(len(serialized_encrypted_quantized_result))
                ]
            )
            if len(result_quant) == 1:
                result_quant = result_quant[0]
        else:
            # Deserialize the encrypted values
            result_quant_encrypted = to_tuple(
                deserialize_encrypted_values(*serialized_encrypted_quantized_result)
            )

            # Decrypt the values
            result_quant = self.client.decrypt(*result_quant_encrypted)

        return result_quant

    # We should find a better name for `serialized_encrypted_quantized_result`
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4476
    def deserialize_decrypt_dequantize(
        self, *serialized_encrypted_quantized_result: Optional[bytes]
    ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]:
        """Deserialize, decrypt and de-quantize the values.

        Args:
            serialized_encrypted_quantized_result (Optional[bytes]): The serialized, encrypted and
                quantized result

        Returns:
            Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]: The clear float values.
        """
        # Decrypt and deserialize the values
        result_quant = to_tuple(self.deserialize_decrypt(*serialized_encrypted_quantized_result))

        # De-quantize the values
        result = to_tuple(self.model.dequantize_output(*result_quant))

        # Apply post-processing to the de-quantized values
        # Side note: `post_processing` method from built-in models (not Quantized Modules) only
        # handles a single input. Calling the following is however not an issue for now as we expect
        # 'result' to be a tuple of length 1 in this case anyway. Still, we need to make sure this
        # does not break in the future if any built-in models starts to handle multiple outputs :
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4474
        assert len(result) == 1 or isinstance(
            self.model, QuantizedModule
        ), "Only 'QuantizedModule' instances can handle multi-outputs."

        # In training mode, note that this step does not make much sense for now. Still, nothing
        # breaks since QuantizedModule don't do anything in post-processing
        result_post_processed = self.model.post_processing(*result)

        return result_post_processed
