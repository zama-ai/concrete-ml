"""APIs for FHE deployment."""

import json
import sys
import zipfile
from pathlib import Path
from typing import Any, Optional

import numpy

from concrete import fhe

from ..common.debugging.custom_assert import assert_true
from ..common.serialization.dumpers import dump
from ..common.serialization.loaders import load
from ..version import __version__ as CML_VERSION

try:
    # 3.8 and above
    # pylint: disable-next=no-name-in-module
    from importlib.metadata import version
except ImportError:  # pragma: no cover
    # 3.7 and below
    # pylint: disable-next=no-name-in-module
    from importlib_metadata import version


class FHEModelServer:
    """Server API to load and run the FHE circuit."""

    server: fhe.Server

    def __init__(self, path_dir: str):
        """Initialize the FHE API.

        Args:
            path_dir (str): the path to the directory where the circuit is saved
        """

        self.path_dir = path_dir

        # Load the FHE circuit
        self.load()

    def load(self):
        """Load the circuit.

        Raises:
            ValueError: if mismatch in versions between serialized file and runtime
        """
        # Load versions for checking
        with zipfile.ZipFile(Path(self.path_dir).joinpath("server.zip")) as client_zip:
            with client_zip.open("versions.json", mode="r") as file:
                versions = json.load(file)

        errors = []
        packages_to_check = {"concrete-python"}
        for package_name, package_version in versions.items():
            if package_name not in packages_to_check:
                continue
            current_version = version(package_name)
            if package_version != current_version:  # pragma: no cover
                errors.append((package_name, package_version, current_version))
        if errors:  # pragma: no cover
            raise ValueError(
                "Version mismatch for packages: \n"
                + "\n".join(f"{error[0]}: {error[1]} != {error[2]}" for error in errors)
            )

        if not versions["python"].startswith(
            f"{sys.version_info.major}.{sys.version_info.minor}"
        ):  # pragma: no cover
            raise ValueError(
                "Not the same Python version between the compiler and the server."
                f"{versions['python']} != {sys.version_info.major}.{sys.version_info.minor}"
            )

        self.server = fhe.Server.load(Path(self.path_dir).joinpath("server.zip"))

    def run(
        self,
        serialized_encrypted_quantized_data: bytes,
        serialized_evaluation_keys: bytes,
    ) -> bytes:
        """Run the model on the server over encrypted data.

        Args:
            serialized_encrypted_quantized_data (bytes): the encrypted, quantized
                and serialized data
            serialized_evaluation_keys (bytes): the serialized evaluation keys

        Returns:
            bytes: the result of the model
        """
        assert_true(self.server is not None, "Model has not been loaded.")

        deserialized_encrypted_quantized_data = fhe.Value.deserialize(
            serialized_encrypted_quantized_data
        )
        deserialized_evaluation_keys = fhe.EvaluationKeys.deserialize(serialized_evaluation_keys)
        result = self.server.run(
            deserialized_encrypted_quantized_data, evaluation_keys=deserialized_evaluation_keys
        )
        serialized_result = result.serialize()
        return serialized_result


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

    def _export_model_to_json(self) -> Path:
        """Export the quantizers to a json file.

        Returns:
            Path: the path to the json file
        """
        serialized_processing = {
            "model_type": self.model.__class__,
            "model_post_processing_params": self.model.post_processing_params,
            "input_quantizers": self.model.input_quantizers,
            "output_quantizers": self.model.output_quantizers,
            "cml_version": CML_VERSION,
        }

        # Export the `is_fitted` attribute for built-in models
        if hasattr(self.model, "is_fitted"):
            serialized_processing["is_fitted"] = self.model.is_fitted

        # Dump json
        json_path = Path(self.path_dir).joinpath("serialized_processing.json")
        with open(json_path, "w", encoding="utf-8") as file:
            dump(serialized_processing, file)
        return json_path

    def save(self, via_mlir: bool = False):
        """Export all needed artifacts for the client and server.

        Arguments:
            via_mlir (bool): serialize with `via_mlir` option from Concrete-Python.
                For more details on the topic please refer to Concrete-Python's documentation.

        Raises:
            Exception: path_dir is not empty
        """
        # Check if the path_dir is empty with pathlib
        listdir = list(Path(self.path_dir).glob("**/*"))
        if len(listdir) > 0:
            raise Exception(
                f"path_dir: {self.path_dir} is not empty."
                "Please delete it before saving a new model."
            )
        self.model.check_model_is_compiled()

        # Model must be compiled with jit=False
        # In a jit model, everything is in memory so it is not serializable.
        assert_true(
            not self.model.fhe_circuit.configuration.jit,
            "The model must be compiled with the configuration option jit=False.",
        )

        # Export the quantizers
        json_path = self._export_model_to_json()

        # First save the circuit for the server
        path_circuit_server = Path(self.path_dir).joinpath("server.zip")
        self.model.fhe_circuit.server.save(path_circuit_server, via_mlir=via_mlir)

        # Save the circuit for the client
        path_circuit_client = Path(self.path_dir).joinpath("client.zip")
        self.model.fhe_circuit.client.save(path_circuit_client)

        with zipfile.ZipFile(path_circuit_client, "a") as zip_file:
            zip_file.write(filename=json_path, arcname="serialized_processing.json")

        # Add versions
        versions_path = Path(self.path_dir).joinpath("versions.json")
        versions = {
            package_name: version(package_name)
            for package_name in ["concrete-ml", "concrete-python"]
        }
        versions[
            "python"
        ] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        with open(versions_path, "w", encoding="utf-8") as file:
            json.dump(fp=file, obj=versions)

        with zipfile.ZipFile(path_circuit_server, "a") as zip_file:
            zip_file.write(filename=versions_path, arcname="versions.json")

        with zipfile.ZipFile(path_circuit_client, "a") as zip_file:
            zip_file.write(filename=versions_path, arcname="versions.json")

        json_path.unlink()


class FHEModelClient:
    """Client API to encrypt and decrypt FHE data."""

    client: fhe.Client

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
        """Load the quantizers along with the FHE specs.

        Raises:
            ValueError: if mismatch in versions between serialized file and runtime
        """
        self.client = fhe.Client.load(Path(self.path_dir).joinpath("client.zip"), self.key_dir)

        # Load the quantizers
        with zipfile.ZipFile(Path(self.path_dir).joinpath("client.zip")) as client_zip:
            with client_zip.open("serialized_processing.json", mode="r") as file:
                serialized_processing = load(file)

        # Load versions for checking
        with zipfile.ZipFile(Path(self.path_dir).joinpath("client.zip")) as client_zip:
            with client_zip.open("versions.json", mode="r") as file:
                versions = json.load(file)

        errors = []
        packages_to_check = {"concrete-python"}
        for package_name, package_version in versions.items():
            if package_name not in packages_to_check:
                continue
            current_version = version(package_name)
            if package_version != current_version:  # pragma: no cover
                errors.append((package_name, package_version, current_version))
        if errors:  # pragma: no cover
            raise ValueError(
                "Version mismatch for packages: \n"
                + "\n".join(f"{error[0]}: {error[1]} != {error[2]}" for error in errors)
            )

        # Make sure the version in serialized_model is the same as CML_VERSION
        assert_true(
            serialized_processing["cml_version"] == CML_VERSION,
            f"The version of Concrete ML library ({CML_VERSION}) is different "
            f"from the one used to save the model ({serialized_processing['cml_version']}). "
            "Please update to the proper Concrete ML version.",
        )

        # Initialize the model
        self.model = serialized_processing["model_type"]()
        self.model.input_quantizers = serialized_processing["input_quantizers"]
        self.model.output_quantizers = serialized_processing["output_quantizers"]

        # Load the `_is_fitted` private attribute for built-in models
        if "is_fitted" in serialized_processing:
            # This private access should be temporary as the Client-Server interface could benefit
            # from built-in serialization load/dump methods
            # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3243
            # pylint: disable-next=protected-access
            self.model._is_fitted = serialized_processing["is_fitted"]

        # Load model parameters
        # Add some checks on post-processing-params
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3131
        self.model.post_processing_params = serialized_processing["model_post_processing_params"]

    def generate_private_and_evaluation_keys(self, force=False):
        """Generate the private and evaluation keys.

        Args:
            force (bool): if True, regenerate the keys even if they already exist
        """
        self.client.keygen(force)

    def get_serialized_evaluation_keys(self) -> bytes:
        """Get the serialized evaluation keys.

        Returns:
            bytes: the evaluation keys
        """
        return self.client.evaluation_keys.serialize()

    def quantize_encrypt_serialize(self, x: numpy.ndarray) -> bytes:
        """Quantize, encrypt and serialize the values.

        Args:
            x (numpy.ndarray): the values to quantize, encrypt and serialize

        Returns:
            bytes: the quantized, encrypted and serialized values
        """
        # Quantize the values
        quantized_x = self.model.quantize_input(x)

        # Encrypt the values
        enc_qx = self.client.encrypt(quantized_x)

        # Serialize the encrypted values to be sent to the server
        serialized_enc_qx = enc_qx.serialize()
        return serialized_enc_qx

    def deserialize_decrypt(self, serialized_encrypted_quantized_result: bytes) -> numpy.ndarray:
        """Deserialize and decrypt the values.

        Args:
            serialized_encrypted_quantized_result (bytes): the serialized, encrypted
                and quantized result

        Returns:
            numpy.ndarray: the decrypted and deserialized values
        """
        # Deserialize the encrypted values
        deserialized_encrypted_quantized_result = fhe.Value.deserialize(
            serialized_encrypted_quantized_result
        )

        # Decrypt the values
        deserialized_decrypted_quantized_result = self.client.decrypt(
            deserialized_encrypted_quantized_result
        )
        assert isinstance(deserialized_decrypted_quantized_result, numpy.ndarray)
        return deserialized_decrypted_quantized_result

    def deserialize_decrypt_dequantize(
        self, serialized_encrypted_quantized_result: bytes
    ) -> numpy.ndarray:
        """Deserialize, decrypt and de-quantize the values.

        Args:
            serialized_encrypted_quantized_result (bytes): the serialized, encrypted
                and quantized result

        Returns:
            numpy.ndarray: the decrypted (de-quantized) values
        """
        # Decrypt and deserialize the values
        deserialized_decrypted_quantized_result = self.deserialize_decrypt(
            serialized_encrypted_quantized_result
        )

        # De-quantize the values
        deserialized_decrypted_dequantized_result = self.model.dequantize_output(
            deserialized_decrypted_quantized_result
        )

        # Apply post-processing the to de-quantized values
        deserialized_decrypted_dequantized_result = self.model.post_processing(
            deserialized_decrypted_dequantized_result
        )

        return deserialized_decrypted_dequantized_result
