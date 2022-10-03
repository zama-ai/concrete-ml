"""APIs for FHE deployment."""

import json
from pathlib import Path
from typing import Any

import concrete.numpy as cnp
import numpy

from ..common.debugging.custom_assert import assert_true
from ..quantization.quantized_module import QuantizedModule
from ..quantization.quantizers import UniformQuantizer
from ..sklearn import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ElasticNet,
    GammaRegressor,
    Lasso,
    LinearRegression,
    LinearSVC,
    LinearSVR,
    LogisticRegression,
    NeuralNetClassifier,
    NeuralNetRegressor,
    PoissonRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    Ridge,
    TweedieRegressor,
    XGBClassifier,
    XGBRegressor,
)
from ..version import __version__ as CML_VERSION

AVAILABLE_MODEL = [
    RandomForestClassifier,
    RandomForestRegressor,
    XGBClassifier,
    XGBRegressor,
    LinearRegression,
    Lasso,
    Ridge,
    ElasticNet,
    LogisticRegression,
    LinearSVC,
    LinearSVR,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    NeuralNetClassifier,
    NeuralNetRegressor,
    TweedieRegressor,
    GammaRegressor,
    PoissonRegressor,
    QuantizedModule,
]


class FHEModelServer:
    """Server API to load and run the FHE circuit."""

    server: cnp.Server

    def __init__(self, path_dir: str):
        """Initialize the FHE API.

        Args:
            path_dir (str): the path to the directory where the circuit is saved
        """

        self.path_dir = path_dir

        # Load the FHE circuit
        self.load()

    def load(self):
        """Load the circuit."""
        self.server = cnp.Server.load(Path(self.path_dir).joinpath("server.zip"))

    def run(
        self,
        serialized_encrypted_quantized_data: cnp.PublicArguments,
        serialized_evaluation_keys: cnp.EvaluationKeys,
    ) -> cnp.PublicResult:
        """Run the model on the server over encrypted data.

        Args:
            serialized_encrypted_quantized_data (cnp.PublicArguments): the encrypted, quantized
                and serialized data
            serialized_evaluation_keys (cnp.EvaluationKeys): the serialized evaluation keys

        Returns:
            cnp.PublicResult: the result of the model
        """
        assert_true(self.server is not None, "Model has not been loaded.")

        unserialized_encrypted_quantized_data = self.server.client_specs.unserialize_public_args(
            serialized_encrypted_quantized_data
        )
        unserialized_evaluation_keys = cnp.EvaluationKeys.unserialize(serialized_evaluation_keys)
        result = self.server.run(
            unserialized_encrypted_quantized_data, unserialized_evaluation_keys
        )
        serialized_result = self.server.client_specs.serialize_public_result(result)
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

    def _clean_dict_types_for_json(self, d: dict) -> dict:
        """Clean all values in the dict to be json serializable.

        Args:
            d (dict): the dict to clean

        Returns:
            dict: the cleaned dict
        """
        key_to_delete = []
        for key, value in d.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                d[key] = [self._clean_dict_types_for_json(v) for v in value]
            elif isinstance(value, dict):
                d[key] = self._clean_dict_types_for_json(value)
            elif isinstance(value, (numpy.generic, numpy.ndarray)):
                d[key] = d[key].tolist()
            elif isinstance(value, (UniformQuantizer)):
                key_to_delete.append(key)
        for key in key_to_delete:
            d.pop(key)
        return d

    def _export_model_to_json(self):
        """Export the quantizers to a json file."""
        serialized_processing = {
            "model_type": self.model.__class__.__name__,
            "model_post_processing_params": self.model.post_processing_params,
            "input_quantizers": [],
            "output_quantizers": [],
        }
        for quantizer in self.model.input_quantizers:
            quantizer_dict = quantizer.__dict__
            serialized_processing["input_quantizers"].append(quantizer_dict)

        for quantizer in self.model.output_quantizers:
            quantizer_dict = quantizer.__dict__
            serialized_processing["output_quantizers"].append(quantizer_dict)

        serialized_processing = self._clean_dict_types_for_json(serialized_processing)

        # Add the version of the current CML library
        serialized_processing["cml_version"] = CML_VERSION
        with open(
            Path(self.path_dir).joinpath("serialized_processing.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(serialized_processing, f)

    def save(self):
        """Export all needed artifacts for the client and server.

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
        assert_true(
            hasattr(self.model, "fhe_circuit"),
            "The model must be compiled and have a fhe_circuit object",
        )

        # Model must be compiled with jit=False
        # In a jit model, everything is in memory so it is not serializable.
        assert_true(
            not self.model.fhe_circuit.configuration.jit,
            "The model must be compiled with the configuration option jit=False.",
        )

        # Export the quantizers
        self._export_model_to_json()

        # First save the circuit for the server
        path_circuit_server = Path(self.path_dir).joinpath("server.zip")
        self.model.fhe_circuit.server.save(path_circuit_server)

        # Save the circuit for the client
        path_circuit_client = Path(self.path_dir).joinpath("client.zip")
        self.model.fhe_circuit.client.save(path_circuit_client)


class FHEModelClient:
    """Client API to encrypt and decrypt FHE data."""

    client: cnp.Client

    def __init__(self, path_dir: str, key_dir: str = None):
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
        self.client = cnp.Client.load(Path(self.path_dir).joinpath("client.zip"), self.key_dir)

        # Load the quantizers
        with open(
            Path(self.path_dir).joinpath("serialized_processing.json"), "r", encoding="utf-8"
        ) as f:
            serialized_processing = json.load(f)

        # Make sure the version in serialized_model is the same as CML_VERSION
        assert_true(
            serialized_processing["cml_version"] == CML_VERSION,
            f"The version of Concrete ML library ({CML_VERSION}) is different "
            f"from the one used to save the model ({serialized_processing['cml_version']}). "
            "Please update to the proper Concrete ML version.",
        )

        # Create dict with available models along with their name
        model_dict = {model_type().__class__.__name__: model_type for model_type in AVAILABLE_MODEL}

        # Initialize the model
        self.model = model_dict[serialized_processing["model_type"]]()
        self.model.input_quantizers = []
        self.model.output_quantizers = []
        for quantizer_dict in serialized_processing["input_quantizers"]:
            self.model.input_quantizers.append(UniformQuantizer(**quantizer_dict))
        for quantizer_dict in serialized_processing["output_quantizers"]:
            self.model.output_quantizers.append(UniformQuantizer(**quantizer_dict))

        # Load model parameters
        self.model.post_processing_params = serialized_processing["model_post_processing_params"]

    def generate_private_and_evaluation_keys(self, force=False):
        """Generate the private and evaluation keys.

        Args:
            force (bool): if True, regenerate the keys even if they already exist
        """
        self.client.keygen(force)

    def get_serialized_evaluation_keys(self) -> cnp.EvaluationKeys:
        """Get the serialized evaluation keys.

        Returns:
            cnp.EvaluationKeys: the evaluation keys
        """
        return self.client.evaluation_keys.serialize()

    def quantize_encrypt_serialize(self, x: numpy.ndarray) -> cnp.PublicArguments:
        """Quantize, encrypt and serialize the values.

        Args:
            x (numpy.ndarray): the values to quantize, encrypt and serialize

        Returns:
            cnp.PublicArguments: the quantized, encrypted and serialized values
        """
        # Quantize the values
        quantized_x = self.model.quantize_input(x)

        # Encrypt the values
        enc_qx = self.client.encrypt(quantized_x)

        # Serialize the encrypted values to be sent to the server
        serialized_enc_qx = self.client.specs.serialize_public_args(enc_qx)
        return serialized_enc_qx

    def deserialize_decrypt(
        self, serialized_encrypted_quantized_result: cnp.PublicArguments
    ) -> numpy.ndarray:
        """Deserialize and decrypt the values.

        Args:
            serialized_encrypted_quantized_result (cnp.PublicArguments): the serialized, encrypted
                and quantized result

        Returns:
            numpy.ndarray: the decrypted and desarialized values
        """
        # Unserialize the encrypted values
        unserialized_encrypted_quantized_result = self.client.specs.unserialize_public_result(
            serialized_encrypted_quantized_result
        )

        # Decrypt the values
        unserialized_decrypted_quantized_result = self.client.decrypt(
            unserialized_encrypted_quantized_result
        )
        return unserialized_decrypted_quantized_result

    def deserialize_decrypt_dequantize(
        self, serialized_encrypted_quantized_result: cnp.PublicArguments
    ) -> numpy.ndarray:
        """Deserialize, decrypt and dequantize the values.

        Args:
            serialized_encrypted_quantized_result (cnp.PublicArguments): the serialized, encrypted
                and quantized result

        Returns:
            numpy.ndarray: the decrypted (dequantized) values
        """
        # Decrypt and desarialize the values
        unserialized_decrypted_quantized_result = self.deserialize_decrypt(
            serialized_encrypted_quantized_result
        )

        # Dequantize the values and apply the model post processing
        unserialized_decrypted_dequantized_result = self.model.post_processing(
            unserialized_decrypted_quantized_result
        )
        return unserialized_decrypted_dequantized_result
