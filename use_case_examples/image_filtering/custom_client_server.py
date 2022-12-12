"Client-server interface implementation for custom models."

from pathlib import Path
from typing import Any

import concrete.numpy as cnp
import numpy as np
from filters import Filter

from concrete.ml.common.debugging.custom_assert import assert_true


class CustomFHEDev:
    """Dev API to save the custom model and then load and run the FHE circuit."""

    model: Any = None

    def __init__(self, path_dir: str, model: Any = None):
        """Initialize the FHE API.

        Args:
            path_dir (str): the path to the directory where the circuit is saved
            model (Any): the model to use for the FHE API
        """

        self.path_dir = Path(path_dir)
        self.model = model

        # Create the directory path if it does not exist yet
        Path(self.path_dir).mkdir(parents=True, exist_ok=True)

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

        # Export the parameters
        self.model.to_json(path_dir=self.path_dir, file_name="serialized_processing")

        # Save the circuit for the server
        path_circuit_server = self.path_dir / "server.zip"
        self.model.fhe_circuit.server.save(path_circuit_server)

        # Save the circuit for the client
        path_circuit_client = self.path_dir / "client.zip"
        self.model.fhe_circuit.client.save(path_circuit_client)


class CustomFHEClient:
    """Client API to encrypt and decrypt FHE data."""

    client: cnp.Client

    def __init__(self, path_dir: str, key_dir: str = None):
        """Initialize the FHE API.

        Args:
            path_dir (str): the path to the directory where the circuit is saved
            key_dir (str): the path to the directory where the keys are stored
        """
        self.path_dir = Path(path_dir)
        self.key_dir = Path(key_dir)

        # If path_dir does not exist, raise an error
        assert_true(
            Path(path_dir).exists(), f"{path_dir} does not exist. Please specify a valid path."
        )

        # Load
        self.load()

    def load(self):  # pylint: disable=no-value-for-parameter
        """Load the parameters along with the FHE specs."""

        # Load the client
        self.client = cnp.Client.load(self.path_dir / "client.zip", self.key_dir)

        # Load the model
        self.model = Filter.from_json(self.path_dir / "serialized_processing.json")

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

    def pre_process_encrypt_serialize(self, x: np.ndarray) -> cnp.PublicArguments:
        """Encrypt and serialize the values.

        Args:
            x (numpy.ndarray): the values to encrypt and serialize

        Returns:
            cnp.PublicArguments: the encrypted and serialized values
        """
        # Pre-process the values
        x = self.model.pre_processing(x)

        # Encrypt the values
        enc_x = self.client.encrypt(x)

        # Serialize the encrypted values to be sent to the server
        serialized_enc_x = self.client.specs.serialize_public_args(enc_x)
        return serialized_enc_x

    def deserialize_decrypt_post_process(
        self, serialized_encrypted_output: cnp.PublicArguments
    ) -> np.ndarray:
        """Deserialize, decrypt and post-process the values.

        Args:
            serialized_encrypted_output (cnp.PublicArguments): the serialized qnd encrypted output

        Returns:
            numpy.ndarray: the decrypted values
        """
        # Deserialize the encrypted values
        deserialized_encrypted_output = self.client.specs.unserialize_public_result(
            serialized_encrypted_output
        )

        # Decrypt the values
        deserialized_decrypted_output = self.client.decrypt(deserialized_encrypted_output)

        # Apply the model post processing
        deserialized_decrypted_output = self.model.post_processing(deserialized_decrypted_output)
        return deserialized_decrypted_output


class CustomFHEServer:
    """Server API to load and run the FHE circuit."""

    server: cnp.Server

    def __init__(self, path_dir: str):
        """Initialize the FHE API.

        Args:
            path_dir (str): the path to the directory where the circuit is saved
        """

        self.path_dir = Path(path_dir)

        # Load the FHE circuit
        self.load()

    def load(self):
        """Load the circuit."""
        self.server = cnp.Server.load(self.path_dir / "server.zip")

    def run(
        self,
        serialized_encrypted_data: cnp.PublicArguments,
        serialized_evaluation_keys: cnp.EvaluationKeys,
    ) -> cnp.PublicResult:
        """Run the model on the server over encrypted data.

        Args:
            serialized_encrypted_data (cnp.PublicArguments): the encrypted and serialized data
            serialized_evaluation_keys (cnp.EvaluationKeys): the serialized evaluation keys

        Returns:
            cnp.PublicResult: the result of the model
        """
        assert_true(self.server is not None, "Model has not been loaded.")

        deserialized_encrypted_data = self.server.client_specs.unserialize_public_args(
            serialized_encrypted_data
        )
        deserialized_evaluation_keys = cnp.EvaluationKeys.unserialize(serialized_evaluation_keys)
        result = self.server.run(deserialized_encrypted_data, deserialized_evaluation_keys)
        serialized_result = self.server.client_specs.serialize_public_result(result)
        return serialized_result
