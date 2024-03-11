import copy
import json
from pathlib import Path
from typing import Dict, Optional

from concrete import fhe

from .utils import (
    deserialize_evaluation_keys,
    deserialize_value,
    encrypt_value,
    serialize_evaluation_keys,
    serialize_value,
)


def get_unsigned_int_dtypes_to_min_max():
    """Map a given dtype to the minimum and maximum values"""
    return {f"uint{n_bits}": (1, 2**n_bits - 1) for n_bits in range(1, 17)}


class EncryptedDataFrameOperator:
    """Encapsulate the methods required for an operator to be executed on encrypted data-frames."""

    # Requires parameters
    required_op_parameters = set(["name", "dtype"])
    required_metadata_parameters = set(["dtype", "n_bits"])

    # Define min and max values for allowed dtypes
    dtype_to_min_max = get_unsigned_int_dtypes_to_min_max()

    # Define the configurations to consider when encrypting values for a given operator:
    # - n: the total number of inputs the circuit requires
    # - pos: the input position to consider when encrypting a value
    op_to_encrypt_config = {
        "merge": {
            "n": 4,
            "pos": 1,
        },
    }

    # Define the list of operators currently supported
    supported_ops = op_to_encrypt_config.keys()

    def __init__(
        self,
        name: str,
        dtype: str,
        pandas_kwargs: Optional[dict] = None,
        encrypted_nan: Optional[fhe.Value] = None,
        evaluation_keys: Optional[fhe.EvaluationKeys] = None,
    ):
        self._check_if_op_is_supported(name)

        self.name = name
        self.dtype = dtype
        self.pandas_kwargs = copy.copy(pandas_kwargs) if pandas_kwargs is not None else {}

        self.encrypted_nan = encrypted_nan
        self.evaluation_keys = evaluation_keys

    def __str__(self) -> str:
        """Avoid printing serialized NaN and evaluation keys."""
        return (
            f"{self.__class__.__name__}(name={self.name}, dtype={self.dtype}, "
            f"pandas_kwargs={self.pandas_kwargs})"
        )

    @classmethod
    def init_and_check_metadata(cls, metadata_dir_path: Path, **kwargs):
        """Initialize the instance and check its coherence with the given metadata."""
        if not EncryptedDataFrameOperator.required_op_parameters.issubset(set(kwargs.keys())):
            raise ValueError(
                f"Operator defined by {kwargs} does not provide the following required arguments: "
                f"{EncryptedDataFrameOperator.required_op_parameters}"
            )

        op = cls(**kwargs)
        op.check_metadata(metadata_dir_path)

        return op

    def check_metadata(self, dir_path: Path):
        """Check that the loaded metadata are coherent."""
        with (dir_path / f"{self.name}/{self.dtype}/metadata.json").open("r") as metadata_file:
            metadata = json.load(metadata_file)

        if set(metadata.keys()) != self.required_metadata_parameters:
            raise ValueError(
                f"Metadata for operator {self.name} is not complete. Expected keys "
                f"{self.required_metadata_parameters} but got {metadata}"
            )

        if metadata["dtype"] != self.dtype:
            raise ValueError(
                f"Loaded dtype ({metadata['dtype']}) does not match the op's dtype ({self.dtype})"
            )

        return metadata

    def _check_if_op_is_supported(self, name: str):
        """Check that the current instance represents an operator that is supported."""
        if name not in self.supported_ops:
            raise ValueError(f"Operator {name} is currently not supported")

    def is_equal_to(self, other):
        """Define the equality between two encrypted data-frames.

        Encrypted NaN and evaluation keys are left out on purpose, as these byte strings can get
        quite big.
        """
        return (
            self.name == other.name
            and self.dtype == other.dtype
            and self.pandas_kwargs == other.pandas_kwargs
        )

    def load_client(self, dir_path: Path):
        """Load a client instance from the given directory."""
        client_path = dir_path / f"{self.name}/{self.dtype}/client.zip"

        if client_path.is_file():
            return fhe.Client.load(client_path)

        raise ValueError(f"Client file {client_path} does not exist.")

    def load_server(self, dir_path: Path):
        """Load a server instance from the given directory."""
        server_path = dir_path / f"{self.name}/{self.dtype}/server.zip"

        if server_path.is_file():
            return fhe.Server.load(server_path)

        raise ValueError(f"Server file {server_path} does not exist.")

    def get_encrypt_config(self):
        """Get the encrypt config for this operator."""
        return self.op_to_encrypt_config[self.name]

    def get_supported_min_max(self):
        """Get the minimum and maximum values allowed for this operator."""
        return self.dtype_to_min_max[self.dtype]

    def generate_encrypted_nan_value(self, client: fhe.Client):
        """Generate and return an encrypted 0, representing a NaN value."""
        self.encrypted_nan = encrypt_value(0, client, **self.get_encrypt_config())
        return self.encrypted_nan

    def retrieve_evaluation_keys(self, client: fhe.Client):
        """Retrieve and store the evaluation keys from the given client."""
        self.evaluation_keys = client.evaluation_keys
        return self.evaluation_keys

    def to_dict(self):
        """Serialize the instance to a dictionary."""
        encrypted_nan = (
            serialize_value(self.encrypted_nan) if self.encrypted_nan is not None else None
        )
        evaluation_keys = (
            serialize_evaluation_keys(self.evaluation_keys)
            if self.evaluation_keys is not None
            else None
        )

        # Get the required parameters as well as the optional Pandas kwargs
        output_params = list(self.required_op_parameters) + ["pandas_kwargs"]

        output_dict = {
            required_op_parameter: getattr(self, required_op_parameter)
            for required_op_parameter in output_params
        }
        output_dict["encrypted_nan"] = encrypted_nan
        output_dict["evaluation_keys"] = evaluation_keys

        return output_dict

    @classmethod
    def from_dict(cls, dict_to_load: Dict):
        """Load an instance from a dictionary."""
        encrypted_nan = (
            deserialize_value(dict_to_load["encrypted_nan"])
            if dict_to_load["encrypted_nan"] is not None
            else None
        )
        evaluation_keys = (
            deserialize_evaluation_keys(dict_to_load["evaluation_keys"])
            if dict_to_load["evaluation_keys"] is not None
            else None
        )

        # Get the required parameters as well as the optional Pandas kwargs
        input_params = list(EncryptedDataFrameOperator.required_op_parameters) + ["pandas_kwargs"]

        input_dict = {
            required_op_parameter: dict_to_load[required_op_parameter]
            for required_op_parameter in input_params
        }

        input_dict["encrypted_nan"] = encrypted_nan
        input_dict["evaluation_keys"] = evaluation_keys

        return cls(**input_dict)
