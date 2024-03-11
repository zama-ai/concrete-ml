from typing import Dict, List

import numpy

from .utils import deserialize_elementwise, serialize_elementwise


class EncryptedDataFrame:
    """Define an encrypted data-frame that can be serialized."""

    def __init__(self, encrypted_values: numpy.ndarray, column_names: List[str]):
        self.encrypted_values = encrypted_values
        self.column_names = list(column_names)
        self.column_names_to_index = {name: index for index, name in enumerate(column_names)}

    def to_dict(self):
        """Serialize the instance to a dictionary."""
        encrypted_values = serialize_elementwise(self.encrypted_values)

        # A Numpy array is not serializable using JSON so we need to convert to a list
        output_dict = {
            "encrypted_values": encrypted_values.tolist(),
            "column_names": self.column_names,
        }

        return output_dict

    @classmethod
    def from_dict(cls, dict_to_load: Dict):
        """Load an instance from a dictionary."""
        encrypted_values = deserialize_elementwise(dict_to_load["encrypted_values"])
        column_names = dict_to_load["column_names"]

        return cls(encrypted_values, column_names)
