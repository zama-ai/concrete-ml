import json
from pathlib import Path
from typing import Dict, List

import numpy
import pandas

from concrete import fhe
from concrete.ml.pandas._operators import encrypted_merge
from concrete.ml.pandas._processing import post_process_to_pandas, pre_process_from_pandas
from concrete.ml.pandas._utils import (
    decrypt_elementwise,
    deserialize_elementwise,
    deserialize_evaluation_keys,
    deserialize_value,
    encrypt_elementwise,
    encrypt_value,
    serialize_elementwise,
    serialize_evaluation_keys,
    serialize_value,
)
from concrete.ml.pandas.client_server import _SERVER_PATH, _get_encrypt_config

_SERVER = fhe.Server.load(_SERVER_PATH)


class EncryptedDataFrame:
    """Define an encrypted data-frame that can be serialized."""

    def __init__(
        self,
        encrypted_values: numpy.ndarray,
        encrypted_nan: fhe.Value,
        evaluation_keys: fhe.EvaluationKeys,
        column_names: List[str],
        dtype_mappings: Dict,
    ):
        self._encrypted_values = encrypted_values
        self._encrypted_nan = encrypted_nan
        self._evaluation_keys = evaluation_keys

        self._column_names = list(column_names)
        self._column_names_to_index = {name: index for index, name in enumerate(column_names)}
        self._dtype_mappings = dtype_mappings

    @property
    def encrypted_values(self):
        return self._encrypted_values

    @property
    def encrypted_nan(self):
        return self._encrypted_nan

    @property
    def evaluation_keys(self):
        return self._evaluation_keys

    @property
    def column_names(self):
        return self._column_names

    @property
    def column_names_to_index(self):
        return self._column_names_to_index

    @property
    def dtype_mappings(self):
        return self._dtype_mappings

    def join(
        self,
        other,
        on=None,
        how="left",
        lsuffix="",
        rsuffix="",
        sort=False,
        validate="1:1",
    ):
        """Execute a Pandas join operator using encrypted data-frames.

        Pandas documentation for version 2.0 can be found here:
        https://pandas.pydata.org/pandas-docs/version/2.0/reference/api/pandas.DataFrame.join.html

        For now, only a left and right join merge is supported. Additionally, default values for
        parameter 'validate' is different than in Pandas.
        """
        # Similar to pandas/core/frame.py
        if how == "cross":
            return self.merge(
                other,
                how=how,
                on=on,
                suffixes=(lsuffix, rsuffix),
                sort=sort,
                validate=validate,
            )
        return self.merge(
            other,
            left_on=on,
            how=how,
            left_index=on is None,
            right_index=True,
            suffixes=(lsuffix, rsuffix),
            sort=sort,
            validate=validate,
        )

    def merge(
        self,
        other,
        how="left",
        on=None,
        left_on=None,
        right_on=None,
        left_index=False,
        right_index=False,
        sort=False,
        suffixes=("_x", "_y"),
        copy=None,
        indicator=False,
        validate="1:1",
    ):
        """Execute a merge using encrypted data-frames with Pandas kwargs.

        Pandas documentation for version 2.0 can be found here:
        https://pandas.pydata.org/pandas-docs/version/2.0/reference/api/pandas.DataFrame.merge.html

        For now, only a left and right join merge is supported. Additionally, default values for
        parameters 'how' and 'validate' are different than in Pandas.
        """
        joined_array, joined_column_names, joined_dtype_mappings = encrypted_merge(
            self,
            other,
            _SERVER,
            how=how,
            on=on,
            left_on=left_on,
            right_on=right_on,
            left_index=left_index,
            right_index=right_index,
            sort=sort,
            suffixes=suffixes,
            copy=copy,
            indicator=indicator,
            validate=validate,
        )

        joined_df = EncryptedDataFrame(
            joined_array,
            self._encrypted_nan,
            self._evaluation_keys,
            joined_column_names,
            joined_dtype_mappings,
        )

        return joined_df

    @classmethod
    def encrypt_from_pandas(cls, pandas_dataframe, client, evaluation_keys):
        pandas_array, dtype_mappings = pre_process_from_pandas(pandas_dataframe)

        # TODO: how to provide encrypt configuration
        encrypted_values = encrypt_elementwise(pandas_array, client, **_get_encrypt_config())
        encrypted_nan = encrypt_value(0, client, **_get_encrypt_config())

        return cls(
            encrypted_values,
            encrypted_nan,
            evaluation_keys,
            pandas_dataframe.columns,
            dtype_mappings,
        )

    @classmethod
    def encrypt_from_csv(cls, file_path, client, evaluation_keys, **pandas_kwargs):
        pandas_dataframe = pandas.read_csv(file_path, **pandas_kwargs)
        return cls.encrypt_from_pandas(pandas_dataframe, client, evaluation_keys)

    def decrypt_to_pandas(self, client):
        clear_array = decrypt_elementwise(self._encrypted_values, client)
        pandas_dataframe = post_process_to_pandas(
            clear_array, self._column_names, self._dtype_mappings
        )
        return pandas_dataframe

    def decrypt_to_csv(self, file_path, **pandas_kwargs):
        pandas_dataframe = self.decrypt_to_pandas()
        return pandas_dataframe.to_csv(file_path, **pandas_kwargs)

    def to_dict(self):
        """Serialize the instance to a dictionary."""
        encrypted_values = serialize_elementwise(self._encrypted_values)
        encrypted_nan = serialize_value(self._encrypted_nan)
        evaluation_keys = serialize_evaluation_keys(self._evaluation_keys)

        # A Numpy array is not serializable using JSON so we need to convert to a list
        output_dict = {
            "encrypted_values": encrypted_values.tolist(),
            "encrypted_nan": encrypted_nan,
            "evaluation_keys": evaluation_keys,
            "column_names": self._column_names,
            "dtype_mappings": self._dtype_mappings,
        }

        return output_dict

    @classmethod
    def from_dict(cls, dict_to_load: Dict):
        """Load an instance from a dictionary."""
        encrypted_values = deserialize_elementwise(dict_to_load["encrypted_values"])
        encrypted_nan = deserialize_value(dict_to_load["encrypted_nan"])
        evaluation_keys = deserialize_evaluation_keys(dict_to_load["evaluation_keys"])
        column_names = dict_to_load["column_names"]
        dtype_mappings = dict_to_load["dtype_mappings"]

        return cls(encrypted_values, encrypted_nan, evaluation_keys, column_names, dtype_mappings)

    def to_json(self, file_path):
        file_path = Path(file_path)

        encrypted_df_dict = self.to_dict()
        with file_path.open("w") as file:
            json.dump(encrypted_df_dict, file)

    @classmethod
    def from_json(cls, file_path):
        file_path = Path(file_path)

        with file_path.open("r") as file:
            encrypted_df_dict = json.load(file)

        return cls.from_dict(encrypted_df_dict)
