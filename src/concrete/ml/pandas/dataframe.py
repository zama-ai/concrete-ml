import json
from pathlib import Path
from typing import Dict, List

import numpy
import pandas

from concrete import fhe
from concrete.ml.pandas._operators import encrypted_merge
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
from concrete.ml.pandas.client_server import _SERVER_PATH, _get_encrypt_config, _get_min_max_allowed

_SERVER = fhe.Server.load(_SERVER_PATH)


def _validate_pandas_df(pandas_df: pandas.DataFrame, min_value: int, max_value: int):
    """Check that the data-frame only contains values between the given min/max."""

    non_integer_columns = list(
        pandas_df.columns[~pandas_df.dtypes.apply(lambda x: numpy.issubdtype(x, numpy.integer))]
    )
    if non_integer_columns:
        raise ValueError(
            f"Columns {non_integer_columns} contain non-integer values, which is currently noy "
            "allowed."
        )

    columns_less_than_min = (pandas_df < min_value).any()
    column_names_less_than_min = columns_less_than_min[columns_less_than_min].index.tolist()

    if column_names_less_than_min:
        raise ValueError(
            f"Columns {column_names_less_than_min} contain values less than {min_value}, "
            "which is currently not allowed."
            + " This is because 0 values are used to represent NaN values for FHE computations."
            * (min_value <= 1)
        )

    columns_greater_than_max = (pandas_df > max_value).any()
    column_names_greater_than_max = columns_greater_than_max[
        columns_greater_than_max
    ].index.tolist()

    if column_names_greater_than_max:
        raise ValueError(
            f"Columns {column_names_greater_than_max} contain values greater than {max_value}, "
            "which is currently not allowed."
        )


def _pre_process_from_pandas(pandas_df: pandas.DataFrame) -> numpy.ndarray:
    """Pre-process the Pandas data-frame."""
    # Make sure the given data-frame only contains integer values between the allowed min and max
    # Additionally, forbid 0 values as they are then used to represent encrypted NaN values
    min, max = _get_min_max_allowed()
    _validate_pandas_df(pandas_df, min, max)

    # Replace NaN values with 0
    pandas_df.fillna(0, inplace=True)

    pandas_array = pandas_df.to_numpy()

    return pandas_array


def _post_process_to_pandas(clear_array: numpy.ndarray, output_column_names: List[str]):
    """Post-process the server's outputs and build a Pandas data-frame from them."""
    # Replace 0 values by NaN
    clear_array_0_to_nan = numpy.where(clear_array == 0, numpy.nan, clear_array)

    # Convert the array with a pandas data-frame
    df = pandas.DataFrame(
        clear_array_0_to_nan,
        columns=output_column_names,
    )
    return df


class EncryptedDataFrame:
    """Define an encrypted data-frame that can be serialized."""

    def __init__(
        self,
        encrypted_values: numpy.ndarray,
        encrypted_nan: fhe.Value,
        evaluation_keys: fhe.EvaluationKeys,
        column_names: List[str],
    ):
        self.encrypted_values = encrypted_values
        self.encrypted_nan = encrypted_nan
        self.evaluation_keys = evaluation_keys
        self.column_names = list(column_names)
        self.column_names_to_index = {name: index for index, name in enumerate(column_names)}

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
        joined_array, joined_column_names = encrypted_merge(
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
            joined_array, self.encrypted_nan, self.evaluation_keys, joined_column_names
        )

        return joined_df

    @classmethod
    def encrypt_from_pandas(cls, pandas_df, client, evaluation_keys):
        pandas_array = _pre_process_from_pandas(pandas_df)

        # TODO: how to provide encrypt configuration
        encrypted_values = encrypt_elementwise(pandas_array, client, **_get_encrypt_config())
        encrypted_nan = encrypt_value(0, client, **_get_encrypt_config())

        return cls(encrypted_values, encrypted_nan, evaluation_keys, pandas_df.columns)

    @classmethod
    def encrypt_from_csv(cls, file_path, client, evaluation_keys, **pandas_kwargs):
        pandas_df = pandas.read_csv(file_path, **pandas_kwargs)
        return cls.encrypt_from_pandas(pandas_df, client, evaluation_keys)

    def decrypt_to_pandas(self, client):
        clear_array = decrypt_elementwise(self.encrypted_values, client)
        pandas_df = _post_process_to_pandas(clear_array, self.column_names)
        return pandas_df

    def decrypt_to_csv(self, file_path, **pandas_kwargs):
        pandas_df = self.decrypt_to_pandas()
        return pandas_df.to_csv(file_path, **pandas_kwargs)

    def to_dict(self):
        """Serialize the instance to a dictionary."""
        encrypted_values = serialize_elementwise(self.encrypted_values)
        encrypted_nan = serialize_value(self.encrypted_nan)
        evaluation_keys = serialize_evaluation_keys(self.evaluation_keys)

        # A Numpy array is not serializable using JSON so we need to convert to a list
        output_dict = {
            "encrypted_values": encrypted_values.tolist(),
            "encrypted_nan": encrypted_nan,
            "evaluation_keys": evaluation_keys,
            "column_names": self.column_names,
        }

        return output_dict

    @classmethod
    def from_dict(cls, dict_to_load: Dict):
        """Load an instance from a dictionary."""
        encrypted_values = deserialize_elementwise(dict_to_load["encrypted_values"])
        encrypted_nan = deserialize_value(dict_to_load["encrypted_nan"])
        evaluation_keys = deserialize_evaluation_keys(dict_to_load["evaluation_keys"])
        column_names = dict_to_load["column_names"]

        return cls(encrypted_values, encrypted_nan, evaluation_keys, column_names)

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
