import json
from pathlib import Path
from typing import Dict, Hashable, List, Optional, Sequence, Tuple, Union

import numpy
import pandas
from pandas.io.formats.format import get_dataframe_repr_params

from concrete import fhe
from concrete.ml.pandas._client_server import get_encrypt_config, load_server
from concrete.ml.pandas._operators import encrypted_merge
from concrete.ml.pandas._processing import post_process_to_pandas, pre_process_from_pandas
from concrete.ml.pandas._utils import (
    decrypt_elementwise,
    deserialize_elementwise,
    deserialize_evaluation_keys,
    deserialize_value,
    encrypt_elementwise,
    encrypt_value,
    get_serialized_representation_elementwise,
    serialize_elementwise,
    serialize_evaluation_keys,
    serialize_value,
)

_SERVER = load_server()

CURRENT_API_VERSION = 1


class EncryptedDataFrame:
    """Define an encrypted data-frame framework that supports Pandas operators and parameters."""

    def __init__(
        self,
        encrypted_values: numpy.ndarray,
        encrypted_nan: fhe.Value,
        evaluation_keys: fhe.EvaluationKeys,
        column_names: List[str],
        dtype_mappings: Dict,
        api_version: int,
    ):
        self._encrypted_values = encrypted_values
        self._encrypted_nan = encrypted_nan
        self._evaluation_keys = evaluation_keys

        self._column_names = list(column_names)
        self._column_names_to_index = {name: index for index, name in enumerate(column_names)}
        self._dtype_mappings = dtype_mappings
        self._api_version = api_version

        self._pandas_repr = self._get_pandas_repr()

    @property
    def encrypted_values(self) -> numpy.ndarray:
        """Get the encrypted values.

        Returns:
            numpy.ndarray: The array containing all encrypted values.
        """
        return self._encrypted_values

    @property
    def encrypted_nan(self) -> fhe.Value:
        """Get the encrypted value representing a NaN.

        Returns:
            fhe.Value: The encrypted representation of a NaN.
        """
        return self._encrypted_nan

    @property
    def evaluation_keys(self) -> fhe.EvaluationKeys:
        """Get the evaluation keys.

        Returns:
            fhe.EvaluationKeys: The evaluation keys.
        """
        return self._evaluation_keys

    @property
    def column_names(self) -> List[str]:
        """Get the data-frame's column names in order.

        Returns:
            List[str]: The data-frame's column names in order.
        """
        return self._column_names

    @property
    def column_names_to_index(self) -> Dict[str, int]:
        """Get the mapping between each column's name and its index position.

        Returns:
            Dict[str, int]: Mapping between column names and their position.
        """
        return self._column_names_to_index

    @property
    def dtype_mappings(self) -> Dict:
        """Get the mappings for non-integer dtypes used in pre and post-processing.

        Returns:
            Dict: The mappings for non-integers dtypes.
        """
        return self._dtype_mappings

    @property
    def api_version(self) -> int:
        """Get the API version used when instantiating this instance.

        Returns:
            int: The data-frame's API version.
        """
        return self._api_version

    def _get_pandas_repr(self) -> pandas.DataFrame:
        """Get the Pandas data-frame representing this encrypted data-frame when printing it.

        Returns:
            pandas.DataFrame: The encrypted data-frame's Pandas representation.
        """
        encrypted_values = serialize_elementwise(self._encrypted_values)

        encrypted_values_repr = get_serialized_representation_elementwise(encrypted_values)

        pandas_repr = pandas.DataFrame(encrypted_values_repr, columns=self._column_names)

        return pandas_repr

    def print_scheme(self) -> pandas.DataFrame:
        """Print the encrypted data-frame's scheme.

        The scheme can include column names, dtypes or dtype mappings.

        Returns:
            pandas.DataFrame: The encrypted data-frame's scheme.
        """
        pandas_repr = pandas.DataFrame(self._dtype_mappings, columns=self._column_names)
        return pandas_repr

    def __repr__(self) -> str:
        """Represent the encrypted data-frame as a string.

        Returns:
            str: The encrypted data-frame's string representation.
        """
        repr_params = get_dataframe_repr_params()
        return self._pandas_repr.to_string(index=False, **repr_params)

    def _repr_html_(self) -> str:
        """Represent the encrypted data-frame as a string for HTML.

        This is used for better displaying the data-frame in Jupyter notebooks.

        Returns:
            str: The encrypted data-frame's string representation for HTML.
        """
        return self._pandas_repr.to_html(index=False)

    def merge(
        self,
        other,
        how: str = "left",
        on: Optional[str] = None,
        left_on: Optional[Union[Hashable, Sequence[Hashable]]] = None,
        right_on: Optional[Union[Hashable, Sequence[Hashable]]] = None,
        left_index: bool = False,
        right_index: bool = False,
        sort: bool = False,
        suffixes: Tuple[Optional[str], Optional[str]] = ("_x", "_y"),
        copy: Optional[bool] = None,
        indicator: Union[bool, str] = False,
        validate: Optional[str] = None,
    ) -> Tuple[numpy.ndarray, List[str], Dict]:
        """Merge two encrypted data-frames in FHE using Pandas parameters.

        Note that for now, only a left and right join is implemented. Additionally, only some Pandas
        parameters are supported, and joining on multiple columns is not available.

        Pandas documentation for version 2.0 can be found here:
        https://pandas.pydata.org/pandas-docs/version/2.0/reference/api/pandas.DataFrame.merge.html

        Args:
            other (EncryptedDataFrame): The other encrypted data-frame.
            server (Server): _description_
            server (Server): The Concrete server to use for running the computations in FHE.
            how (str): Type of merge to be performed, one of {'left', 'right'}.
                * left: use only keys from left frame, similar to a SQL left outer join;
                preserve key order.
                * right: use only keys from right frame, similar to a SQL right outer join;
                preserve key order.
            on (Optional[str]): Column name to join on. These must be found in both DataFrames. If it is
                None then this defaults to the intersection of the columns in both DataFrames. Default
                to None.
            left_on (Optional[Union[Hashable, Sequence[Hashable]]]): Currently not supported, please
                keep the default value. Default to None.
            right_on (Optional[Union[Hashable, Sequence[Hashable]]]): Currently not supported, please
                keep the default value. Default to None.
            left_index (bool): Currently not supported, please keep the default value. Default to False.
            right_index (bool): Currently not supported, please keep the default value. Default
                to False.
            sort (bool): Currently not supported, please keep the default value. Default to False.
            suffixes (Tuple[Optional[str], Optional[str]]): A length-2 sequence where each element is
                optionally a string indicating the suffix to add to overlapping column names in `left`
                and `right` respectively. Pass a value of `None` instead of a string to indicate that
                the column name from `left` or `right` should be left as-is, with no suffix. At least
                one of the values must not be None.. Default to ("_x", "_y").
            copy (Optional[bool]): Currently not supported, please keep the default value. Default to
                None.
            indicator (Union[bool, str]): Currently not supported, please keep the default value.
                Default to False.
            validate (Optional[str]): Currently not supported, please keep the default value. Default
                to None.

        Raises:
            ValueError: If the merge is expected to be done on multiple columns.
            NotImplementedError: If parameter 'how' is set to anything else than one
                of {'left', 'right'}.

        Returns:
            Tuple[numpy.ndarray, List[str], Dict]: The values representing the joined encrypted
                data-frame, the associated columns as well as the mappings needed for mapping the
                integers back to their initial string values.
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

        # TODO: better chose which enc nan / eval keys / api version to use
        joined_df = EncryptedDataFrame(
            joined_array,
            self._encrypted_nan,
            self._evaluation_keys,
            joined_column_names,
            joined_dtype_mappings,
            self._api_version,
        )

        return joined_df

    @classmethod
    def encrypt_from_pandas(cls, pandas_dataframe: pandas.DataFrame, client: fhe.Client):
        """Encrypt a Pandas data-frame.

        Args:
            pandas_dataframe (pandas.DataFrame): The Pandas data-frame to encrypt.
            client (fhe.Client): The Concrete client to use for encryption.

        Returns:
            EncryptedDataFrame: The encrypted data-frame.
        """
        pandas_array, dtype_mappings = pre_process_from_pandas(pandas_dataframe)

        # TODO: how to provide encrypt configuration
        encrypted_values = encrypt_elementwise(pandas_array, client, **get_encrypt_config())
        encrypted_nan = encrypt_value(0, client, **get_encrypt_config())

        return cls(
            encrypted_values,
            encrypted_nan,
            client.evaluation_keys,
            pandas_dataframe.columns,
            dtype_mappings,
            CURRENT_API_VERSION,
        )

    def decrypt_to_pandas(self, client: fhe.Client) -> pandas.DataFrame:
        """Decrypt the data-frame and return a Pandas data-frame.

        Args:
            client (fhe.Client): The Concrete client to use for decryption.

        Returns:
            pandas.DataFrame: The Pandas data-frame built on the decrypted values..
        """
        clear_array = decrypt_elementwise(self._encrypted_values, client)
        pandas_dataframe = post_process_to_pandas(
            clear_array, self._column_names, self._dtype_mappings
        )
        return pandas_dataframe

    def _to_dict(self) -> Dict:
        """Serialize the encrypted data-frame as a dictionary.

        Returns:
            Dict: The serialized data-frame.
        """
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
            "api_version": self._api_version,
        }

        return output_dict

    @classmethod
    def _from_dict(cls, dict_to_load: Dict):
        """Load a serialized encrypted data-frame from a dictionary.

        Args:
            dict_to_load (Dict): The serialized encrypted data-frame.

        Returns:
            EncryptedDataFrame: The loaded encrypted data-frame.
        """
        encrypted_values = deserialize_elementwise(dict_to_load["encrypted_values"])
        encrypted_nan = deserialize_value(dict_to_load["encrypted_nan"])
        evaluation_keys = deserialize_evaluation_keys(dict_to_load["evaluation_keys"])
        column_names = dict_to_load["column_names"]
        dtype_mappings = dict_to_load["dtype_mappings"]
        api_version = dict_to_load["api_version"]

        return cls(
            encrypted_values,
            encrypted_nan,
            evaluation_keys,
            column_names,
            dtype_mappings,
            api_version,
        )

    def save(self, path: Union[Path, str]):
        """Save the encrypted data-frame on disk.

        Args:
            path (Union[Path, str]): The path where to save the encrypted data-frame.
        """
        path = Path(path)

        encrypted_df_dict = self._to_dict()
        with path.open("w") as file:
            json.dump(encrypted_df_dict, file)

    @classmethod
    def load(cls, path: Union[Path, str]):
        """Load an encrypted data-frame from disk.

        Args:
            path (Union[Path, str]): The path where to load the encrypted data-frame.

        Returns:
            EncryptedDataFrame: The loaded encrypted data-frame.
        """
        path = Path(path)

        with path.open("r") as file:
            encrypted_df_dict = json.load(file)

        return cls._from_dict(encrypted_df_dict)
