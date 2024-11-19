"""Define the encrypted data-frame framework."""

import json
from collections.abc import Hashable
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
from zipfile import ZIP_STORED, ZipFile

import numpy
import pandas
from pandas.io.formats.format import get_dataframe_repr_params

from concrete import fhe
from concrete.ml.pandas._development import load_server
from concrete.ml.pandas._operators import encrypted_merge
from concrete.ml.pandas._utils import (
    deserialize_elementwise,
    deserialize_evaluation_keys,
    deserialize_value,
    get_serialized_representation_elementwise,
    serialize_elementwise,
    serialize_evaluation_keys,
    serialize_value,
)

_SERVER = load_server()


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
        self._column_names_to_position = {name: index for index, name in enumerate(column_names)}
        self._dtype_mappings = dtype_mappings
        self._api_version = api_version

        # Generate and store the Pandas representation at initialization in order to avoid having
        # to serialize values each time it is needed
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
    def column_names_to_position(self) -> Dict[str, int]:
        """Get the mapping between each column's name and its index position.

        Returns:
            Dict[str, int]: Mapping between column names and their position.
        """
        return self._column_names_to_position

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

        # Encrypted values needs to be serialized in order to be displayed
        encrypted_values = serialize_elementwise(self._encrypted_values)

        # Serialized encrypted values are very long, so we need to only display part of it
        encrypted_values_repr = get_serialized_representation_elementwise(encrypted_values)

        # Display the representation as a Pandas data-frame
        pandas_repr = pandas.DataFrame(encrypted_values_repr, columns=self._column_names)

        return pandas_repr

    def get_schema(self) -> pandas.DataFrame:
        """Get the encrypted data-frame's scheme.

        The scheme can include column names, dtypes or dtype mappings. It is displayed as a Pandas
        data-frame for better readability.

        Returns:
            pandas.DataFrame: The encrypted data-frame's scheme.
        """
        pandas_repr = pandas.DataFrame(self._dtype_mappings, columns=self._column_names)
        return pandas_repr

    # pylint: disable-next=invalid-repr-returned
    def __repr__(self) -> str:
        """Represent the encrypted data-frame as a string.

        Returns:
            str: The encrypted data-frame's string representation.
        """
        # Retrieve Pandas' repr parameters and use them to convert the encrypted data-frame's repr
        # to string
        repr_params = get_dataframe_repr_params()
        pandas_repr_str = self._pandas_repr.to_string(index=False, **repr_params)

        assert isinstance(pandas_repr_str, str)
        return pandas_repr_str

    def _repr_html_(self) -> str:
        """Represent the encrypted data-frame as a string for HTML.

        This is used for better displaying the data-frame in Jupyter notebooks.

        Returns:
            str: The encrypted data-frame's string representation for HTML.
        """
        return self._pandas_repr.to_html(index=False)

    # pylint: disable-next=too-many-arguments, invalid-name
    def merge(
        self,
        other,
        how: str = "left",
        on: Optional[str] = None,  # pylint: disable=invalid-name
        left_on: Optional[Union[Hashable, Sequence[Hashable]]] = None,
        right_on: Optional[Union[Hashable, Sequence[Hashable]]] = None,
        left_index: bool = False,
        right_index: bool = False,
        sort: bool = False,
        suffixes: Tuple[Optional[str], Optional[str]] = ("_x", "_y"),
        copy: Optional[bool] = None,
        indicator: Union[bool, str] = False,
        validate: Optional[str] = None,
    ):
        """Merge two encrypted data-frames in FHE using Pandas parameters.

        Note that for now, only a left and right join is implemented. Additionally, only some Pandas
        parameters are supported, and joining on multiple columns is not available.

        Pandas documentation for version 2.0 can be found here:
        https://pandas.pydata.org/pandas-docs/version/2.0/reference/api/pandas.DataFrame.merge.html

        Args:
            other (EncryptedDataFrame): The other encrypted data-frame.
            how (str): Type of merge to be performed, one of {'left', 'right'}.
                * left: use only keys from left frame, similar to a SQL left outer join;
                preserve key order.
                * right: use only keys from right frame, similar to a SQL right outer join;
                preserve key order.
            on (Optional[str]): Column name to join on. These must be found in both DataFrames. If
                it is None then this defaults to the intersection of the columns in both DataFrames.
                Default to None.
            left_on (Optional[Union[Hashable, Sequence[Hashable]]]): Currently not supported, please
                keep the default value. Default to None.
            right_on (Optional[Union[Hashable, Sequence[Hashable]]]): Currently not supported,
                please keep the default value. Default to None.
            left_index (bool): Currently not supported, please keep the default value. Default to
                False.
            right_index (bool): Currently not supported, please keep the default value. Default
                to False.
            sort (bool): Currently not supported, please keep the default value. Default to False.
            suffixes (Tuple[Optional[str], Optional[str]]): A length-2 sequence where each element
                is optionally a string indicating the suffix to add to overlapping column names in
                `left` and `right` respectively. Pass a value of `None` instead of a string to
                indicate that the column name from `left` or `right` should be left as-is, with no
                suffix. At least one of the values must not be None.. Default to ("_x", "_y").
            copy (Optional[bool]): Currently not supported, please keep the default value. Default
                to None.
            indicator (Union[bool, str]): Currently not supported, please keep the default value.
                Default to False.
            validate (Optional[str]): Currently not supported, please keep the default value.
                Default to None.

        Returns:
            EncryptedDataFrame: The joined encrypted data-frame.
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

        # Once multi-operator is supported, make sure to provide relevant keys and objects
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4342
        joined_df = EncryptedDataFrame(
            joined_array,
            self._encrypted_nan,
            self._evaluation_keys,
            joined_column_names,
            joined_dtype_mappings,
            self._api_version,
        )

        return joined_df

    def _to_dict_and_eval_keys(self) -> Tuple[Dict, fhe.EvaluationKeys]:
        """Serialize the encrypted data-frame as a dictionary and evaluations keys.

        Returns:
            Dict: The serialized data-frame.
            fhe.EvaluationKeys: The serialized evaluations keys.
        """
        # Serialize encrypted values element-wise
        encrypted_values = serialize_elementwise(self._encrypted_values)
        encrypted_nan = serialize_value(self._encrypted_nan)

        evaluation_keys = serialize_evaluation_keys(self._evaluation_keys)

        # Avoid sending column names and string mappings to server, instead use hashes
        # FIXME : https://github.com/zama-ai/concrete-ml-internal/issues/4342
        # Additionally, Numpy arrays are not serializable using JSON so we need to convert them
        # to lists
        output_dict = {
            "encrypted_values": encrypted_values.tolist(),
            "encrypted_nan": encrypted_nan,
            "column_names": self._column_names,
            "dtype_mappings": self._dtype_mappings,
            "api_version": self._api_version,
        }

        return output_dict, evaluation_keys

    @classmethod
    def _from_dict_and_eval_keys(cls, dict_to_load: Dict, evaluation_keys: fhe.EvaluationKeys):
        """Load a serialized encrypted data-frame from a dictionary and evaluations keys.

        Args:
            dict_to_load (Dict): The serialized encrypted data-frame.
            evaluation_keys (fhe.EvaluationKeys): The serialized evaluations keys.

        Returns:
            EncryptedDataFrame: The loaded encrypted data-frame.
        """
        # Deserialize encrypted values element-wise
        encrypted_values = deserialize_elementwise(dict_to_load["encrypted_values"])
        encrypted_nan = deserialize_value(dict_to_load["encrypted_nan"])

        evaluation_keys = deserialize_evaluation_keys(evaluation_keys)

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

        if path.suffix != ".zip":
            path = path.with_suffix(".zip")

        encrypted_df_dict, evaluation_keys = self._to_dict_and_eval_keys()

        encrypted_df_json_bytes = json.dumps(encrypted_df_dict).encode(encoding="utf-8")

        with ZipFile(path, "w", compression=ZIP_STORED, allowZip64=True) as zip_file:
            zip_file.writestr("encrypted_dataframe.json", encrypted_df_json_bytes)
            zip_file.writestr("evaluation_keys", evaluation_keys)

    @classmethod
    def load(cls, path: Union[Path, str]):
        """Load an encrypted data-frame from disk.

        Args:
            path (Union[Path, str]): The path where to load the encrypted data-frame.

        Returns:
            EncryptedDataFrame: The loaded encrypted data-frame.
        """
        path = Path(path)

        if path.suffix != ".zip":
            path = path.with_suffix(".zip")

        with ZipFile(path, "r", compression=ZIP_STORED, allowZip64=True) as zip_file:
            with zip_file.open("encrypted_dataframe.json") as encrypted_df_json_file:
                encrypted_df_json_bytes = encrypted_df_json_file.read()
                encrypted_df_dict = json.loads(encrypted_df_json_bytes)

            with zip_file.open("evaluation_keys") as evaluation_keys_file:
                evaluation_keys = evaluation_keys_file.read()

        return cls._from_dict_and_eval_keys(encrypted_df_dict, evaluation_keys)
