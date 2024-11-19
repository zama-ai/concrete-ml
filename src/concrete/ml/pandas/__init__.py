"""Public API for encrypted data-frames."""

from collections.abc import Hashable
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

from .client_engine import ClientEngine
from .dataframe import EncryptedDataFrame


def load_encrypted_dataframe(path: Union[Path, str]) -> EncryptedDataFrame:
    """Load a serialized encrypted data-frame.

    Args:
        path (Union[Path, str]): The path to consider for loading the serialized encrypted
            data-frame.

    Returns:
        EncryptedDataFrame: The loaded encrypted data-frame.
    """
    return EncryptedDataFrame.load(path)


# pylint: disable-next=too-many-arguments, invalid-name
def merge(
    left_encrypted: EncryptedDataFrame,
    right_encrypted: EncryptedDataFrame,
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
) -> EncryptedDataFrame:
    """Merge two encrypted data-frames in FHE using Pandas parameters.

    Note that for now, only a left and right join is implemented. Additionally, only some Pandas
    parameters are supported, and joining on multiple columns is not available.

    Pandas documentation for version 2.0 can be found here:
    https://pandas.pydata.org/pandas-docs/version/2.0/reference/api/pandas.DataFrame.merge.html

    Args:
        left_encrypted (EncryptedDataFrame): The left encrypted data-frame.
        right_encrypted (EncryptedDataFrame): The right encrypted data-frame.
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
    return left_encrypted.merge(
        right_encrypted,
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
