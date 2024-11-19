"""Implement Pandas operators in FHE using encrypted data-frames."""

from collections.abc import Hashable
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy
import pandas
from concrete.fhe import Server
from pandas.core.reshape.merge import _MergeOperation

# List of Pandas parameters per operator that are not currently supported
UNSUPPORTED_PANDAS_PARAMETERS = {
    "merge": {
        "left_on": None,
        "right_on": None,
        "left_index": False,
        "right_index": False,
        "sort": False,
        "copy": None,
        "indicator": False,
        "validate": None,
    },
}


def check_parameter_is_supported(parameter: Any, parameter_name: str, operator: str):
    """Check that the given Pandas parameter is supported by the Concrete ML operator.

    Args:
        parameter (Any): The Pandas parameter to consider.
        parameter_name (str): The Pandas parameter's name.
        operator (str): The Concrete ML operator to check.

    Raises:
        ValueError: If the parameter is not supported by the operator.
    """
    default_parameter = UNSUPPORTED_PANDAS_PARAMETERS[operator].get(parameter_name, None)

    if parameter is not default_parameter:
        raise ValueError(
            f"Parameter '{parameter_name}' is not currently supported. Got {parameter}."
        )


def check_dtype_of_selected_column_for_merge(left_encrypted, right_encrypted, selected_column: str):
    """Check that the selected column dtype matches between the two encrypted data-frames.

    Args:
        left_encrypted (EncryptedDataFrame): The left encrypted data-frame.
        right_encrypted (EncryptedDataFrame): The right encrypted data-frame.
        selected_column (str): The selected column name, common to both encrypted data-frames.

    Raises:
        ValueError: If both dtypes do not match.
        ValueError: If both dtypes represent floating point values.
        ValueError: If both dtypes represent string values but the mappings do not match.
    """
    # Get the column selected for the merge in both data-frames
    selected_column_left, selected_column_right = (
        left_encrypted.dtype_mappings[selected_column],
        right_encrypted.dtype_mappings[selected_column],
    )

    # Get the columns' initial dtype
    dtype_left, dtype_right = numpy.dtype(selected_column_left["dtype"]), numpy.dtype(
        selected_column_right["dtype"]
    )

    # If both columns' dtype match, check that they are supported
    if dtype_left == dtype_right:

        # If the columns contain floating points, merging is not allowed
        if numpy.issubdtype(dtype_left, numpy.floating):
            raise ValueError(
                f"Column '{selected_column}' cannot be selected for merging both data-frames "
                f"because it has a floating dtype ({dtype_left})"
            )

        # If the columns contain strings, make sure the mappings match
        if dtype_left == "object":
            str_mapping_left = selected_column_left["str_to_int"]
            str_mapping_right = selected_column_right["str_to_int"]

            # Avoid sending string mappings to server, instead use and check hashes
            # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4342
            if str_mapping_left != str_mapping_right:
                raise ValueError(
                    f"Mappings for string values in both common column '{selected_column}' do "
                    "not match."
                )
    else:
        raise ValueError(
            f"Dtypes of both common column '{selected_column}' do not match. Got {dtype_left} "
            f"(left) and {dtype_right} (right)."
        )


# pylint: disable-next=invalid-name
def encrypted_left_right_join(
    left_encrypted,
    right_encrypted,
    server: Server,
    how: str,
    on: Optional[str],  # pylint: disable=invalid-name
) -> numpy.ndarray:
    """Compute a left/right join in FHE between two encrypted data-frames using Pandas parameters.

    Note that for now, only a left and right join is implemented. Additionally, only some Pandas
    parameters are supported, and joining on multiple columns is not available.

    The algorithm benefits from Concrete Python's composability feature. The idea is that for loops
    are done in the clear, meaning positional indexes are not encrypte and only the data is. In the
    case of a left merge, we need to select the encrypted value from the right data-frame for a
    given (left) row and (right) column position. In order to do that, a for loop goes through
    the right rows and runs the FHE circuit in a composable manner. The goal is to basically
    multiply the right column values with a mask which contains a single 1 at the row position where
    the left and right key matches, and then sum everything to retrieve the selected value. The
    main benefit of using composability instead of a dict mult and sum is that it does not require
    to know the number of columns and rows at compilation time. More details can be found in the
    '_development.py' file.

    Args:
        left_encrypted (EncryptedDataFrame): The left encrypted data-frame.
        right_encrypted (EncryptedDataFrame): The right encrypted data-frame.
        server (Server): The Concrete server to use for running the computations in FHE.
        how (str): Type of merge to be performed, one of {'left', 'right'}.
            * left: use only keys from left frame, similar to a SQL left outer join;
            preserve key order.
            * right: use only keys from right frame, similar to a SQL right outer join;
            preserve key order.
        on (Optional[str]): Column name to join on. These must be found in both DataFrames. If it is
            None then this defaults to the intersection of the columns in both DataFrames.

    Returns:
        numpy.ndarray: The values representing the joined encrypted data-frame.
    """
    allowed_how = ["left", "right"]
    assert how in allowed_how, f"Parameter 'how' must be in {allowed_how}. Got {how}."

    # In case of a right merge, swap the input data-frames
    if how == "right":
        left_encrypted, right_encrypted = right_encrypted, left_encrypted

    joined_rows = []

    # Retrieve the left and right column's position on which keys to merge
    left_key_column_position = left_encrypted.column_names_to_position[on]
    right_key_column_position = right_encrypted.column_names_to_position[on]

    # Retrieve the number of useful rows and columns
    n_rows_left = left_encrypted.encrypted_values.shape[0]
    n_columns_right = right_encrypted.encrypted_values.shape[1]
    n_rows_right = right_encrypted.encrypted_values.shape[0]

    # Loop over the left data frame's number of rows (which will become the joined data frame's
    # number of rows)
    for i_left in range(n_rows_left):

        # For left merge, all left values are exactly equal to the left data-frame
        array_joined_i_left = left_encrypted.encrypted_values[i_left, :]

        # In case of a right merge, remove the column containing the keys on which to merge. This
        # avoid unnecessary FHE computations as the output keys will exactly match the one contained
        # in the (initial) left data-frame. The reason why this is needed only for the right merge
        # is because, in Pandas, this selected column is always kept on the output data-frame's
        # left side. The column is manually inserted back at the end of this function
        if how == "right":
            array_joined_i_left = numpy.delete(
                array_joined_i_left, left_key_column_position, axis=0
            )

        left_row_to_join = array_joined_i_left.tolist()

        # Retrieve the left data frame's key to merge on
        left_key = left_encrypted.encrypted_values[i_left, left_key_column_position]

        right_row_to_join = []

        # Loop over the right data-frame's number of columns
        for j_right in range(n_columns_right):

            # Skip the right's index column
            if j_right == right_key_column_position:
                continue

            # Default value is NaN
            right_value_to_join = right_encrypted.encrypted_nan

            # Loop over the right data-frame's number of rows in order to check if one row's key
            # matches the on-going left key
            for i_right in range(n_rows_right):

                # Retrieve the right data-frame's value to sum if both keys match
                value_to_put_right = right_encrypted.encrypted_values[i_right, j_right]

                # Retrieve the right data frame's key to merge on
                right_key = right_encrypted.encrypted_values[i_right, right_key_column_position]

                merge_inputs = (right_value_to_join, value_to_put_right, left_key, right_key)

                # Run the FHE execution:
                # - on the first iteration, this is applied on a 0 (representing a NaN) and the
                #   right data-frame's value
                # - on the following iterations, this is applied between the previous accumulated
                # value and the right data-frame's value.
                # Basically, if both keys match, the function adds the accumulated value with the
                # right data-frame's value. If they don't, it just adds 0 to the accumulated value.
                # In practice, keys only match once throughout this very loop as keys are assumed to
                # be unique on both data-frames.
                right_value_to_join = server.run(
                    *merge_inputs, evaluation_keys=left_encrypted.evaluation_keys
                )

            right_row_to_join.append(right_value_to_join)

        # In case of a right merge, since data-frames wee initially swapped, swap back the values
        # when re-building the joined data-frame
        if how == "right":
            joined_row = right_row_to_join + left_row_to_join
        else:
            joined_row = left_row_to_join + right_row_to_join

        joined_rows.append(joined_row)

    array_joined = numpy.array(joined_rows)

    # In case of a right merge, as mentioned above, the column containing the right keys needs to be
    # manually re-inserted. This avoids unnecessary FHE computations
    if how == "right":
        array_joined = numpy.hstack(
            (
                array_joined[:, :right_key_column_position],
                left_encrypted.encrypted_values[
                    :, left_key_column_position : left_key_column_position + 1
                ],
                array_joined[:, right_key_column_position:],
            ),
        )

    return array_joined


# pylint: disable-next=too-many-arguments, invalid-name
def encrypted_merge(
    left_encrypted,
    right_encrypted,
    server: Server,
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
) -> Tuple[numpy.ndarray, List[str], Dict]:
    """Merge two encrypted data-frames in FHE using Pandas parameters.

    Note that for now, only a left and right join is implemented. Additionally, only some Pandas
    parameters are supported, and joining on multiple columns is not available.

    Pandas documentation for version 2.0 can be found here:
    https://pandas.pydata.org/pandas-docs/version/2.0/reference/api/pandas.DataFrame.merge.html

    Args:
        left_encrypted (EncryptedDataFrame): The left encrypted data-frame.
        right_encrypted (EncryptedDataFrame): The right encrypted data-frame.
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
    # Implement other merge types
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4342
    if how not in ["left", "right"]:
        raise NotImplementedError(f"Merge type '{how}' is not currently implemented.")

    # Support relevant pandas parameters
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4342
    for parameter, parameter_name in [
        (left_on, "left_on"),
        (right_on, "right_on"),
        (left_index, "left_index"),
        (right_index, "right_index"),
        (sort, "sort"),
        (copy, "copy"),
        (indicator, "indicator"),
        (validate, "validate"),
    ]:
        check_parameter_is_supported(parameter, parameter_name, "merge")

    # Build empty Pandas data-frames based on the encrypted data-frames' column names
    empty_df_left = pandas.DataFrame(index=range(1), columns=left_encrypted.column_names)
    empty_df_right = pandas.DataFrame(index=range(1), columns=right_encrypted.column_names)

    # Check that the merge is valid using Pandas' underlying merge operator. This step allows us not
    # to re-implement validation steps for Pandas parameters
    # Additionally, it has the benefit of being able to retrieve useful attributes, like the
    # expected output column names or the column name(s) to merge on in case 'on=None'
    empty_merge_op = _MergeOperation(
        empty_df_left,
        empty_df_right,
        how=how,
        on=on,
        left_on=left_on,
        right_on=right_on,
        left_index=left_index,
        right_index=right_index,
        sort=sort,
        suffixes=suffixes,
        indicator=indicator,
        validate=validate,
    )

    # Retrieve the expected joined column names
    empty_df_joined = empty_merge_op.get_result()
    joined_column_names = list(empty_df_joined.columns)

    # Support multi-column merge
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4342
    if len(empty_merge_op.join_names) != 1:
        raise ValueError("Merging on 0 or several columns is not currently available.")

    # Retrieve the common column name on which to merge
    selected_column = empty_merge_op.join_names[0]

    # Check that the merge is allowed
    check_dtype_of_selected_column_for_merge(left_encrypted, right_encrypted, selected_column)

    # Join the mappings in order to recover strings and floats in post-processing (client side)
    joined_dtype_mappings = {**left_encrypted.dtype_mappings, **right_encrypted.dtype_mappings}

    # Add a way to ensure that 'selected_column' only contains unique values in both data-frames
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4342
    joined_array = encrypted_left_right_join(
        left_encrypted, right_encrypted, server, how, selected_column
    )

    return joined_array, joined_column_names, joined_dtype_mappings
