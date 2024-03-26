import numpy
import pandas
from pandas.core.reshape.merge import _MergeOperation

UNSUPPORTED_PANDAS_PARAMETERS = {
    "merge": [
        "left_on",
        "right_on",
        "left_index",
        "right_index",
        "sort",
        "copy",
        "indicator",
        "validate",
    ],
}


def check_parameter_is_supported(param, param_name, condition, operator):
    if param_name in UNSUPPORTED_PANDAS_PARAMETERS[operator] and not condition:
        raise ValueError(f"Parameter '{param_name}' is not currently supported. Got {param}.")


def check_coherence_selected_column_for_merge(df_left, df_right, on):
    on_left, on_right = df_left.dtype_mappings[on], df_right.dtype_mappings[on]
    dtype_left, dtype_right = numpy.dtype(on_left["dtype"]), numpy.dtype(on_right["dtype"])

    if dtype_left == dtype_right:
        if numpy.issubdtype(dtype_left, numpy.floating):
            raise ValueError(
                f"Column '{on}' cannot be selected for merging both data-frames because it has a "
                f"floating dtype ({dtype_left})"
            )
        elif dtype_left == "object":
            str_mapping_left = on_left["str_to_int"]
            str_mapping_right = on_right["str_to_int"]

            # TODO: add hash
            if str_mapping_left != str_mapping_right:
                raise ValueError(
                    f"Mappings for string values in both common column '{on}' do not match."
                )
    else:
        raise ValueError(
            f"Dtypes of both common column '{on}' do not match. Got {dtype_left} (left) and "
            f"{dtype_right} (right)."
        )


def encrypted_left_right_join(
    df_left,
    df_right,
    server,
    how,
    on,
):
    """Execute a left join using encrypted data-frames with Pandas kwargs.

    For now, only a left and right join merge is supported.
    """

    allowed_how = ["left", "right"]
    assert how in allowed_how, f"Parameter 'how' must be in {allowed_how}. Got {how}."

    if how == "right":
        df_left, df_right = df_right, df_left

    array_joined = []

    # _df_clear won't be accessible on the server's side, so two options :
    # - we define an empty data-frame of the same shape when loading
    # - we store and save/load the shapes using new class attributes
    n_rows_left = df_left.encrypted_values.shape[0]
    n_columns_right = df_right.encrypted_values.shape[1]

    # Retrieve the left and right column's index on which keys to merge
    left_key_column_index = df_left.column_names_to_index[on]
    right_key_column_index = df_right.column_names_to_index[on]

    # Loop over the left data frame's number of rows (which will become the joined data frame's
    # number of rows)
    n_rows_right = df_right.encrypted_values.shape[0]
    for i_left in range(n_rows_left):

        # For left merge, all left values are exactly equal to the left data frame
        array_joined_i_left = df_left.encrypted_values[i_left, :]

        if how == "right":
            array_joined_i_left = numpy.delete(array_joined_i_left, left_key_column_index, axis=0)

        left_row_to_join = array_joined_i_left.tolist()

        # Retrieve the left data frame's key to merge on
        left_key = df_left.encrypted_values[i_left, left_key_column_index]

        right_row_to_join = []

        # Loop over the right data-frame's number of columns
        for j_right in range(n_columns_right):

            # Skip the right's index column
            if j_right == right_key_column_index:
                continue

            # Default value is NaN
            right_value_to_join = df_right.encrypted_nan

            # Loop over the right data-frame's number of rows in order to check if one row's key
            # matches the on-going left key
            for i_right in range(n_rows_right):

                # Retrieve the right data-frame's value to sum if both keys match
                value_to_put_right = df_right.encrypted_values[i_right, j_right]

                # Retrieve the right data frame's key to merge on
                right_key = df_right.encrypted_values[i_right, right_key_column_index]

                # Sum the values:
                # - on the first iteration, this sums a 0 (representing a NaN) with the right data-frame's
                # - on the following iterations, the sum is applied between the previous sum's
                # value. If both keys match, this results in this value, else in 0
                # result and the new selected value.
                # At the end of the loop, since keys are unique in the right data-frame, the overall
                # sum was applied on at most a single non-zero value (ie, both keys matched during
                # an iteration only)
                merge_inputs = (right_value_to_join, value_to_put_right, left_key, right_key)

                # TODO: how to use evaluation_keys ?
                right_value_to_join = server.run(
                    *merge_inputs, evaluation_keys=df_left.evaluation_keys
                )

            right_row_to_join.append(right_value_to_join)

        # For left merge, the remaining right values are either 0 (NaN) or the right data-frame's
        # values for which the associated key matched with the left key
        if how == "left":
            joined_row = left_row_to_join + right_row_to_join
        else:
            joined_row = right_row_to_join + left_row_to_join

        array_joined.append(joined_row)

    array_joined = numpy.array(array_joined)

    if how == "right":
        array_joined = numpy.hstack(
            (
                array_joined[:, :right_key_column_index],
                df_left.encrypted_values[:, left_key_column_index : left_key_column_index + 1],
                array_joined[:, right_key_column_index:],
            ),
        )

    return array_joined


def encrypted_merge(
    df_left,
    df_right,
    server,
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
    validate=None,
):
    """Execute a merge using encrypted data-frames with Pandas kwargs.

    For now, only a left and right join merge is supported. Additionally, default value for
    parameter 'how' is different than in Pandas.
    """
    # Raise errors for unsupported parameters
    for param, param_name, condition in [
        (left_on, "left_on", left_on is None),
        (right_on, "right_on", right_on is None),
        (left_index, "left_index", left_index == False),
        (right_index, "right_index", right_index == False),
        (sort, "sort", sort == False),
        (copy, "copy", copy is None),
        (indicator, "indicator", indicator == False),
        (validate, "validate", validate is None),
    ]:
        check_parameter_is_supported(param, param_name, condition, "merge")

    # Retrieve the input column names and build empty data-frames based on them
    # Insert
    empty_df_left = pandas.DataFrame(index=range(1), columns=df_left.column_names)
    empty_df_right = pandas.DataFrame(index=range(1), columns=df_right.column_names)

    # Check input validation
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

    # Compute the expected joined columns
    empty_df_joined = empty_merge_op.get_result()
    joined_column_names = list(empty_df_joined.columns)

    if len(empty_merge_op.join_names) != 1:
        raise ValueError("Merging on 0 or several columns is not currently available.")

    on = empty_merge_op.join_names[0]

    check_coherence_selected_column_for_merge(df_left, df_right, on)

    joined_dtype_mappings = {**df_left.dtype_mappings, **df_right.dtype_mappings}

    if how in ["left", "right"]:

        joined_array = encrypted_left_right_join(df_left, df_right, server, how, on)

        return joined_array, joined_column_names, joined_dtype_mappings

    raise NotImplemented(f"Merge type '{how}' is not currently implemented.")
