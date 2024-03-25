import copy
from collections import defaultdict
from typing import List

import numpy
import pandas

from concrete.ml.pandas.client_server import _get_min_max_allowed


def compute_scale_zero_point(column, q_min, q_max):
    values_min, values_max = column.min(), column.max()
    scale = (q_max - q_min) / (values_max - values_min)

    # TODO: add round for ZP when changing management of NaN values
    # This is because rounding ZP + rounding of quant can make values reach 0, which is not allowed
    # as it's used for representing NaN values
    zero_point = values_min * scale - q_min
    return scale, zero_point


def quant(x, scale, zero_point):
    return numpy.round(scale * x - zero_point).astype(numpy.int64)


def dequant(q_x, scale, zero_point, dtype=None):
    x = (q_x + zero_point) / scale

    if dtype is None:
        dtype = numpy.float32

    return x.astype(dtype)


def pre_process_dtypes(pandas_dataframe):
    pandas_dataframe = copy.copy(pandas_dataframe)

    dtype_mappings = defaultdict(dict)

    q_min, q_max = _get_min_max_allowed()

    for column_name in pandas_dataframe.columns:
        column = pandas_dataframe[column_name]
        column_dtype = column.dtype

        dtype_mappings[column_name]["dtype"] = str(column_dtype)

        if numpy.issubdtype(column_dtype, numpy.integer):
            out_of_bounds = (column < q_min).any() or (column > q_max).any()

            if out_of_bounds:
                raise ValueError(
                    f"Column '{column_name}' (dtype={column_dtype}) contains values that are out "
                    f"of bounds. Expected values to be in interval [min={q_min}, max={q_max}], but "
                    f"found [min={column.min()}, max={column.max()}]."
                )

        elif numpy.issubdtype(column_dtype, numpy.floating):
            scale, zero_point = compute_scale_zero_point(column, q_min, q_max)

            q_column = quant(column, scale, zero_point)

            pandas_dataframe[column_name] = q_column

            dtype_mappings[column_name]["scale"] = scale
            dtype_mappings[column_name]["zero_point"] = zero_point

        elif column_dtype == "object":
            is_str = column.apply(lambda x: isinstance(x, str) or numpy.isnan(x)).all()

            if is_str:
                str_to_int = {
                    str_value: i + 1
                    for i, str_value in enumerate(column.unique())
                    if isinstance(str_value, str)
                }

                n_unique_values = max(str_to_int.values())
                if n_unique_values > q_max:
                    raise ValueError(
                        f"Column '{column_name}' (dtype={column_dtype}) contains too many unique "
                        f"values. Expected {q_max} unique values at most, got {n_unique_values}."
                    )

                q_column = column.map(str_to_int)

                pandas_dataframe[column_name] = q_column

                dtype_mappings[column_name]["str_to_int"] = str_to_int

                # TODO: add hash

            else:
                raise ValueError(
                    f"Column '{column_name}' (dtype={column_dtype}) contains non-string values, "
                    "which is not currently supported."
                )

        else:
            raise ValueError(
                f"Column '{column_name}' has dtype '{column_dtype})', which is not currently "
                "supported."
            )

    return pandas_dataframe, dtype_mappings


def post_process_dtypes(pandas_dataframe, dtype_mappings):
    pandas_dataframe = copy.copy(pandas_dataframe)

    for column_name in pandas_dataframe.columns:
        q_column = pandas_dataframe[column_name]
        initial_column_dtype = numpy.dtype(dtype_mappings[column_name]["dtype"])

        if numpy.issubdtype(initial_column_dtype, numpy.integer):
            if not q_column.isna().any():
                pandas_dataframe[column_name] = q_column.astype(initial_column_dtype)

        elif numpy.issubdtype(initial_column_dtype, numpy.floating):
            scale = dtype_mappings[column_name]["scale"]
            zero_point = dtype_mappings[column_name]["zero_point"]
            column = dequant(q_column, scale, zero_point, dtype=initial_column_dtype)

            pandas_dataframe[column_name] = column

        elif initial_column_dtype == "object":
            string_to_int = dtype_mappings[column_name]["str_to_int"]
            int_to_string = {v: k for k, v in string_to_int.items()}

            column = q_column.map(int_to_string)

            pandas_dataframe[column_name] = column

        else:
            raise ValueError(
                f"Column '{column_name}' has expected dtype '{initial_column_dtype}', which is "
                f"unexpected."
            )

    return pandas_dataframe


def pre_process_from_pandas(pandas_dataframe: pandas.DataFrame) -> numpy.ndarray:
    """Pre-process the Pandas data-frame."""
    # TODO: better handle indexes
    if not isinstance(pandas_dataframe.index, pandas.RangeIndex):
        raise ValueError(
            "The data-frame's index has not been reset. Please make sure to not put relevant data "
            "in the index and instead store it in a dedicated column. Encrypted data-frames do not "
            "currently support any index-based operations."
        )

    pandas_dataframe, dtype_mappings = pre_process_dtypes(pandas_dataframe)

    # Replace NaN values with 0
    pandas_dataframe.fillna(0, inplace=True)

    pandas_array = pandas_dataframe.to_numpy()

    return pandas_array, dtype_mappings


def post_process_to_pandas(
    clear_array: numpy.ndarray, output_column_names: List[str], dtype_mappings
):
    """Post-process the server's outputs and build a Pandas data-frame from them."""
    # Replace 0 values by NaN
    clear_array_0_to_nan = numpy.where(clear_array == 0, numpy.nan, clear_array)

    # Convert the array with a pandas data-frame
    pandas_dataframe = pandas.DataFrame(
        clear_array_0_to_nan,
        columns=output_column_names,
    )

    pandas_dataframe = post_process_dtypes(pandas_dataframe, dtype_mappings)

    return pandas_dataframe
