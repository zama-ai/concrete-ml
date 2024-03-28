"""Define pre-processing and post-processing steps for encrypted data-frames."""
import copy
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, cast

import numpy
import pandas

from concrete.ml.pandas._client_server import get_min_max_allowed
from concrete.ml.quantization.quantizers import STABILITY_CONST


def compute_scale_zero_point(column: pandas.Series, q_min: int, q_max: int) -> Tuple[float, float]:
    """Compute the scale and zero point to use for quantizing / de-quantizing the given column.

    Note that the scale and zero point are computed so that values are quantized uniformly from
    range [column.min(), column.max()] (float) to range [q_min, q_max] (int).

    Args:
        column (pandas.Series): The column to consider.
        q_min (int): The minimum quantized value to consider.
        q_max (int): The maximum quantized value to consider.

    Returns:
        Tuple[float, float]: The scale and zero-point.
    """
    values_min, values_max = column.min(), column.max()

    if values_max - values_min < STABILITY_CONST:
        if numpy.abs(values_max) < STABILITY_CONST:
            scale = 1.0
            zero_point = -q_min
        else:
            scale = 1 / values_max
            zero_point = 0

    else:
        scale = (q_max - q_min) / (values_max - values_min)

        # TODO: add round for ZP when changing management of NaN values
        # This is because rounding ZP + rounding of quant can make values reach 0, which is not
        # allowed as it is used for representing NaN values
        zero_point = values_min * scale - q_min

    return scale, zero_point


def quant(x: pandas.Series, scale: float, zero_point: float) -> pandas.Series:
    """Quantize the column.

    Args:
        x (pandas.Series): The column to quantize.
        scale (float): The scale to consider.
        zero_point (float): The zero-point to consider.

    Returns:
        pandas.Series: The quantized column.
    """
    return numpy.round(scale * x - zero_point)


def dequant(
    q_x: pandas.Series, scale: float, zero_point: float, dtype: Optional[numpy.dtype] = None
) -> pandas.Series:
    """De-quantize the column.

    Args:
        q_x (pandas.Series): The column to de-quantize.
        scale (float): The scale to consider.
        zero_point (float): The zero-point to consider.
        dtype (Optional[numpy.dtype]): The dtype to use for casting the de-quantized value. Default
            to None, which represents a 'float32' dtype.

    Returns:
        pandas.Series: The de-quantized column.
    """
    x = (q_x + zero_point) / scale

    if dtype is None:
        dtype = cast(numpy.dtype, numpy.float32)

    return x.astype(dtype)


# TODO: provide a way to input string mapping from user
def pre_process_dtypes(pandas_dataframe: pandas.DataFrame) -> Tuple[pandas.DataFrame, Dict]:
    """Pre-process the Pandas data-frame and check that input dtypes and ranges are supported.

    Currently, three input dtypes are supported : integers (within a specific range), floating
    points and objects (strings only, with a maximum amount of unique values). Additionally, NaN
    values are supported.

    Args:
        pandas_dataframe (pandas.DataFrame): The Pandas data-frame to pre-process.

    Raises:
        ValueError: If the values of a column with an integer dtype are out of bounds.
        ValueError: If the amount of unique values found in a column with strings exceeds the
            maximum allowed.
        ValueError: If a column with an 'object' dtype contains other values than strings and NaNs.
        ValueError: If a column has a dtype that is not supported.

    Returns:
        Tuple[pandas.DataFrame, Dict]: The pre-processed Pandas data-frame, as well as the mappings
            to use for recovering float and string values in post-processing.
    """
    pandas_dataframe = copy.copy(pandas_dataframe)

    dtype_mappings: Dict[str, Dict] = defaultdict(dict)

    q_min, q_max = get_min_max_allowed()

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
            is_str = column.apply(lambda x: isinstance(x, str) or not pandas.notna(x)).all()

            if is_str:
                str_to_int = {
                    str_value: i + 1 for i, str_value in enumerate(column.dropna().unique())
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
                f"Column '{column_name}' has dtype '{column_dtype}', which is not currently "
                "supported."
            )

    return pandas_dataframe, dtype_mappings


def pre_process_from_pandas(pandas_dataframe: pandas.DataFrame) -> Tuple[numpy.ndarray, Dict]:
    """Pre-process the Pandas data-frame.

    Args:
        pandas_dataframe (pandas.DataFrame): The Pandas data-frame to pre-process.

    Raises:
        ValueError: If the data-frame's index has not been reset (meaning the index is not a
            RangeIndex object).

    Returns:
        Tuple[numpy.ndarray, Dict]: The pre-processed values that can be encrypted, as well as the
            mappings to use for recovering float and string values in post-processing.

    """
    # TODO: better handle indexes
    if not isinstance(pandas_dataframe.index, pandas.RangeIndex):
        raise ValueError(
            "The data-frame's index has not been reset. Please make sure to not put relevant data "
            "in the index and instead store it in a dedicated column. Encrypted data-frames do not "
            "currently support any index-based operations."
        )

    q_pandas_dataframe, dtype_mappings = pre_process_dtypes(pandas_dataframe)

    # Replace NaN values with 0
    q_pandas_dataframe.fillna(0, inplace=True)

    q_array = q_pandas_dataframe.to_numpy(dtype=numpy.int64)

    return q_array, dtype_mappings


def post_process_dtypes(
    pandas_dataframe: pandas.DataFrame, dtype_mappings: Dict
) -> pandas.DataFrame:
    """Post-process the pandas data-frame.

    Args:
        pandas_dataframe (pandas.DataFrame): The Pandas data-frame to post-process.
        dtype_mappings (Dict): The mappings to consider for recovering float and string values.

    Raises:
        ValueError: If one of the column has an unsupported dtype.

    Returns:
        pandas.DataFrame: The post-processed data-frame.
    """
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
                f"Column '{column_name}' has dtype '{initial_column_dtype}', which is unexpected "
                "and thus not supported."
            )

    return pandas_dataframe


def post_process_to_pandas(
    clear_array: numpy.ndarray, output_column_names: List[str], dtype_mappings: Dict
) -> pandas.DataFrame:
    """Post-process the decrypted values and use them to build a Pandas data-frame.

    Args:
        clear_array (numpy.ndarray): The values to consider.
        output_column_names (List[str]): The column names to consider when building the Pandas
            data-frame.
        dtype_mappings (Dict): The mapping to use for recovering float and string values.

    Returns:
        pandas.DataFrame: The output Pandas data-frame.
    """
    # Replace 0 values by NaN
    clear_array_0_to_nan = numpy.where(clear_array == 0, numpy.nan, clear_array)

    # Convert the array with a pandas data-frame
    pandas_dataframe = pandas.DataFrame(
        clear_array_0_to_nan,
        columns=output_column_names,
    )

    pandas_dataframe = post_process_dtypes(pandas_dataframe, dtype_mappings)

    return pandas_dataframe
