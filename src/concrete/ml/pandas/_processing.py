"""Define pre-processing and post-processing steps for encrypted data-frames."""

import copy
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy
import pandas

from concrete.ml.pandas._development import get_min_max_allowed
from concrete.ml.quantization.quantizers import STABILITY_CONST

SCHEMA_FLOAT_KEYS = ["min", "max"]


def is_str_or_none(column: pandas.Series) -> bool:
    """Determine if the data-frames only contains string and None values or not.

    Args:
        column (pandas.Series): The data-frame to consider.

    Returns:
        bool: If the data-frames only contains string and None values or not.
    """
    return column.apply(lambda x: isinstance(x, str) or not pandas.notna(x)).all()


def compute_scale_zero_point(
    f_min: float, f_max: float, q_min: int, q_max: int
) -> Tuple[float, float]:
    """Compute the scale and zero point to use for quantizing / de-quantizing the given column.

    Note that the scale and zero point are computed so that values are quantized uniformly from
    range [f_min, f_max] (float) to range [q_min, q_max] (int).

    Args:
        f_min (float): The minimum float value observed.
        f_max (float): The maximum float value observed.
        q_min (int): The minimum quantized value to target.
        q_max (int): The maximum quantized value to target.

    Returns:
        Tuple[float, float]: The scale and zero-point.
    """

    # If there si a single float value in the column, the scale and zero-point need to be handled
    # differently
    if f_max - f_min < STABILITY_CONST:

        # If this single float value is 0, make sure it is not quantized to 0
        if numpy.abs(f_max) < STABILITY_CONST:
            scale = 1.0
            zero_point = -q_min

        # Else, quantize it to 1
        else:
            scale = 1 / f_max
            zero_point = 0

    else:
        scale = (q_max - q_min) / (f_max - f_min)

        # Zero-point must be rounded once NaN values are not represented by 0 anymore
        # The issue is that we currently need to avoid quantized values to reach 0, but having a
        # round here + in the 'quant' method can make this happen.
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4342
        # Disable mypy until it is fixed
        zero_point = f_min * scale - q_min  # type: ignore[assignment]

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
    q_x: pandas.Series, scale: float, zero_point: float, dtype: numpy.dtype
) -> pandas.Series:
    """De-quantize the column.

    Args:
        q_x (pandas.Series): The column to de-quantize.
        scale (float): The scale to consider.
        zero_point (float): The zero-point to consider.
        dtype (numpy.dtype): The dtype to use for casting the de-quantized value.

    Returns:
        pandas.Series: The de-quantized column.
    """
    x = (q_x + zero_point) / scale

    return x.astype(dtype)


def check_schema_format(pandas_dataframe: pandas.DataFrame, schema: Optional[Dict] = None) -> None:
    """Check that the given schema has a proper expected format.

    Args:
        pandas_dataframe (pandas.DataFrame): The data-frame associated to the given schema.
        schema (Optional[Dict]): The schema to check, which can be None. Default to None.

    Raises:
        ValueError: If the given schema is not a dict.
        ValueError: If the given schema contains column names that do not appear in the data-frame.
        ValueError: If one of the columns' mapping is not a dict.
    """
    if schema is None:
        return

    if not isinstance(schema, dict):
        raise ValueError(
            "When set, parameter 'schema' must be a dictionary that associates some of the "
            f"data-frame's column names to their value mappings. Got {type(schema)=}"
        )

    column_names = list(pandas_dataframe.columns)

    for column_name, column_mapping in schema.items():
        if column_name not in column_names:
            raise ValueError(
                f"Column name '{column_name}' found in the given schema cannot be found in the "
                f"input data-frame. Expected one of {column_names}"
            )

        if not isinstance(column_mapping, dict):
            raise ValueError(
                f"Mapping for column '{column_name}' is not a dictionary. Got "
                f"{type(column_mapping)=}"
            )


# pylint: disable=too-many-branches, too-many-statements
def pre_process_dtypes(
    pandas_dataframe: pandas.DataFrame, schema: Optional[Dict] = None
) -> Tuple[pandas.DataFrame, Dict]:
    """Pre-process the Pandas data-frame and check that input dtypes and ranges are supported.

    Currently, three input dtypes are supported : integers (within a specific range), floating
    points and objects (strings only, with a maximum amount of unique values). Additionally, NaN
    values are supported.

    Args:
        pandas_dataframe (pandas.DataFrame): The Pandas data-frame to pre-process.
        schema (Optional[Dict]): The input schema to consider. Default to None.

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

    # Get the minimum and maximum quantized values allowed in an encrypted data-frame
    q_min, q_max = get_min_max_allowed()

    # Avoid sending column names to server, instead use hashes
    # FIXME : https://github.com/zama-ai/concrete-ml-internal/issues/4342
    # pylint: disable=too-many-nested-blocks
    for column_name in pandas_dataframe.columns:
        column = pandas_dataframe[column_name]
        column_dtype = column.dtype

        # Store the initial dtype, in order to cast in post-processing
        dtype_mappings[column_name]["dtype"] = str(column_dtype)

        # If the column contains integers, make sure they are not out of bounds
        if numpy.issubdtype(column_dtype, numpy.integer):
            if schema is not None and column_name in schema:
                raise ValueError(
                    f"Column '{column_name}' contains integer values and therefore does not "
                    "require any mappings. Please remove it"
                )

            if column.min() < q_min or column.max() > q_max:
                raise ValueError(
                    f"Column '{column_name}' (dtype={column_dtype}) contains values that are out "
                    f"of bounds. Expected values to be in interval [min={q_min}, max={q_max}], but "
                    f"found [min={column.min()}, max={column.max()}]."
                )

        # If the column contains floats, quantize the values
        elif numpy.issubdtype(column_dtype, numpy.floating):
            if schema is not None and column_name in schema:
                float_min_max = schema[column_name]

                if not all(
                    float_mapping_key in SCHEMA_FLOAT_KEYS
                    for float_mapping_key in float_min_max.keys()
                ):
                    raise ValueError(
                        f"Column '{column_name}' contains float values but the associated mapping "
                        f"does not contain proper keys. Expected {sorted(SCHEMA_FLOAT_KEYS)}, but "
                        f"got {sorted(float_min_max.keys())}"
                    )

                f_min, f_max = float_min_max["min"], float_min_max["max"]

                if column.min() < f_min or column.max() > f_max:
                    raise ValueError(
                        f"Column '{column_name}' (dtype={column_dtype}) contains values that are "
                        f"out of bounds. Expected values to be in interval [min={f_min}, "
                        f"max={f_max}], as determined by the given schema, but found "
                        f"[min={column.min()}, max={column.max()}]."
                    )

            else:
                f_min, f_max = column.min(), column.max()

            scale, zero_point = compute_scale_zero_point(f_min, f_max, q_min, q_max)

            q_column = quant(column, scale, zero_point)

            pandas_dataframe[column_name] = q_column

            # Store the scale and zero point for de-quantization in post-processing
            dtype_mappings[column_name]["scale"] = scale
            dtype_mappings[column_name]["zero_point"] = zero_point

        # If the column contains objects, make sure it is only made of strings or NaN values
        elif column_dtype == "object":
            if is_str_or_none(column):
                if schema is not None and column_name in schema:
                    str_to_int = schema[column_name]

                    column_values = set(column.values) - set([None, numpy.NaN])
                    string_mapping_keys = set(str_to_int.keys())

                    # Allow custom mapping for NaN values once they are not represented by 0 anymore
                    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4342
                    if numpy.NaN in string_mapping_keys:
                        raise ValueError(
                            f"String mapping for column '{column_name}' contains numpy.NaN as a "
                            "key, which is currently forbidden"
                        )

                    forgotten_string_values = column_values - string_mapping_keys

                    if forgotten_string_values:
                        raise ValueError(
                            f"String mapping keys for column '{column_name}' are not considering "
                            "all values from the data-frame. Missing values: "
                            f"{sorted(forgotten_string_values)}"
                        )

                    for string_mapping_key, string_mapping_value in str_to_int.items():
                        if not isinstance(string_mapping_value, int):
                            raise ValueError(
                                f"String mapping values for column '{column_name}' must be "
                                f"integers. Got {type(string_mapping_value)} for key "
                                f"{string_mapping_key}"
                            )

                        if string_mapping_value < q_min or string_mapping_value > q_max:
                            raise ValueError(
                                f"String mapping values for column '{column_name}' are out of "
                                f"bounds. Expected values to be in interval [min={q_min}, "
                                f"max={q_max}] but got {string_mapping_value} for key "
                                f"{string_mapping_key}"
                            )

                    if len(str_to_int.values()) != len(set(str_to_int.values())):
                        raise ValueError(
                            f"String mapping values for column '{column_name}' must be unique. Got "
                            f"{str_to_int.values()}"
                        )

                else:

                    # Build a mapping between the unique strings values and integers
                    str_to_int = {
                        str_value: i + 1 for i, str_value in enumerate(column.dropna().unique())
                    }

                # Make sure the number of unique values do not goes over the maximum integer value
                # allowed in an encrypted data-frame
                n_unique_values = max(str_to_int.values())
                if n_unique_values > q_max:
                    raise ValueError(
                        f"Column '{column_name}' (dtype={column_dtype}) contains too many unique "
                        f"values. Expected {q_max} unique values at most, got {n_unique_values}."
                    )

                q_column = column.map(str_to_int)

                pandas_dataframe[column_name] = q_column

                # Store the mapping in order to recover the initial values in post-processing
                # Avoid sending string mappings to server, instead use and check hashes
                # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4342
                dtype_mappings[column_name]["str_to_int"] = str_to_int

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


def pre_process_from_pandas(
    pandas_dataframe: pandas.DataFrame, schema: Optional[Dict] = None
) -> Tuple[numpy.ndarray, Dict]:
    """Pre-process the Pandas data-frame.

    Args:
        pandas_dataframe (pandas.DataFrame): The Pandas data-frame to pre-process.
        schema (Optional[Dict]): The input schema to consider. Default to None.

    Raises:
        ValueError: If the data-frame's index has not been reset (meaning the index is not a
            RangeIndex object).

    Returns:
        Tuple[numpy.ndarray, Dict]: The pre-processed values that can be encrypted, as well as the
            mappings to use for recovering float and string values in post-processing.

    """
    # Support Index of Pandas data-frames
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4342
    if not isinstance(pandas_dataframe.index, pandas.RangeIndex):
        raise ValueError(
            "The data-frame's index has not been reset. Please make sure to not put relevant data "
            "in the index and instead store it in a dedicated column. Encrypted data-frames do not "
            "currently support any index-based operations."
        )

    # Check that values are supported and build the mappings
    q_pandas_dataframe, dtype_mappings = pre_process_dtypes(pandas_dataframe, schema=schema)

    # Replace NaN values with 0
    # Remove this once NaN values are not represented by 0 anymore
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4342
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

        # Retrieve the column's initial dtype
        initial_column_dtype = numpy.dtype(dtype_mappings[column_name]["dtype"])

        # If the column contained integers, cast the values to the initial dtype, unless it contains
        # some NaN values (as they are represented with a float dtype)
        if numpy.issubdtype(initial_column_dtype, numpy.integer):
            if not q_column.isna().any():
                pandas_dataframe[column_name] = q_column.astype(initial_column_dtype)

        # If the column contained floats, de-quantize the values using the stored scale and
        # zero-point
        elif numpy.issubdtype(initial_column_dtype, numpy.floating):
            scale = dtype_mappings[column_name]["scale"]
            zero_point = dtype_mappings[column_name]["zero_point"]

            column = dequant(q_column, scale, zero_point, dtype=initial_column_dtype)

            pandas_dataframe[column_name] = column

        # If the column contained objects (strings), revert the mapping and retrieve the initial
        # values
        elif initial_column_dtype == "object":

            # Revert the mapping from integers to strings
            string_to_int = dtype_mappings[column_name]["str_to_int"]
            int_to_string = {v: k for k, v in string_to_int.items()}

            # Apply the reverted string mapping
            column = q_column.map(int_to_string)

            pandas_dataframe[column_name] = column

        # Else, the data might have been modified since pre-processing
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
    # Remove this once NaN values are not represented by 0 anymore
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4342
    clear_array_0_to_nan = numpy.where(clear_array == 0, numpy.nan, clear_array)

    # Build the joined Pandas data-frame using the de-serialized encrypted data-frame
    pandas_dataframe = pandas.DataFrame(
        clear_array_0_to_nan,
        columns=output_column_names,
    )

    # Post-process the values to match Pandas' output
    pandas_dataframe = post_process_dtypes(pandas_dataframe, dtype_mappings)

    return pandas_dataframe
