import time
from pathlib import Path
from typing import List

import numpy
import pandas
from pandas.core.reshape.merge import _MergeOperation

from concrete import fhe

from .dataframe import EncryptedDataFrame
from .operator import EncryptedDataFrameOperator


class EncryptedDataFrameServer:
    """Server object for running any Pandas ops that are implemented using encrypted data-frames.

    A server provides a list of supported Pandas operators that have been compiled for FHE and
    implemented for supporting encrypted data-frames.

    Running a server means we run a given list of operators in a given order with given inputs.
    For now, we only support a single operator (left join) and the API is not really able to support
    the combination of several operators in a general way.
    """

    def __init__(self, deployment_dir: Path):
        self.ops_to_server = self._load_all_servers(deployment_dir)
        self.supported_ops = self.ops_to_server.keys()
        self.supported_dtype_per_ops = {
            op: dtypes.keys() for op, dtypes in self.ops_to_server.items()
        }

    def _load_servers_per_op(self, deployment_dir: Path, op_dir: Path):
        """Load all server objects (all dtypes) available for the given operator."""
        return {
            dtype_dir.name: EncryptedDataFrameOperator(op_dir.name, dtype_dir.name).load_server(
                deployment_dir
            )
            for dtype_dir in op_dir.iterdir()
            if dtype_dir.is_dir()
        }

    def _load_all_servers(self, deployment_dir: Path):
        """Load all server objects (all ops and dtypes) available."""
        return {
            op_dir.name: self._load_servers_per_op(deployment_dir, op_dir)
            for op_dir in deployment_dir.iterdir()
            if op_dir.is_dir()
        }

    def _check_ops_and_dtypes_are_supported(self, ops: List[EncryptedDataFrameOperator]):
        """Check that all given operators are supported by the server.

        Supporting an operator means that an FHE circuit have been compiled and that a method
        sharing the same name have been implemented.
        """
        ops_with_no_implementation, ops_with_no_server, dtypes_per_ops_with_no_server = [], [], []

        for op in ops:
            # If no method sharing the op's name can be found
            if getattr(self, op.name, None) is None:
                ops_with_no_implementation.append(op.name)

            # If no circuits can be found for this op
            if op.name not in self.supported_ops:
                ops_with_no_server.append(op.name)

            # If no circuits can be found for the op's dtype
            elif op.dtype not in self.supported_dtype_per_ops[op.name]:
                dtypes_per_ops_with_no_server.append((op.name, op.dtype))

        error_message = ""
        if ops_with_no_implementation:
            error_message += (
                f"Operators {ops_with_no_implementation} do not seem to be implemented.\n"
            )

        if ops_with_no_server:
            error_message += f"Server(s) for operators {ops_with_no_server} cannot be found.\n"

        if dtypes_per_ops_with_no_server:
            error_message += (
                f"Server(s) for operators {dtypes_per_ops_with_no_server[0]} with dtype "
                f"{dtypes_per_ops_with_no_server[1]} cannot be found."
            )

    def get_server(self, op_name: str, dtype: str):
        """Retrieve the FHE server object associated with the op's name and dtype."""
        return self.ops_to_server[op_name][dtype]

    def _left_join(
        self,
        evaluation_keys: fhe.EvaluationKeys,
        input_left: EncryptedDataFrame,
        input_right: EncryptedDataFrame,
        encrypted_nan: fhe.Value,
        dtype: str,
        on=None,
        left_on=None,
        right_on=None,
        left_index=False,
        right_index=False,
        sort=False,
        copy=None,
        indicator=False,
        validate="1:1",
    ):
        """Execute a left join using encrypted data-frames with Pandas kwargs.

        Not all Pandas kwargs are currently supported, and some ('sort','indicator') probably will
        never be.
        """
        # Raise errors for unsupported parameters
        if left_on is not None or right_on is not None or left_index or right_index:
            raise ValueError(
                "The following parameters are currently not supported, please use 'how' instead : "
                f"{left_on=}," * left_on
                is not None + f"{right_on=}," * right_on
                is not None + f"{left_index=}," * left_index + f"{right_index=}," * right_index
            )

        if sort:
            raise ValueError("Sorting by the join key is not currently supported.")

        if copy is not None:
            raise ValueError("Parameter 'copy' is not currently supported.")

        if indicator:
            raise ValueError("Parameter 'indicator' is not currently supported.")

        if validate not in ("1:1", "one-to-one"):
            raise ValueError("Indices must be unique in both data-frames.")

        array_joined = []

        # _df_clear won't be accessible on the server's side, so two options :
        # - we define an empty data-frame of the same shape when loading
        # - we store and save/load the shapes using new class attributes
        n_rows_left = input_left.encrypted_values.shape[0]
        n_columns_right = input_right.encrypted_values.shape[1]

        # Retrieve the left and right column's index on which keys to merge
        left_key_column_index = input_left.column_names_to_index[on]
        right_key_column_index = input_right.column_names_to_index[on]

        # Loop over the left data frame's number of rows (which will become the joined data frame's
        # number of rows)
        n_rows_right = input_right.encrypted_values.shape[0]
        for i_left in range(n_rows_left):

            # For left merge, all left values are exactly equal to the left data frame
            array_joined_i_left = input_left.encrypted_values[i_left, :].tolist()

            # Retrieve the left data frame's key to merge on
            left_key = input_left.encrypted_values[i_left, left_key_column_index]

            right_row_to_join = []

            # Loop over the right data-frame's number of columns
            for j_right in range(n_columns_right):

                # Skip the right's index column
                if j_right == right_key_column_index:
                    continue

                # Default value is NaN
                right_value_to_join = encrypted_nan

                # Loop over the right data-frame's number of rows in order to check if one row's key
                # matches the on-going left key
                for i_right in range(n_rows_right):

                    # Retrieve the right data-frame's value to sum if both keys match
                    value_to_put_right = input_right.encrypted_values[i_right, j_right]

                    # Retrieve the right data frame's key to merge on
                    right_key = input_right.encrypted_values[i_right, right_key_column_index]

                    # Sum the values:
                    # - on the first iteration, this sums a 0 (representing a NaN) with the right data-frame's
                    # - on the following iterations, the sum is applied between the previous sum's
                    # value. If both keys match, this results in this value, else in 0
                    # result and the new selected value.
                    # At the end of the loop, since keys are unique in the right data-frame, the overall
                    # sum was applied on at most a single non-zero value (ie, both keys matched during
                    # an iteration only)
                    merge_inputs = (right_value_to_join, value_to_put_right, left_key, right_key)

                    server = self.get_server("merge", dtype)

                    right_value_to_join = server.run(*merge_inputs, evaluation_keys=evaluation_keys)

                right_row_to_join.append(right_value_to_join)
            array_joined.append(array_joined_i_left)

            # For left merge, the remaining right values are either 0 (NaN) or the right data-frame's
            # values for which the associated key matched with the left key
            array_joined_i_left.extend(right_row_to_join)

        array_joined = numpy.array(array_joined)

        return array_joined

    def merge(
        self,
        op: EncryptedDataFrameOperator,
        input_left: EncryptedDataFrame,
        input_right: EncryptedDataFrame,
    ):
        """Execute a merge using encrypted data-frames with Pandas kwargs.

        For now, only a left merge is supported.
        """
        # Retrieve the input column names and build empty data-frames based on them
        # Insert
        empty_df_left = pandas.DataFrame(index=range(1), columns=input_left.column_names)
        empty_df_right = pandas.DataFrame(index=range(1), columns=input_right.column_names)

        # Check input validation
        empty_merge_op = _MergeOperation(empty_df_left, empty_df_right, **op.pandas_kwargs)

        # Compute the expected joined columns
        empty_df_joined = empty_merge_op.get_result()
        joined_column_names = list(empty_df_joined.columns)

        if len(empty_merge_op.join_names) != 1:
            raise ValueError("Merging on 0 or several columns is currently not available.")

        # Column names have been generated by pandas' _MergeOperation tool
        op.pandas_kwargs.pop("suffixes", None)

        if op.pandas_kwargs["how"] == "left":
            op.pandas_kwargs.pop("how", None)

            joined_values = self._left_join(
                op.evaluation_keys,
                input_left,
                input_right,
                op.encrypted_nan,
                op.dtype,
                **op.pandas_kwargs,
            )

            return joined_values, joined_column_names

    def run(
        self,
        input_left: EncryptedDataFrame,
        input_right: EncryptedDataFrame,
        ops_left: EncryptedDataFrameOperator,
        ops_right: EncryptedDataFrameOperator,
    ):
        """Run the given operators using the given inputs.

        Running a server means we run a given list of operators in a given order with given inputs.
        For now, we only support a single operator (left join) and the API is not really able to
        support the combination of several operators in a general way.
        """
        # Load the inputs from dictionaries
        input_left = EncryptedDataFrame.from_dict(input_left)
        input_right = EncryptedDataFrame.from_dict(input_right)

        # Load the operators from the dictionaries
        ops_left = [EncryptedDataFrameOperator.from_dict(op_left) for op_left in ops_left]
        ops_right = [EncryptedDataFrameOperator.from_dict(op_right) for op_right in ops_right]

        # Check that the operators are supported by the server
        self._check_ops_and_dtypes_are_supported(ops_left)
        self._check_ops_and_dtypes_are_supported(ops_right)

        execution_start = time.time()

        # Execute the requested operators
        # TODO: this does not generalize to several operators yet, as the given inputs are
        # directly given into the loop's operator, and there is no way of inputting an operator's
        # output into a following operator's input yet
        for i, (op_left, op_right) in enumerate(zip(ops_left, ops_right)):

            # Check that both operators agree on what to run
            if not op_left.is_equal_to(op_right):
                raise ValueError(
                    f"Left and right operators {i} do not match. Got {op_left} (left) and "
                    f"{op_right} (right)."
                )

            # Retrieve the server method associated to the operator to run
            op_method = getattr(self, op_left.name)

            # This assumes both NaN representations have been encrypted using the same private
            # key, and that evaluation keys are the same
            output_values, output_columns = op_method(
                op_left,
                input_left,
                input_right,
            )

        execution_end = time.time() - execution_start

        print(f"Total FHE execution: {execution_end:.2f}s")

        # Build the output encrypted data-frame and serialize it
        encrypted_df_output = EncryptedDataFrame(output_values, output_columns)
        return encrypted_df_output.to_dict()
