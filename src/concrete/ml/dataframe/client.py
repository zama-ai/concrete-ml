from typing import Dict, List

import numpy
import pandas

from .dataframe import EncryptedDataFrame
from .operator import EncryptedDataFrameOperator
from .utils import decrypt_elementwise, encrypt_elementwise


class EncryptedDataFrameClient:
    """Client object for pre-processing, encrypting and serializing data-frames to the server.

    A client is defined by a list of operators and their associated FHE clients.
    """

    def __init__(self, ops_kwargs: Dict, deployment_dir):

        # Ordered encrypted pandas operators to consider
        self.ops = [
            EncryptedDataFrameOperator.init_and_check_metadata(
                metadata_dir_path=deployment_dir, **op_kwargs
            )
            for op_kwargs in ops_kwargs
        ]

        # Ordered FHE clients to consider
        self.clients = [op.load_client(deployment_dir) for op in self.ops]

    @staticmethod
    def _validate_data_frame(df: pandas.DataFrame, min_value: int, max_value: int):
        """Check that the data-frame only contains values between the given min/max."""
        columns_less_than_min = (df < min_value).any()
        column_names_less_than_min = columns_less_than_min[columns_less_than_min].index.tolist()

        if column_names_less_than_min:
            raise ValueError(
                f"Columns {column_names_less_than_min} contain values less than {min_value}, "
                "which is not allowed."
                + " This is because 0 values are used to represent NaN values for FHE computations."
                * min_value
                == 0
            )

        columns_greater_than_max = (df > max_value).any()
        column_names_greater_than_max = columns_greater_than_max[
            columns_greater_than_max
        ].index.tolist()

        if column_names_greater_than_max:
            raise ValueError(
                f"Columns {column_names_greater_than_max} contain values greater than {max_value}, "
                "which is not allowed."
            )

    def generate_keys(self, force: bool):
        """Generate the keys for all fhe clients."""
        for client in self.clients:
            client.keygen(force)

    def pre_process(self, input_df: pandas.DataFrame, op_position: int = 0) -> numpy.ndarray:
        """Pre-process the Pandas data-frame using the operator identified by the given position."""
        # Make sure the given data-frame only contains values of the expected dtype (unsigned
        # integers of a given bit-width)
        # Additionally, forbid 0 values as they are then used to represent encrypted NaN values
        min, max = self.ops[op_position].get_supported_min_max()
        self._validate_data_frame(input_df, min, max)

        # Replace NaN values with 0
        input_df.fillna(0, inplace=True)

        array_to_encrypt = input_df.to_numpy()

        return array_to_encrypt

    # Use first op's client to encrypt values by default
    def encrypt(
        self, array_to_encrypt: numpy.ndarray, force_keygen: bool = False, op_position: int = 0
    ):
        """Encrypt the values using the operator's FHE client identified by the given position."""
        self.generate_keys(force=force_keygen)

        op = self.ops[op_position]
        client = self.clients[op_position]

        encrypted_values = encrypt_elementwise(
            array_to_encrypt,
            client,
            **op.get_encrypt_config(),
        )

        return encrypted_values

    def generate_encrypted_nan_values(self):
        """Generate encrypted NaN representations for all FHE clients."""
        for op, client in zip(self.ops, self.clients):
            op.generate_encrypted_nan_value(client)

    def retrieve_evaluation_keys(self):
        """Retrieve and store evaluations keys for all FHE clients."""
        for op, client in zip(self.ops, self.clients):
            op.retrieve_evaluation_keys(client)

    def pre_process_encrypt_serialize(self, input_df: pandas.DataFrame, force_keygen: bool = False):
        """Process the Pandas data-frame, then encrypt and serialize the values.

        We currently assume that input values are only made for the first operator, meaning they
        are encrypted using this operator's FHE client.
        """
        # TODO: for now, we assume that the only inputs to encrypt are the ones from the first op
        first_op_position = 0

        array_to_encrypt = self.pre_process(input_df, op_position=first_op_position)
        encrypted_values = self.encrypt(
            array_to_encrypt, force_keygen=force_keygen, op_position=first_op_position
        )

        encrypted_df_input = EncryptedDataFrame(encrypted_values, input_df.columns)

        serialized_df_input = encrypted_df_input.to_dict()

        return serialized_df_input

    def get_serialized_ops(self):
        """Update and serialize the operators."""

        self.generate_encrypted_nan_values()
        self.retrieve_evaluation_keys()

        serialized_ops = [op.to_dict() for op in self.ops]

        return serialized_ops

    # Use last op's client to decrypt values
    def decrypt(self, encrypted_values: numpy.ndarray, op_position: int = -1):
        """Decrypt the values using the operator's FHE client identified by the given position."""
        return decrypt_elementwise(encrypted_values, self.clients[op_position])

    @staticmethod
    def post_process(output_array: numpy.ndarray, output_column_names: List[str]):
        """Post-process the server's outputs and build a Pandas data-frame from them."""
        # Replace 0 values by NaN
        output_array_0_to_nan = numpy.where(output_array == 0, numpy.nan, output_array)

        # Convert the array with a pandas data-frame
        df = pandas.DataFrame(
            output_array_0_to_nan,
            columns=output_column_names,
        )
        return df

    def deserialize_decrypt_post_process(self, server_output: Dict):
        """Process the server's outputs and provide them as a Pandas data-frame."""
        encrypted_df_output = EncryptedDataFrame.from_dict(server_output)

        output_array = self.decrypt(encrypted_df_output.encrypted_values)
        output_data_frame = self.post_process(output_array, encrypted_df_output.column_names)
        return output_data_frame
