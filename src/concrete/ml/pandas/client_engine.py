from pathlib import Path
from typing import Optional, Union

import pandas

from concrete import fhe
from concrete.ml.pandas._client_server import CLIENT_PATH, get_encrypt_config
from concrete.ml.pandas._processing import post_process_to_pandas, pre_process_from_pandas
from concrete.ml.pandas._utils import decrypt_elementwise, encrypt_elementwise, encrypt_value
from concrete.ml.pandas.dataframe import EncryptedDataFrame

CURRENT_API_VERSION = 1


class ClientEngine:
    """Define a framework that manages keys."""

    def __init__(
        self, keygen: bool = True, keys_path: Optional[Union[Path, str]] = None
    ) -> fhe.Client:
        self.client = fhe.Client.load(CLIENT_PATH)

        if keygen:
            self.keygen(keys_path=keys_path)

    def keygen(self, keys_path: Optional[Union[Path, str]] = None):
        """Generate the keys.

        Args:
            keys_path (Optional[Union[Path, str]]): The path where to save the keys. Note that if
                some keys already exist in that path, the client will use them instead of generating
                new ones. Default to None.
        """
        if keys_path is not None:
            self.client.keys.load_if_exists_generate_and_save_otherwise(keys_path)
        else:
            self.client.keygen(True)

    def encrypt_from_pandas(self, pandas_dataframe: pandas.DataFrame) -> EncryptedDataFrame:
        """Encrypt a Pandas data-frame using the loaded client.

        Args:
            pandas_dataframe (DataFrame): The Pandas data-frame to encrypt.

        Returns:
            EncryptedDataFrame: The encrypted data-frame.
        """
        pandas_array, dtype_mappings = pre_process_from_pandas(pandas_dataframe)

        # TODO: how to provide encrypt configuration
        encrypted_values = encrypt_elementwise(pandas_array, self.client, **get_encrypt_config())
        encrypted_nan = encrypt_value(0, self.client, **get_encrypt_config())

        return EncryptedDataFrame(
            encrypted_values,
            encrypted_nan,
            self.client.evaluation_keys,
            pandas_dataframe.columns,
            dtype_mappings,
            CURRENT_API_VERSION,
        )

    def decrypt_to_pandas(self, encrypted_dataframe: EncryptedDataFrame) -> pandas.DataFrame:
        """Decrypt an encrypted data-frame using the loaded client and return a Pandas data-frame.

        Args:
            encrypted_dataframe (EncryptedDataFrame): The encrypted data-frame to decrypt.

        Returns:
            pandas.DataFrame: The Pandas data-frame built on the decrypted values.
        """
        clear_array = decrypt_elementwise(encrypted_dataframe._encrypted_values, self.client)
        pandas_dataframe = post_process_to_pandas(
            clear_array, encrypted_dataframe._column_names, encrypted_dataframe._dtype_mappings
        )
        return pandas_dataframe
