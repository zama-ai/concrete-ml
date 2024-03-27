"""Public API for encrypted data-frames."""
from pathlib import Path
from typing import Union

from concrete.fhe import Client
from pandas import DataFrame

from ._client_server import load_client
from .dataframe import EncryptedDataFrame


def encrypt_from_pandas(pandas_dataframe: DataFrame, client: Client) -> EncryptedDataFrame:
    """Encrypt a Pandas data-frame.

    Args:
        pandas_dataframe (DataFrame): The Pandas data-frame to encrypt.
        client (Client): The Concrete client to use for encrypting the data-frame.

    Returns:
        EncryptedDataFrame: The encrypted data-frame.
    """
    return EncryptedDataFrame.encrypt_from_pandas(pandas_dataframe, client)


def save_encrypted_dataframe(encrypted_dataframe: EncryptedDataFrame, path: Union[Path, str]):
    """Serialize and save an encrypted data-frame.

    Args:
        encrypted_dataframe (EncryptedDataFrame): The encrypted data-frame to serialize and save.
        path (Union[Path, str]): The path to consider for serializing and saving the encrypted
            data-frame.
    """
    encrypted_dataframe.save(path)


def load_encrypted_dataframe(path: Union[Path, str]) -> EncryptedDataFrame:
    """Load a serialized encrypted data-frame.

    Args:
        path (Union[Path, str]): The path to consider for loading the serialized encrypted
            data-frame.

    Returns:
        EncryptedDataFrame: The loaded encrypted data-frame.
    """
    return EncryptedDataFrame.load(path)


def merge(
    left_encrypted: EncryptedDataFrame, right_encrypted: EncryptedDataFrame, **pandas_kwargs
) -> EncryptedDataFrame:
    """Merge two encrypted data-frames using Pandas parameters.

    Args:
        left_encrypted (EncryptedDataFrame): The left encrypted data-frame.
        right_encrypted (EncryptedDataFrame): The right encrypted data-frame.

    Returns:
        EncryptedDataFrame: The merged encrypted data-frame.
    """
    return left_encrypted.merge(right_encrypted, **pandas_kwargs)
