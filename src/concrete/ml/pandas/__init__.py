"""Public API for encrypted data-frames."""
from pathlib import Path
from typing import Union

from .client_engine import ClientEngine
from .dataframe import EncryptedDataFrame


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
