from ._client_server import load_client
from .dataframe import EncryptedDataFrame


def encrypt_from_pandas(pandas_dataframe, client):
    return EncryptedDataFrame.encrypt_from_pandas(pandas_dataframe, client)


def load_encrypted_dataframe(file_path):
    return EncryptedDataFrame.load(file_path)


def save_encrypted_dataframe(encrypted_dataframe: EncryptedDataFrame, file_path):
    return encrypted_dataframe.save(file_path)


def merge(left_encrypted: EncryptedDataFrame, right_encrypted: EncryptedDataFrame, **pandas_kwargs):
    return left_encrypted.merge(right_encrypted, **pandas_kwargs)
