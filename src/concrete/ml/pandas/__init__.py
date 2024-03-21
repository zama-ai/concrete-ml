from .client_server import get_client_and_eval_keys, load_client
from .dataframe import EncryptedDataFrame


def encrypt_from_pandas(pandas_dataframe, client, evaluation_keys):
    return EncryptedDataFrame.encrypt_from_pandas(pandas_dataframe, client, evaluation_keys)


def encrypt_from_csv(file_path, client, evaluation_keys, **pandas_kwargs):
    return EncryptedDataFrame.encrypt_from_csv(file_path, client, evaluation_keys, **pandas_kwargs)


def load_encrypted_dataframe(file_path):
    return EncryptedDataFrame.from_json(file_path)


def join(
    left_encrypted: EncryptedDataFrame,
    right_encrypted: EncryptedDataFrame,
    evaluation_keys,
    **pandas_kwargs
):
    return left_encrypted.join(right_encrypted, evaluation_keys, **pandas_kwargs)


def merge(
    left_encrypted: EncryptedDataFrame,
    right_encrypted: EncryptedDataFrame,
    evaluation_keys,
    **pandas_kwargs
):
    return left_encrypted.merge(right_encrypted, evaluation_keys, **pandas_kwargs)
