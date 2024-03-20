from .client_server import get_client_and_eval_keys, load_client, load_server
from .dataframe import EncryptedDataFrame


def encrypt_from_pandas(pandas_df, client, evaluation_keys):
    return EncryptedDataFrame.encrypt_from_pandas(pandas_df, client, evaluation_keys)


def encrypt_from_csv(file_path, client, evaluation_keys, **pandas_kwargs):
    return EncryptedDataFrame.encrypt_from_csv(file_path, client, evaluation_keys, **pandas_kwargs)


def load_encrypted_dataframe(file_path):
    return EncryptedDataFrame.from_json(file_path)


def join(
    encrypted_df_left: EncryptedDataFrame,
    encrypted_df_right: EncryptedDataFrame,
    evaluation_keys,
    server,
    **pandas_kwargs
):
    return encrypted_df_left.join(encrypted_df_right, evaluation_keys, server, **pandas_kwargs)


def merge(
    encrypted_df_left: EncryptedDataFrame,
    encrypted_df_right: EncryptedDataFrame,
    evaluation_keys,
    server,
    **pandas_kwargs
):
    return encrypted_df_left.merge(encrypted_df_right, evaluation_keys, server, **pandas_kwargs)
