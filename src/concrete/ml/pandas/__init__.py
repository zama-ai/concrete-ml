from .client_server import (
    SUPPORTED_PANDAS_OPS_AND_KWARGS,
    get_client_and_eval_keys,
    get_clients_and_eval_keys,
    load_client,
    load_clients,
    load_server,
    load_servers,
    save_clients_servers,
)
from .dataframe import EncryptedDataFrame


def encrypt_from_pandas(pandas_df, client, evaluation_keys, operator):
    return EncryptedDataFrame.encrypt_from_pandas(pandas_df, client, evaluation_keys, operator)


def encrypt_from_csv(file_path, client, evaluation_keys, operator, **pandas_kwargs):
    return EncryptedDataFrame.encrypt_from_csv(
        file_path, client, evaluation_keys, operator, **pandas_kwargs
    )


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
