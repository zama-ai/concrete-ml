"""Utility functions for deployment."""

from typing import Optional, Tuple, Union

from concrete import fhe


def serialize_encrypted_values(
    *values_enc: Optional[fhe.Value],
) -> Union[Optional[bytes], Tuple[Optional[bytes], ...]]:
    """Serialize encrypted values.

    If a value is None, None is returned.

    Args:
        values_enc (Optional[fhe.Value]): The values to serialize.

    Returns:
        Union[Optional[bytes], Optional[Tuple[bytes]]]: The serialized values.
    """
    values_enc_serialized = tuple(
        value_enc.serialize() if value_enc is not None else None for value_enc in values_enc
    )

    if len(values_enc_serialized) == 1:
        return values_enc_serialized[0]

    return values_enc_serialized


def deserialize_encrypted_values(
    *values_serialized: Optional[bytes],
) -> Union[Optional[fhe.Value], Optional[Tuple[fhe.Value]]]:
    """Deserialize encrypted values.

    If a value is None, None is returned.

    Args:
        values_serialized (Optional[bytes]): The values to deserialize.

    Returns:
        Union[Optional[fhe.Value], Optional[Tuple[fhe.Value]]]: The deserialized values.
    """
    values_enc = tuple(
        fhe.Value.deserialize(value_serialized) if value_serialized is not None else None
        for value_serialized in values_serialized
    )

    if len(values_enc) == 1:
        return values_enc[0]

    return values_enc
