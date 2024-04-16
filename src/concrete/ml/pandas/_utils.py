"""Define utility functions for encrypted data-frames."""

import functools
from typing import List, Optional, Tuple, Union

import numpy

from concrete import fhe


def encrypt_value(
    value: Optional[Union[int, numpy.ndarray, List]], client: fhe.Client, n: int, pos: int
) -> Optional[Union[fhe.Value, Tuple[Optional[fhe.Value], ...]]]:
    """Encrypt a value using a Concrete client and the given configuration parameters.

    Args:
        value (Optional[Union[int, numpy.ndarray, List]]): The value(s) to encrypt.
        client (fhe.Client): The client to use for encryption.
        n (int): The total number of inputs the client's circuit considers.
        pos (int): The input's position to consider when encrypting it.

    Returns:
        Optional[Union[fhe.Value, Tuple[Optional[fhe.Value], ...]]]: A 'n'-tuple containing the
            encrypted value at position 'pos' and None elsewhere.
    """
    # Build the input to use for encrypting the value
    # In Concrete Python, if the underlying circuit asks for 4 inputs but we only want to encrypt
    # a value using the 2nd input, we need to provide (None, value, None, None) to tne '.encrypt'
    clear_inputs = [None] * n
    clear_inputs[pos] = value  # type: ignore[assignment]

    encrypted_outputs = client.encrypt(*clear_inputs)

    # Similarly, using the above example, the output becomes (None, encrypted_output, None, None)
    encrypted_output = encrypted_outputs[pos]

    return encrypted_output


def decrypt_value(
    value: fhe.Value, client: fhe.Client
) -> Optional[Union[int, numpy.ndarray, Tuple[Optional[Union[int, numpy.ndarray]], ...]]]:
    """Decrypt an FHE value using a Concrete client.

    Args:
        value (fhe.Value): The FHE value(s) to decrypt.
        client (fhe.Client): The client to use for decryption.

    Returns:
        Optional[Union[int, numpy.ndarray, Tuple[Optional[Union[int, numpy.ndarray]], ...]]]: The
            decrypted value(s).
    """
    return client.decrypt(value)


def encrypt_elementwise(
    array: numpy.ndarray, client: fhe.Client, n: int, pos: int
) -> numpy.ndarray:
    """Encrypt an array element-wise.

    Arguments:
        array (numpy.ndarray): The array whose values to encrypt.
        client (fhe.Client): The client to use for encryption.
        n (int): The total number of inputs the client's circuit considers.
        pos (int): The input's position to consider when encrypting it.

    Returns:
        numpy.ndarray: An array containing encrypted values only.
    """
    encrypt_func = functools.partial(encrypt_value, client=client, n=n, pos=pos)
    return numpy.vectorize(encrypt_func)(array)


def decrypt_elementwise(array: numpy.ndarray, client: fhe.Client) -> numpy.ndarray:
    """Decrypt an array element-wise.

    Args:
        array (numpy.ndarray): The array whose values to decrypt.
        client (fhe.Client): The client to use for decryption.

    Returns:
        numpy.ndarray: An array containing decrypted values only.
    """
    decrypt_func = functools.partial(decrypt_value, client=client)
    return numpy.vectorize(decrypt_func)(array)


def serialize_value(encrypted_value: fhe.Value) -> str:
    """Serialize an FHE value into a string of hexadecimal numbers.

    Args:
        encrypted_value (fhe.Value): The FHE value to serialize.

    Returns:
        str: The serialized FHE value as a string of hexadecimal numbers.
    """
    return encrypted_value.serialize().hex()


def deserialize_value(serialized_value: str) -> fhe.Value:
    """Deserialize a string of hexadecimal numbers into an FHE value.

    Args:
        serialized_value (str): The string to deserialize.

    Returns:
        str: The deserialized FHE value.
    """

    return fhe.Value.deserialize(bytes.fromhex(serialized_value))


def serialize_elementwise(array: numpy.ndarray) -> numpy.ndarray:
    """Serialize an array made of encrypted values element-wise.

    Args:
        array (numpy.ndarray): The array to serialize.

    Returns:
        numpy.ndarray: An array containing serialized encrypted values only.
    """
    return numpy.vectorize(serialize_value, otypes=[object])(array)


def deserialize_elementwise(array: numpy.ndarray) -> numpy.ndarray:
    """Deserialize an array made of serialized encrypted values element-wise.

    Args:
        array (numpy.ndarray): The array to deserialize.

    Returns:
        numpy.ndarray: An array containing deserialized encrypted values only.
    """
    return numpy.vectorize(deserialize_value)(array)


def serialize_evaluation_keys(evaluation_keys: fhe.EvaluationKeys) -> bytes:
    """Serialize the evaluation keys into bytes.

    Args:
        evaluation_keys (fhe.EvaluationKeys): The evaluation keys to serialize.

    Returns:
        bytes: The serialized evaluation keys.
    """
    return evaluation_keys.serialize()


def deserialize_evaluation_keys(serialized_evaluation_keys: bytes) -> fhe.EvaluationKeys:
    """Deserialize the evaluation keys.

    Args:
        serialized_evaluation_keys (bytes): The evaluation keys to deserialize.

    Returns:
        fhe.EvaluationKeys: The deserialized evaluation keys.
    """
    return fhe.EvaluationKeys.deserialize(serialized_evaluation_keys)


def slice_hex_str(hex_str: str, n: int = 10) -> str:
    """Extract the n hexadecimal numbers found in the middle of the given string.

    This method is used for printing a small part of encrypted values, serialized as
    hexadecimal numbers. Dots are added before and after the sliced string.

    Args:
        hex_str (str): A string made of hexadecimal numbers.
        n (int): The amount of characters to extract. Default to 10.

    Returns:
        str: The extracted numbers, with dots before and after.
    """
    # Get the string's middle index
    start_index = len(hex_str) // 2
    assert start_index + n < len(hex_str)

    return ".." + hex_str[start_index : start_index + n] + ".."


def get_serialized_representation_elementwise(array: numpy.ndarray) -> numpy.ndarray:
    """Get a representation of serialized values stored in an array.

    Args:
        array (numpy.ndarray): The array to consider, made of serialized values represented as a
            string of hexadecimal numbers.

    Returns:
        numpy.ndarray: An array containing the serialized values' representations.
    """
    return numpy.vectorize(slice_hex_str)(array)
