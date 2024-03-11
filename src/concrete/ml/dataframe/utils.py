import functools

import numpy

from concrete import fhe


def encrypt_value(value: int, client: fhe.Client, n: int, pos: int = None):
    """Encrypt a value given a circuit and its configuration.

    Arguments:
        value: The value to encrypt.
        client: The client to use for encrypting.
        n: The total number of inputs the circuit asks for.
        pos: The circuit's input position to consider for encrypting the value.
    """
    if pos is None:
        encrypted_output = client.encrypt(value)
    else:
        clear_inputs = [None] * n
        clear_inputs[pos] = value

        encrypted_outputs = client.encrypt(*clear_inputs)

        encrypted_output = encrypted_outputs[pos]

    return encrypted_output


def encrypt_elementwise(array: numpy.ndarray, client: fhe.Client, n: int, pos: int = None):
    """Encrypt an array element-wise.

    Arguments:
        array: The array whose values to encrypt.
        client: The client to use for encrypting.
        n: The total number of inputs the circuit asks for.
        pos: The circuit's input position to consider for encrypting the value.
    """
    encrypt_func = functools.partial(encrypt_value, client=client, n=n, pos=pos)
    return numpy.vectorize(encrypt_func)(array)


def decrypt_elementwise(array: numpy.ndarray, client: fhe.Client):
    """Decrypt an array element-wise."""
    decrypt_func = lambda x: client.decrypt(x)
    return numpy.vectorize(decrypt_func)(array)


def serialize_value(value: fhe.Value):
    """Serialize an FHE value into a byte string."""
    return value.serialize().hex()


def serialize_evaluation_keys(value: fhe.Value):
    """Serialize evaluation keys into a byte string."""
    return serialize_value(value)


def serialize_elementwise(array: numpy.ndarray):
    """Serialize an array element-wise."""
    return numpy.vectorize(serialize_value, otypes=[object])(array)


def deserialize_value(value: str):
    """Deserialize an FHE value represented as a byte string."""
    return fhe.Value.deserialize(bytes.fromhex(value))


def deserialize_elementwise(array: numpy.ndarray):
    """Deserialize an array element-wise."""
    return numpy.vectorize(deserialize_value)(array)


def deserialize_evaluation_keys(evaluation_keys: str):
    """Deserialize evaluation keys represented as a byte string."""
    return fhe.EvaluationKeys.deserialize(bytes.fromhex(evaluation_keys))
