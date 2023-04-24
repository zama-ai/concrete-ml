"""Dump functions for serialization."""
import json
from typing import Any, TextIO

from .encoder import ConcreteEncoder


def dumps(obj: Any) -> str:
    """Dump any object as a string.

    Arguments:
        obj (Any): Object to dump.

    Returns:
        str: A string representation of the object.
    """

    return json.dumps(obj, cls=ConcreteEncoder)


def dump(obj: Any, file: TextIO):
    """Dump any Concrete ML object in a file.

    Arguments:
        obj (Any): The object to dump.
        file (TextIO): The file to dump the serialized object into.
    """
    file.write(dumps(obj))
