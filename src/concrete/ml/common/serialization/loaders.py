"""Load functions for serialization."""
import json
from typing import IO, Any, Union

from .decoder import ConcreteDecoder


def loads(content: Union[str, bytes]) -> Any:
    """Load any Concrete ML object that provide a `dump_dict` method.

    Arguments:
        content (Union[str, bytes]): A serialized object.

    Returns:
        Any: The object itself.
    """
    return json.loads(content, cls=ConcreteDecoder)


def load(file: Union[IO[str], IO[bytes]]):
    """Load any Concrete ML object that provide a `load_dict` method.

    Arguments:
        file (Union[IO[str], IO[bytes]): The file containing the serialized object.

    Returns:
        Any: The object itself.
    """
    content = file.read()
    return loads(content)
