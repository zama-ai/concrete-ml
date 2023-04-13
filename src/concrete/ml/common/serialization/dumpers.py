"""Dump functions for serialization."""
import json
from typing import Any, TextIO, Union

from numpy.random import RandomState

# pylint: disable=relative-beyond-top-level
from .encoder import CustomEncoder


def dumps_random_state(random_state: Union[RandomState, int, None]) -> str:
    """Dump random state to string.

    Arguments:
        random_state (Union[RandomState, int, None]): a random state

    Returns:
        str: a serialized version of the random state
    """

    if isinstance(random_state, int):
        return str(random_state)
    if random_state is None:
        return "null"
    return json.dumps(random_state.get_state(), cls=CustomEncoder)


def dump(obj: Any, file: TextIO):
    """Dump any Concrete ML object that has a dump method.

    Arguments:
        obj (Any): the object to dump.
        file (TextIO): a file containing the serialized object.
    """
    file.write(dumps(obj))


def dumps(obj: Any) -> str:
    """Dump as string any object.

    If the object has some `dumps` method then it uses that.
    Otherwise the object is casted as `str`.

    Arguments:
        obj (Any): any object.

    Returns:
        str: a string representation of the object.
    """

    if hasattr(obj, "dumps"):
        return obj.dumps()
    return json.dumps(obj, cls=CustomEncoder)
