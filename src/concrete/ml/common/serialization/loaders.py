"""Load functions for serialization."""
import json
from typing import Any, Callable, Dict, TextIO, Union

import onnx
from numpy.random import RandomState

# pylint: disable=relative-beyond-top-level
from ...quantization.quantizers import QuantizedArray, UniformQuantizer
from ...sklearn.base import _ALL_SKLEARN_MODELS

# pylint: disable=invalid-name
LOADS_METHODS: Dict[str, Callable] = {}


def load_dict(metadata: Dict[str, Any]) -> Any:
    """Load any Concrete ML object that has a dump method.

    Arguments:
        metadata (Dict[str, Any]): a dict of a serialized object.

    Returns:
        Any: the object itself.

    Raises:
        ValueError: if "cml_dumped_class_name" key is not in the serialized object.
    """
    # pylint: disable-next=global-statement
    global LOADS_METHODS
    if not LOADS_METHODS:
        LOADS_METHODS = {
            model_class.__name__: model_class.load_dict
            for model_class in _ALL_SKLEARN_MODELS
            if hasattr(model_class, "load_dict")
        }
        LOADS_METHODS["QuantizedArray"] = QuantizedArray.load_dict
        LOADS_METHODS["UniformQuantizer"] = UniformQuantizer.load_dict

    try:
        class_name = metadata["cml_dumped_class_name"]
    except KeyError as exception:
        raise ValueError("The content provided is not a Concrete ML dumped model.") from exception

    return LOADS_METHODS[class_name](metadata)


def loads(content: str) -> Any:
    """Load any Concrete ML object that has a dump method.

    Arguments:
        content (str): a serialized object.

    Returns:
        Any: the object itself.
    """
    parsed = json.loads(content)
    return load_dict(parsed)


def load(file: TextIO):
    """Load any Concrete ML object that has a dump method.

    Arguments:
        file (TextIO): a file containing the serialized object.

    Returns:
        Any: the object itself.
    """
    content = file.read()
    return loads(content)


def loads_onnx(serialized_onnx: str) -> onnx.ModelProto:
    """Load serialized onnx model.

    Arguments:
        serialized_onnx (str): a serialized onnx model.

    Returns:
        onnx.ModelProto: the onnx model
    """

    return onnx.load_model_from_string(bytes.fromhex(serialized_onnx))


def loads_random_state(serialized_random_state: str) -> Union[RandomState, int, None]:
    """Load random state from string.

    Arguments:
        serialized_random_state (str): a serialized version of the random state

    Returns:
        random_state (Union[RandomState, int, None]): a random state


    """
    deserialized = json.loads(serialized_random_state)
    if isinstance(deserialized, int) or (deserialized is None):
        return deserialized
    assert isinstance(deserialized, list)
    deserialized = tuple(deserialized)
    random_state = RandomState()
    random_state.set_state(deserialized)
    return random_state
