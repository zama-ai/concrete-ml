"""Serialization."""
import json
from typing import Any, Callable, Dict, TextIO, Union

import concrete.numpy as cnp
import numpy
import onnx
import sklearn
from numpy.random import RandomState

from concrete.ml import dumps_sklearn


# Register all models
class CustomEncoder(json.JSONEncoder):
    """CustomEncoder: custom json encoder to handle non-native types."""

    # pylint: disable-next=too-many-return-statements
    def default(self, o: Any) -> Any:
        """Overload default serialization.

        Arguments:
            o (Any): the object to serialize.

        Returns:
            The serialized object.

        Raises:
            NotImplementedError: if a cnp.Circuit is given.
        """

        # Use __getstate__, __setstate__ if RandomState, either easy
        if isinstance(o, RandomState):
            return o.get_state()

        if isinstance(o, sklearn.base.BaseEstimator):
            return dumps_sklearn(o).hex()

        if isinstance(o, onnx.ModelProto):
            return dumps_onnx(o)

        if isinstance(o, numpy.integer):
            return int(o)

        if isinstance(o, type):
            return o.__name__

        # Match all the types you want to handle in your converter
        if isinstance(o, numpy.ndarray):
            return o.tolist()

        if hasattr(o, "dump_dict"):
            return o.dump_dict()

        # FIXME: https://github.com/zama-ai/concrete-numpy-internal/issues/1841
        if isinstance(o, cnp.Circuit):  # pragma: no cover
            raise NotImplementedError(
                "Concrete-Numpy Circuit object serialization is not implemented yet"
            )

        # Call the default method for other types
        return json.JSONEncoder.default(self, o)  # pragma: no cover


def dumps_onnx(onnx_model: onnx.ModelProto) -> str:
    """Dump onnx model as string.

    Arguments:
        onnx_model (onnx.ModelProto): an onnx model.

    Returns:
        str: a serialized version of the onnx model.
    """
    return onnx_model.SerializeToString().hex()


def loads_onnx(serialized_onnx: str) -> onnx.ModelProto:
    """Load serialized onnx model.

    Arguments:
        serialized_onnx (str): a serialized onnx model.

    Returns:
        onnx.ModelProto: the onnx model
    """

    return onnx.load_model_from_string(bytes.fromhex(serialized_onnx))


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


def loads_random_state(serialized_random_state: str) -> Union[RandomState, int, None]:
    """Load random state from string.

    Arguments:
        serialized_random_state (str): a serialized version of the random state

    Returns:
        random_state (Union[RandomState, int, None]): a random state


    """
    unserialized = json.loads(serialized_random_state)
    if isinstance(unserialized, int) or (unserialized is None):
        return unserialized
    assert isinstance(unserialized, list)
    unserialized = tuple(unserialized)
    random_state = RandomState()
    random_state.set_state(unserialized)
    return random_state


LOADS_METHODS: Dict[str, Callable] = {}


def loads(content: str) -> Any:
    """Load any CML object that has a dump method.

    Arguments:
        content (str): a serialized object.

    Returns:
        Any: the object itself.
    """
    parsed = json.loads(content)
    return load_dict(parsed)


def load_dict(metadata: Dict[str, Any]) -> Any:
    """Load any CML object that has a dump method.

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
        # Populate LOADS_METHODS now to avoid circular imports
        # pylint: disable-next=import-outside-toplevel
        from concrete.ml.quantization.quantizers import QuantizedArray, UniformQuantizer

        # pylint: disable-next=import-outside-toplevel
        from concrete.ml.sklearn.base import _ALL_SKLEARN_MODELS

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
        raise ValueError("The content provided is not a CML dumped model.") from exception

    return LOADS_METHODS[class_name](metadata)


def load(file: TextIO):
    """Load any CML object that has a dump method.

    Arguments:
        file (TextIO): a file containing the serialized object.

    Returns:
        Any: the object itself.
    """
    content = file.read()
    return loads(content)


def dump(obj: Any, file: TextIO):
    """Dump any CML object that has a dump method.

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
