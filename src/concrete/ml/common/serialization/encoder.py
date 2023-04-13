"""Custom encoder for serialization."""
import json
from typing import Any

import numpy
import onnx
import sklearn
from numpy.random import RandomState

from concrete import fhe

from ... import dumps_sklearn


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
            NotImplementedError: if a fhe.Circuit is given.
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
        if isinstance(o, fhe.Circuit):  # pragma: no cover
            raise NotImplementedError(
                "Concrete Circuit object serialization is not implemented yet"
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
