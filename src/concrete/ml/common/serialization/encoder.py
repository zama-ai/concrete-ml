"""Custom encoder for serialization."""
import inspect
import json
from json.encoder import _make_iterencode  # type: ignore[attr-defined]
from json.encoder import encode_basestring  # type: ignore[attr-defined]
from json.encoder import encode_basestring_ascii  # type: ignore[attr-defined]
from json.encoder import INFINITY, JSONEncoder
from typing import Any, Callable, Dict, Generator, Type

import numpy
import onnx
import sklearn
import torch
from numpy.random import RandomState
from skorch.dataset import ValidSplit

from concrete import fhe

from . import USE_SKOPS

# If USE_SKOPS is False or Skops can't be imported, default to pickle
try:
    if USE_SKOPS:
        from skops.io import dumps as pickle_or_skops_dumps
    else:  # pragma: no cover
        raise ImportError()
except ImportError:  # pragma: no cover
    USE_SKOPS = False
    from pickle import dumps as pickle_or_skops_dumps


def dump_name_and_value(name: str, value: Any, **kwargs) -> Dict:
    """Dump the value into a custom dict format.

    Args:
        name (str): The custom name to use. This name should be unique for each type to encode, as
            it is used in the ConcreteDecoder class to detect the initial type and apply the proper
            load method to the serialized object.
        value (Any): The serialized value to dump.
        **kwargs (dict): Additional arguments to dump.

    Returns:
        Dict: The serialized custom format that includes both the serialized value and its type
            name.
    """
    name_and_value = {
        "type_name": name,
        "serialized_value": value,
    }

    assert "type_name" not in kwargs and "serialized_value" not in kwargs

    name_and_value.update(kwargs)
    return name_and_value


class ConcreteEncoder(JSONEncoder):
    """Custom json encoder to handle non-native types found in serialized Concrete ML objects.

    Non-native types are serialized manually and dumped in a custom dict format that stores both the
    serialization value of the object and its associated type name.

    The name should be unique for each type, as it is used in the ConcreteDecoder class to detect
    the initial type and apply the proper load method to the serialized object. The serialized
    value is the value that was serialized manually in a native type. Additional arguments such
    as a numpy array's dtype are also properly serialized. If an object has an unexpected type or
    is not serializable, an error is thrown.

    The ConcreteEncoder is only meant to encode Concrete-ML's built-in models and therefore only
    supports the necessary types. For example, torch.Tensor objects are not serializable using this
    encoder as built-in models only use numpy arrays. However, the list of supported types might
    expand in future releases if new models are added and need new types.
    """

    @staticmethod
    def isinstance(o: Any, cls: Type) -> bool:
        """Define a custom isinstance method.

        Natively, among other types, the JSONENcoder handles integers, floating points and tuples.
        However, a numpy.integer (resp. numpy.floating) object is automatically casted to a built-in
        int (resp. float) object, without keeping their dtype information. Similarly, a tuple is
        casted to a list, meaning that it will then be loaded as a list, which notably does not have
        the uniqueness property and therefore might cause issues in complex structures such as
        QuantizedModule instances. This is an issue as JSONEncoder only calls its customizable
        `default` method at the end of the parsing. We thus need to provide this custom isinstance
        method in order to make the encoder avoid handling these specific types until `default` is
        reached (where they are properly serialized using our custom format).

        Args:
            o (Any): The object to serialize.
            cls (Type): The type to compare the object with.

        Returns:
            bool: If the object is of the given type. False if it is a numpy.floating, numpy.integer
                or a tuple.
        """
        if isinstance(o, numpy.floating):
            return False

        if isinstance(o, numpy.integer):
            return False

        if isinstance(o, tuple):
            return False

        return isinstance(o, cls)

    # Coverage is disabled many times in the following method as it is a slightly modified version
    # of JSONEncoder's iterencode and thus should not be fully tested
    def iterencode(self, o: Any, _one_shot: bool = False) -> Generator:
        """Encode the given object and yield each string representation as available.

        This method overrides the JSONEncoder's native iterencode one in order to pass our custom
        isinstance method to the `_make_iterencode` function. More information in `isinstance`'s
        docstring. For simplicity, iterencode does not give the ability to use the initial
        `c_make_encoder` function, as it would required to override it in C.

        Args:
            o (Any): The object to serialize.
            _one_shot (bool): This parameter is not used since the `_make_iterencode` function has
                been removed from the method.

        Returns:
            Generator: Yield each string representation as available.
        """
        if self.check_circular:
            markers = {}  # type: ignore[var-annotated]
        else:
            markers = None  # pragma: no cover
        if self.ensure_ascii:
            _encoder = encode_basestring_ascii
        else:
            _encoder = encode_basestring  # pragma: no cover

        def floatstr(
            o, allow_nan=self.allow_nan, _repr=float.__repr__, _inf=INFINITY, _neginf=-INFINITY
        ):
            # Check for specials.  Note that this type of test is processor
            # and/or platform-specific, so do tests which don't depend on the
            # internals.

            # pylint: disable-next=comparison-with-itself
            if o != o:
                text = "NaN"
            elif o == _inf:
                text = "Infinity"  # pragma: no cover
            elif o == _neginf:
                text = "-Infinity"  # pragma: no cover
            else:
                return _repr(o)

            if not allow_nan:
                raise ValueError(  # pragma: no cover
                    "Out of range float values are not JSON compliant: " + repr(o)
                )

            return text

        # Force `_make_iterencode` to use our custom `isinstance` method
        _iterencode = _make_iterencode(
            markers,
            self.default,
            _encoder,
            self.indent,
            floatstr,
            self.key_separator,
            self.item_separator,
            self.sort_keys,
            self.skipkeys,
            False,
            isinstance=self.isinstance,
        )
        return _iterencode(o, 0)

    # pylint: disable-next=too-many-return-statements, too-many-branches
    def default(self, o: Any) -> Any:
        """Define a custom default method that enables dumping any supported serialized values.

        Arguments:
            o (Any): The object to serialize.

        Returns:
            Any: The serialized object. Non-native types are returned as a dict of a specific
                format.

        Raises:
            NotImplementedError: If an FHE.Circuit, a Callable or a Generator object is given.
        """

        # Serializing a Circuit object is currently not supported
        # FIXME: https://github.com/zama-ai/concrete-numpy-internal/issues/1841
        if isinstance(o, fhe.Circuit):
            raise NotImplementedError("Concrete Circuit object serialization is not implemented.")

        if isinstance(o, RandomState):
            return dump_name_and_value("RandomState", o.get_state())

        # scikit-learn does not provide a particular dumping/loading method. We thus need to dump
        # these models using either Skops or pickle as a hexadecimal byte string. Additionally,
        # Concrete ML models, which currently inherit from scikit-learn models, have their own
        # serialization methods. We therefore make sure that they do not get serialized here
        if isinstance(o, sklearn.base.BaseEstimator) and not hasattr(o, "_is_a_public_cml_model"):
            return dump_name_and_value("sklearn_model", pickle_or_skops_dumps(o).hex())

        if isinstance(o, onnx.ModelProto):
            return dump_name_and_value("onnx_model", o.SerializeToString().hex())

        # The list is sorted before being serialized in order to be able to properly compare two
        # JSON strings, as sets do not have any order notion but lists do.
        if isinstance(o, set):
            return dump_name_and_value("set", sorted(list(o)))

        if isinstance(o, tuple):
            return dump_name_and_value("tuple", list(o))

        # Dump the numpy integer value along its dtype
        if isinstance(o, numpy.integer):
            kwargs = {"dtype": str(o.dtype)}
            return dump_name_and_value("numpy_integer", int(o), **kwargs)

        # Dump the numpy float value along its dtype
        if isinstance(o, numpy.floating):
            kwargs = {"dtype": str(o.dtype)}
            return dump_name_and_value("numpy_float", float(o), **kwargs)

        # Dump the numpy array along its dtype
        if isinstance(o, numpy.ndarray):
            kwargs = {"dtype": str(o.dtype)}
            return dump_name_and_value("numpy_array", o.tolist(), **kwargs)

        # This specific type is widely used in QuantizedModule instances and therefore is treated
        # separately
        # pylint: disable-next=protected-access
        if o is inspect._empty:
            return dump_name_and_value("inspect_empty", "inspect._empty")

        # Ideally, serializing type objects should be avoided as it either requires to import all
        # supported types when loading or specify each trusted object to Skops, which can become
        # very exhaustive. However, QNNs still provide few type objects (mostly coming from skorch)
        if isinstance(o, type):
            return dump_name_and_value("type", pickle_or_skops_dumps(o).hex())

        if isinstance(o, torch.device):
            return dump_name_and_value("torch_device", str(o))

        # A ValidSplit instance is used by default when initializing a QNN but skorch does not
        # provide any simple ways for serializing it. Since this such an instance is used by
        # default, ValidSplit instances are treated manually as well. However, this does not work
        # if the cross-validation strategy is set using a Generator object (see below)
        if isinstance(o, ValidSplit):
            if isinstance(o.cv, Generator):
                raise NotImplementedError(
                    "Serializing a custom Generator object is not secure and is therefore "
                    "disabled. Please choose a different cross-validation splitting strategy."
                )
            return dump_name_and_value("valid_split", vars(o))

        # All serializable classes from Concrete ML provide a `dump_dict` method that serializes
        # their attributes
        if hasattr(o, "dump_dict"):
            return dump_name_and_value(type(o).__name__, o.dump_dict())

        # Serializing Callable and Generator objects is disabled as it cannot be done securely. More
        # precisely, anyone could modify the json file after dumping such object and therefore make
        # the JSONDecoder load any undesired function
        # Disable mypy as running isinstance with a Callable type unexpectedly raises an issue:
        # https://github.com/python/mypy/issues/3060
        if isinstance(o, (Callable, Generator)):  # type: ignore[arg-type]
            raise NotImplementedError(
                "Serializing a custom Callable or Generator object is not secure and is therefore "
                f"disabled. Got {str(o)}."
            )

        # Call the default method for other native types (e.g., dict, str, bool, ...)
        return json.JSONEncoder.default(self, o)
