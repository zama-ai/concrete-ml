"""Custom decoder for serialization."""
import inspect
import json
from typing import Any, Dict, Type

import numpy
import onnx
import torch
from numpy.random import RandomState
from skorch.dataset import ValidSplit

from ...quantization.base_quantized_op import ALL_QUANTIZED_OPS
from ...quantization.quantized_module import QuantizedModule
from ...quantization.quantizers import (
    MinMaxQuantizationStats,
    QuantizationOptions,
    QuantizedArray,
    UniformQuantizationParameters,
    UniformQuantizer,
)
from ...sklearn import get_sklearn_models
from . import SUPPORTED_TORCH_ACTIVATIONS, USE_SKOPS

# If USE_SKOPS is False or Skops can't be imported, default to pickle
try:
    if USE_SKOPS:
        from skops.io import loads as pickle_or_skops_loads
    else:  # pragma: no cover
        raise ImportError()
except ImportError:  # pragma: no cover
    USE_SKOPS = False
    from pickle import loads as pickle_or_skops_loads


def _get_fully_qualified_name(object_class: Type) -> str:
    """Get the fully qualified name of a class.

    Args:
        object_class (Type): The class to consider.

    Returns:
        str: The fully qualified name.
    """
    return object_class.__module__ + "." + object_class.__name__


_TRUSTED_TORCH_ACTIVATIONS = [
    _get_fully_qualified_name(activation_class) for activation_class in SUPPORTED_TORCH_ACTIVATIONS
]

_TRUSTED_CONCRETE_MODELS = [
    _get_fully_qualified_name(model_class) for model_class in get_sklearn_models()["all"]
]

# Define all the trusted types that Skops should consider
TRUSTED_SKOPS = (
    _TRUSTED_TORCH_ACTIVATIONS
    + _TRUSTED_CONCRETE_MODELS
    + [_get_fully_qualified_name(QuantizedModule)]
    + [
        "numpy.int64",
        "numpy.float64",
        "numpy.int32",
        "xgboost.core.Booster",
        "xgboost.sklearn.XGBClassifier",
        "xgboost.sklearn.XGBRegressor",
        "sklearn._loss.glm_distribution.DistributionBoundary",
        "sklearn._loss.glm_distribution.TweedieDistribution",
        "sklearn._loss.glm_distribution.GammaDistribution",
        "sklearn._loss.glm_distribution.PoissonDistribution",
        "sklearn.linear_model._glm.link.LogLink",
        "sklearn.linear_model._glm.link.IdentityLink",
        "sklearn._loss.link.IdentityLink",
        "sklearn._loss.link.Interval",
        "sklearn._loss.link.LogLink",
        "sklearn._loss.link.LogLink",
        "sklearn._loss._loss.CyHalfTweedieLossIdentity",
        "sklearn._loss.loss.HalfTweedieLossIdentity",
        "sklearn._loss._loss.CyHalfPoissonLoss",
        "sklearn._loss.loss.HalfPoissonLoss",
        "sklearn._loss._loss.CyHalfGammaLoss",
        "sklearn._loss.loss.HalfGammaLoss",
        "sklearn._loss._loss.CyHalfTweedieLoss",
        "sklearn._loss.loss.HalfTweedieLoss",
        "torch.utils.data.dataloader.DataLoader",
        "torch.utils.data.dataset.Dataset",
        "skorch.dataset.Dataset",
        "skorch.dataset.ValidSplit",
        "inspect._empty",
    ]
)

# Initialize the list of all classes that can be serialized in Concrete ML (i.e., that have a
# `dump_dict` and `load_dict` method)
# pylint: disable=invalid-name
SERIALIZABLE_CLASSES: Dict[str, Type] = {}


# pylint: disable-next=too-many-return-statements, too-many-branches
def object_hook(d: Any) -> Any:
    """Define a custom object hook that enables loading any supported serialized values.

    If the input's type is non-native, then we expect it to have the following format.More
    information is available in the ConcreteEncoder class.

    Args:
        d (Any): The serialized value to load.

    Returns:
        Any: The loaded value.

    Raises:
        NotImplementedError: If the serialized object does not provides a `dump_dict` method as
            expected.
    """

    # If the current serialized value has the expected custom format, parse it and load the
    # appropriate object
    if "type_name" in d and "serialized_value" in d:

        type_name, serialized_value = d["type_name"], d["serialized_value"]

        if type_name == "RandomState":
            random_state = RandomState()
            random_state.set_state(serialized_value)
            return random_state

        # If the value is a scikit-learn model or a type object, load it using Skops (or Pickle
        # if Skops is not available or disabled)
        if type_name in ["sklearn_model", "type"]:
            loads_sklearn_kwargs = {}

            # If Skops is available, indicate the trusted objects to the loader. An error is
            # thrown if an object of an unexpected type is encountered
            if USE_SKOPS:
                loads_sklearn_kwargs["trusted"] = TRUSTED_SKOPS

            return pickle_or_skops_loads(bytes.fromhex(serialized_value), **loads_sklearn_kwargs)

        if type_name == "onnx_model":
            return onnx.load_model_from_string(bytes.fromhex(serialized_value))

        if type_name == "set":
            return set(serialized_value)

        if type_name == "tuple":
            return tuple(serialized_value)

        # Load the numpy integer value with the initial dtype
        if type_name == "numpy_integer":
            assert "dtype" in d, "Loading a dumped numpy integer requires a `dtype` parameter"

            return numpy.int64(serialized_value).astype(d["dtype"])

        # Load the numpy float value with the initial dtype
        if type_name == "numpy_float":
            assert "dtype" in d, "Loading a dumped numpy float requires a `dtype` parameter"

            return numpy.float64(serialized_value).astype(d["dtype"])

        # Load the numpy array with the initial dtype
        if type_name == "numpy_array":
            assert "dtype" in d, "Loading a dumped numpy array requires a `dtype` parameter"

            return numpy.asarray(serialized_value).astype(d["dtype"])

        if type_name == "torch_device":
            return torch.device(serialized_value)

        if type_name == "valid_split":
            return ValidSplit(**serialized_value)

        if type_name == "inspect_empty":
            # pylint: disable-next=protected-access
            return inspect._empty

        # pylint: disable-next=global-statement
        global SERIALIZABLE_CLASSES

        # Define the list of all classes that can be serialized in Concrete ML (i.e., that have a
        # `dump_dict` and `load_dict` method) if not already done
        if not SERIALIZABLE_CLASSES:
            serializable_classes = (
                get_sklearn_models()["all"]
                + list(ALL_QUANTIZED_OPS)
                + [
                    QuantizedArray,
                    QuantizedModule,
                    UniformQuantizer,
                    QuantizationOptions,
                    UniformQuantizationParameters,
                    MinMaxQuantizationStats,
                ]
            )

            # Map these classes with their names
            SERIALIZABLE_CLASSES = {
                model_class.__name__: model_class for model_class in serializable_classes
            }

        # If the value reaches this point and the initial object was properly serialized, we
        # expect it to be a class from Concrete ML that implements a `load_dict` method
        if type_name in SERIALIZABLE_CLASSES:
            serializable_class = SERIALIZABLE_CLASSES[type_name]

            assert hasattr(serializable_class, "load_dict"), (
                f"Class {type_name} does not support a 'load_dict' method and therefore "
                "cannot be serialized."
            )

            return serializable_class.load_dict(serialized_value)

        # Else, throw an error as this is not expected
        raise NotImplementedError(f"{type_name} does not support the `load_dict` method.")

    # If the current serialized value does not have the expected custom format, then it should
    # have a native type and can be handled by the JSONEncoder. Else, an error is thrown.
    return d


class ConcreteDecoder(json.JSONDecoder):
    """Custom json decoder to handle non-native types found in serialized Concrete ML objects."""

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=object_hook, *args, **kwargs)
