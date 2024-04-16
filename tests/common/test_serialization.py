"""Test serialization.

Here we test the custom dump(s)/load(s) functions for all supported objects. We also check that
serializing unsupported object types properly throws an error.
"""

import inspect
import io
import warnings
from functools import partial

import numpy
import onnx
import pytest
import sklearn
import sklearn.base
import torch
from concrete.fhe.compilation import Circuit
from numpy.random import RandomState
from sklearn.datasets import make_regression
from sklearn.exceptions import ConvergenceWarning
from skops.io.exceptions import UntrustedTypesFoundException
from skorch.dataset import ValidSplit
from torch import nn

from concrete.ml.common.serialization import (
    SUPPORTED_TORCH_ACTIVATIONS,
    UNSUPPORTED_TORCH_ACTIVATIONS,
    USE_SKOPS,
)
from concrete.ml.common.serialization.dumpers import dumps
from concrete.ml.common.serialization.loaders import loads
from concrete.ml.pytest.torch_models import SimpleNet
from concrete.ml.pytest.utils import check_serialization, values_are_equal
from concrete.ml.quantization import QuantizedModule
from concrete.ml.sklearn import (
    LinearRegression,
    _get_sklearn_all_models,
    _get_sklearn_linear_models,
    _get_sklearn_tree_models,
)


def valid_split_instances_are_equal(instance_1: ValidSplit, instance_2: ValidSplit) -> bool:
    """Check if two ValidSplit instances are equal.

    Args:
        instance_1 (ValidSplit): The first ValidSplit object to consider.
        instance_2 (ValidSplit): The second ValidSplit object to consider.

    Returns:
        bool: If both instances are equal.
    """
    for attribute in ["cv", "stratified", "random_state"]:
        value_1, value_2 = getattr(instance_1, attribute), getattr(instance_2, attribute)
        if not values_are_equal(value_1, value_2):
            return False
    return True


def sklearn_predictions_are_equal(
    sklearn_model_1: sklearn.base.BaseEstimator,
    sklearn_model_2: sklearn.base.BaseEstimator,
    x: numpy.ndarray,
) -> bool:
    """Check that the predictions made by both Scikit-Learn models are equal.

    scikit-learn does not provide any simple way of comparing two models (attribute-wise) as no
    __eq__ method is implemented. Therefore, we consider models identical if they both provide the
    same predictions.

    Args:
        sklearn_model_1 (sklearn.base.BaseEstimator): The first scikit-learn model to consider.
        sklearn_model_2 (sklearn.base.BaseEstimator): The second scikit-learn model to consider.
        x (numpy.ndarray): The input to use for running the predictions.

    Returns:
        bool: If predictions from both models are equal.
    """

    predictions_1 = sklearn_model_1.predict(x)
    predictions_2 = sklearn_model_2.predict(x)
    return values_are_equal(predictions_1, predictions_2)


def get_a_fhe_circuit() -> Circuit:
    """Generate an arbitrary Circuit object.

    Returns:
        Circuit: An arbitrary circuit object.
    """
    # Create the data for regression
    # pylint: disable-next=unbalanced-tuple-unpacking
    x, y = make_regression()

    # Instantiate, fit and compile a linear regression model from Scikit Learn in order to retrieve
    # its underlying FHE Circuit
    model = LinearRegression()
    model.fit(x, y)
    fhe_circuit = model.compile(x)

    return fhe_circuit


@pytest.mark.parametrize(
    "random_state, random_state_type",
    [pytest.param(None, None), pytest.param(0, int), pytest.param(RandomState(0), RandomState)],
)
def test_serialize_random_state(random_state, random_state_type):
    """Test serialization of random_state objects."""
    check_serialization(random_state, random_state_type)


@pytest.mark.parametrize(
    "concrete_model_class",
    _get_sklearn_linear_models() + _get_sklearn_tree_models(),
)
def test_serialize_sklearn_model(concrete_model_class, load_data):
    """Test serialization of sklearn_model objects."""
    # Create the data
    x, y = load_data(concrete_model_class)

    # Instantiate and fit a Concrete model to recover its underlying Scikit Learn model
    concrete_model = concrete_model_class()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        _, sklearn_model = concrete_model.fit_benchmark(x, y)

    # Both JSON string are not compared as scikit-learn models are serialized using Skops or pickle,
    # which does not make string comparison possible
    check_serialization(
        sklearn_model,
        sklearn.base.BaseEstimator,
        equal_method=partial(sklearn_predictions_are_equal, x=x),
        check_str=False,
    )


def test_serialize_onnx():
    """Test serialization of onnx graphs."""
    inputs = torch.zeros(10)[None, ...]
    model = SimpleNet()
    model(inputs)
    io_stream = io.BytesIO(initial_bytes=b"")
    torch.onnx.export(
        model=model,
        args=inputs,
        f=io_stream,
    )
    value = onnx.load_model_from_string(io_stream.getvalue())

    check_serialization(value, onnx.ModelProto)


def test_serialize_set():
    """Test serialization of set objects."""
    value = {1, 2, 3, 4}

    check_serialization(value, set)


def test_serialize_tuple():
    """Test serialization of tuple objects."""
    value = (1, 2, 3, 4)

    check_serialization(value, tuple)


@pytest.mark.parametrize(
    "dtype",
    [
        numpy.int8,
        numpy.int16,
        numpy.int32,
        numpy.int64,
    ],
)
def test_serialize_numpy_integer(dtype):
    """Test serialization of numpy.integer objects."""
    value = numpy.int64(10).astype(dtype)

    check_serialization(value, numpy.integer)


@pytest.mark.parametrize(
    "dtype",
    [
        numpy.float32,
        numpy.float64,
    ],
)
def test_serialize_numpy_float(dtype):
    """Test serialization of numpy.floating objects."""
    value = numpy.float64(10.2).astype(dtype)

    check_serialization(value, numpy.floating)


@pytest.mark.parametrize(
    "dtype",
    [
        numpy.int8,
        numpy.int16,
        numpy.int32,
        numpy.int64,
        numpy.float32,
        numpy.float64,
    ],
)
def test_serialize_numpy_array(dtype):
    """Test serialization of numpy.ndarray objects."""
    value = numpy.random.random((10, 10, 10, 3)).astype(dtype)

    check_serialization(value, numpy.ndarray)


# Test the most important types
@pytest.mark.parametrize(
    "value",
    SUPPORTED_TORCH_ACTIVATIONS + _get_sklearn_all_models() + [QuantizedModule],
)
def test_serialize_type(value):
    """Test serialization of type objects (trusted by Skops)."""
    value = torch.nn.modules.activation.ReLU

    check_serialization(value, type, check_str=False)


def test_serialize_torch_device():
    """Test serialization of torch device objects."""
    value = torch.device("cpu")
    check_serialization(value, torch.device)


@pytest.mark.parametrize(
    "cross_validation_split, random_state",
    [
        pytest.param(None, None),
        pytest.param(5, None),
        pytest.param(0.5, None),
        pytest.param(0.5, 0),
        pytest.param(0.5, RandomState(0)),
        pytest.param([1, 2, 3], None),
    ],
)
@pytest.mark.parametrize(
    "stratified",
    [
        True,
        False,
    ],
)
def test_serialize_valid_split(cross_validation_split, stratified, random_state):
    """Test serialization of ValidSplit skorch objects."""
    value = ValidSplit(
        cv=cross_validation_split,
        stratified=stratified,
        random_state=random_state,
    )

    check_serialization(value, ValidSplit, equal_method=valid_split_instances_are_equal)


@pytest.mark.parametrize(
    "unsupported_object, expected_error, expected_message",
    [
        pytest.param(
            lambda x: x + 1,
            NotImplementedError,
            (
                "Serializing a custom Callable or Generator object is not secure and is therefore "
                "disabled.*"
            ),
        ),
        pytest.param(
            (x for x in [1]),
            NotImplementedError,
            (
                "Serializing a custom Callable or Generator object is not secure and is therefore "
                "disabled.*"
            ),
        ),
        pytest.param(
            ValidSplit(cv=(x for x in [1])),
            NotImplementedError,
            (
                "Serializing a custom Generator object is not secure and is therefore "
                "disabled. Please choose a different cross-validation splitting strategy."
            ),
        ),
        pytest.param(
            torch.Tensor([3, 4]),
            TypeError,
            "Object of type Tensor is not JSON serializable",
        ),
        # Serializing a Circuit object is currently not supported
        # FIXME: https://github.com/zama-ai/concrete-numpy-internal/issues/1841
        pytest.param(
            get_a_fhe_circuit(),
            NotImplementedError,
            "Concrete Circuit object serialization is not implemented.",
        ),
    ],
)
def test_error_raises_dumps(unsupported_object, expected_error, expected_message):
    """Test that trying to dump unsupported object correctly raises an error."""
    with pytest.raises(expected_error, match=expected_message):
        dumps(unsupported_object)


@pytest.mark.parametrize(
    "unsupported_object, expected_error, expected_message",
    [
        pytest.param(
            {
                "type_name": "wrong_serialization",
                "serialized_value": None,
            },
            NotImplementedError,
            "wrong_serialization does not support the `load_dict` method.",
        ),
        pytest.param(
            RandomState,
            UntrustedTypesFoundException,
            "Untrusted types found in the file:.*",
        ),
    ],
)
def test_error_raises_loads(unsupported_object, expected_error, expected_message):
    """Test that trying to load unsupported object correctly raises an error."""

    if expected_error == UntrustedTypesFoundException and not USE_SKOPS:
        return

    # Loading an object of an unexpected serialized should throw an error
    wrong_serialization_str = dumps(unsupported_object)
    with pytest.raises(expected_error, match=expected_message):
        loads(wrong_serialization_str)


def test_torch_activations():
    """Test supported and unsupported torch activation list."""

    # Torch activation list defined in Concrete ML
    all_torch_activations_cml = [
        activation.__name__
        for activation in SUPPORTED_TORCH_ACTIVATIONS + UNSUPPORTED_TORCH_ACTIVATIONS
    ]

    # Torch activation list imported from Torch
    all_torch_activations_torch = [
        activation.__name__
        for _, activation in inspect.getmembers(nn.modules.activation)
        if inspect.isclass(activation) and "torch.nn.modules.activation" in str(activation)
    ]

    assert sorted(all_torch_activations_cml) == sorted(all_torch_activations_torch), (
        "Difference found between activations imported from Torch and the ones considered in "
        "Concrete ML: "
        f"{list(set(all_torch_activations_cml).symmetric_difference(all_torch_activations_torch))}"
    )
