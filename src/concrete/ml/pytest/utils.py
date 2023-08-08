"""Common functions or lists for test files, which can't be put in fixtures."""
import io
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy
import pytest
import torch
from numpy.random import RandomState
from torch import nn

from ..common.serialization.dumpers import dump, dumps
from ..common.serialization.loaders import load, loads
from ..common.utils import get_model_class, get_model_name, is_model_class_in_a_list, is_pandas_type
from ..sklearn import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ElasticNet,
    GammaRegressor,
    Lasso,
    LinearRegression,
    LinearSVC,
    LinearSVR,
    LogisticRegression,
    NeuralNetClassifier,
    NeuralNetRegressor,
    PoissonRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    Ridge,
    TweedieRegressor,
    XGBClassifier,
    XGBRegressor,
    get_sklearn_neural_net_models,
)

_regressor_models = [
    XGBRegressor,
    GammaRegressor,
    LinearRegression,
    Lasso,
    Ridge,
    ElasticNet,
    LinearSVR,
    PoissonRegressor,
    TweedieRegressor,
    partial(TweedieRegressor, link="auto", power=0.0),
    partial(TweedieRegressor, link="auto", power=2.8),
    partial(TweedieRegressor, link="log", power=1.0),
    partial(TweedieRegressor, link="identity", power=0.0),
    DecisionTreeRegressor,
    RandomForestRegressor,
    partial(
        NeuralNetRegressor,
        module__n_layers=3,
        module__n_w_bits=2,
        module__n_a_bits=2,
        module__n_accum_bits=7,  # Stay with 7 bits for test exec time
        module__n_hidden_neurons_multiplier=1,
        module__activation_function=nn.ReLU,
        max_epochs=10,
        verbose=0,
        callbacks="disable",
    ),
]

_classifier_models = [
    DecisionTreeClassifier,
    RandomForestClassifier,
    XGBClassifier,
    LinearSVC,
    LogisticRegression,
    partial(
        NeuralNetClassifier,
        module__n_layers=3,
        module__activation_function=nn.ReLU,
        max_epochs=10,
        verbose=0,
        callbacks="disable",
    ),
]

# Get the data-sets. The data generation is seeded in load_data.
_classifiers_and_datasets = [
    pytest.param(
        model,
        {
            "n_samples": 1000,
            "n_features": 10,
            "n_classes": n_classes,
            "n_informative": 10,
            "n_redundant": 0,
        },
        id=get_model_name(model),
    )
    for model in _classifier_models
    for n_classes in [2, 4]
]

# Get the data-sets. The data generation is seeded in load_data.
# Only LinearRegression supports multi targets
# GammaRegressor, PoissonRegressor and TweedieRegressor only handle positive target values
_regressors_and_datasets = [
    pytest.param(
        model,
        {
            "n_samples": 200,
            "n_features": 10,
            "n_informative": 10,
            "n_targets": 2 if model == LinearRegression else 1,
            "noise": 0,
        },
        id=get_model_name(model),
    )
    for model in _regressor_models
]

# All scikit-learn models in Concrete ML
sklearn_models_and_datasets = _classifiers_and_datasets + _regressors_and_datasets


def get_random_extract_of_sklearn_models_and_datasets():
    """Return a random sublist of sklearn_models_and_datasets.

    The sublist contains exactly one model of each kind.

    Returns:
        the sublist

    """
    unique_model_classes = []
    done = {}

    for m in sklearn_models_and_datasets:
        t = m.values
        typ = get_model_class(t[0])

        if typ not in done:
            done[typ] = True
            unique_model_classes.append(m)

    # To avoid to make mistakes and return empty list
    assert len(sklearn_models_and_datasets) == 28
    assert len(unique_model_classes) == 18

    return unique_model_classes


def instantiate_model_generic(model_class, n_bits, **parameters):
    """Instantiate any Concrete ML model type.

    Args:
        model_class (class): The type of the model to instantiate.
        n_bits (int): The number of quantization to use when initializing the model. For QNNs,
            default parameters are used based on whether `n_bits` is greater or smaller than 8.
        parameters (dict): Hyper-parameters for the model instantiation. For QNNs, these parameters
            will override the matching default ones.

    Returns:
        model_name (str): The type of the model as a string.
        model (object): The model instance.
    """
    # If the model is a QNN, set the model using appropriate bit-widths
    if is_model_class_in_a_list(model_class, get_sklearn_neural_net_models()):
        extra_kwargs = {}
        if n_bits > 8:
            extra_kwargs["module__n_w_bits"] = 3
            extra_kwargs["module__n_a_bits"] = 3
            extra_kwargs["module__n_accum_bits"] = 12
        else:
            extra_kwargs["module__n_w_bits"] = 2
            extra_kwargs["module__n_a_bits"] = 2
            extra_kwargs["module__n_accum_bits"] = 7

        extra_kwargs.update(parameters)
        model = model_class(**extra_kwargs)

    # Else, set the model using n_bits
    else:
        model = model_class(n_bits=n_bits, **parameters)

    # Seed the model if it handles "random_state" as a parameter
    model_params = model.get_params()
    if "random_state" in model_params:
        model_params["random_state"] = numpy.random.randint(0, 2**15)

        model.set_params(**model_params)

    return model


def data_calibration_processing(data, n_sample: int, targets=None):
    """Reduce size of the given data-set.

    Args:
        data: The input container to consider
        n_sample (int): Number of samples to keep if the given data-set
        targets: If `dataset` is a `torch.utils.data.Dataset`, it typically contains both the data
            and the corresponding targets. In this case, `targets` must be set to `None`.
            If `data` is instance of `torch.Tensor` or 'numpy.ndarray`, `targets` is expected.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: The input data and the target (respectively x and y).

    Raises:
        TypeError: If the 'data-set' does not match any expected type.
    """
    assert n_sample >= 1, "`n_sample` must be greater than or equal to `1`"

    if isinstance(data, torch.utils.data.dataloader.DataLoader):
        n_elements = 0
        all_y, all_x = [], []
        # Iterate over n batches, to apply any necessary torch transformations to the data-set
        for x_batch, y_batch in data:
            all_x.append(x_batch.numpy())
            all_y.append(y_batch.numpy())
            # Number of elements per batch
            n_elements += y_batch.shape[0]
            # If the number of elements within n batches is >= `n_element`, break the loop
            # `n_sample` is reached.
            if n_sample <= n_elements:
                break

        x, y = numpy.concatenate(all_x), numpy.concatenate(all_y)
    elif targets is not None and is_pandas_type(data) and is_pandas_type(targets):
        x = data.to_numpy()
        y = targets.to_numpy()

    elif (
        targets is not None
        and isinstance(targets, (List, numpy.ndarray, torch.Tensor))  # type: ignore[arg-type]
        and isinstance(data, (numpy.ndarray, torch.Tensor))
    ):
        x = numpy.array(data)
        y = numpy.array(targets)
    else:
        raise TypeError(
            "Only numpy arrays, torch tensors and torchvision data-sets are supported. "
            f"Got `{type(data)}` as input type and `{type(targets)}` as target type"
        )

    n_sample = min(len(x), n_sample)

    # Generates a random sample from a given 1-D array
    random_sample = numpy.random.choice(len(x), n_sample, replace=False)

    x = x[random_sample]
    y = y[random_sample]

    return x, y


def load_torch_model(
    model_class: torch.nn.Module,
    state_dict_or_path: Optional[Union[str, Path, Dict[str, Any]]],
    params: Dict,
    device: str = "cpu",
) -> torch.nn.Module:
    """Load an object saved with torch.save() from a file or dict.

    Args:
        model_class (torch.nn.Module): A PyTorch or Brevitas network.
        state_dict_or_path (Optional[Union[str, Path, Dict[str, Any]]]): Path or state_dict
        params (Dict): Model's parameters
        device (str):  Device type.

    Returns:
        torch.nn.Module: A PyTorch or Brevitas network.
    """
    model = model_class(**params)

    if state_dict_or_path is not None:
        if isinstance(state_dict_or_path, (str, Path)):
            state_dict = torch.load(state_dict_or_path, map_location=device)
        else:
            state_dict = state_dict_or_path
        model.load_state_dict(state_dict)

    return model


# pylint: disable-next=too-many-return-statements
def values_are_equal(value_1: Any, value_2: Any) -> bool:
    """Indicate if two values are equal.

    This method takes into account objects of type None, numpy.ndarray, numpy.floating,
    numpy.integer, numpy.random.RandomState or any instance that provides a `__eq__` method.

    Args:
        value_2 (Any): The first value to consider.
        value_1 (Any): The second value to consider.

    Returns:
        bool: If the two values are equal.
    """
    if value_1 is None:
        return value_2 is None

    if isinstance(value_1, numpy.ndarray):
        return (
            isinstance(value_2, numpy.ndarray)
            and numpy.array_equal(value_1, value_2)
            and value_1.dtype == value_2.dtype
        )

    if isinstance(value_1, (numpy.floating, numpy.integer)):
        return (
            isinstance(value_2, (numpy.floating, numpy.integer))
            and value_1 == value_2
            and value_1.dtype == value_2.dtype
        )

    if isinstance(value_1, RandomState):
        if isinstance(value_2, RandomState):
            state_1, state_2 = value_1.get_state(), value_2.get_state()

            # Check that values from both states are equal
            for elt_1, elt_2 in zip(state_1, state_2):
                if not numpy.array_equal(elt_1, elt_2):
                    return False
            return True
        return False

    return value_1 == value_2


def check_serialization(
    object_to_serialize: Any,
    expected_type: Type,
    equal_method: Optional[Callable] = None,
    check_str: bool = True,
):
    """Check that the given object can properly be serialized.

    This function serializes all objects using the `dump`, `dumps`, `load` and `loads` functions
    from Concrete ML. If the given object provides a `dump` and `dumps` method, they are also
    serialized using these.

    Args:
        object_to_serialize (Any): The object to serialize.
        expected_type (Type): The object's expected type.
        equal_method (Optional[Callable]): The function to use to compare the two loaded objects.
            Default to `values_are_equal`.
        check_str (bool): If the JSON strings should also be checked. Default to True.
    """
    # apidocs does not work properly when the function is directly in the default value.
    if equal_method is None:
        equal_method = values_are_equal

    assert (
        isinstance(object_to_serialize, expected_type)
        if expected_type is not None
        else object_to_serialize is None
    )

    dump_method_to_test = [False]

    # If the given object provides a `dump`, `dumps` `dump_dict` method (which indicates that they
    # are Concrete ML serializable classes), run the check using these as well
    if (
        hasattr(object_to_serialize, "dump")
        and hasattr(object_to_serialize, "dumps")
        and hasattr(object_to_serialize, "dump_dict")
    ):
        dump_method_to_test.append(True)

    for use_dump_method in dump_method_to_test:

        # Dump the object as a string
        if use_dump_method:
            dumped_str = object_to_serialize.dumps()
        else:
            dumped_str = dumps(object_to_serialize)

        # Load the object using a string
        loaded = loads(dumped_str)

        # Assert that the loaded object is equal to the initial one
        assert isinstance(loaded, expected_type) if expected_type is not None else loaded is None, (
            f"Loaded object (from string) is not of the expected type. Expected {expected_type}, "
            f"got {type(loaded)}."
        )
        assert equal_method(object_to_serialize, loaded), (
            "Loaded object (from string) is not equal to the initial one, using equal method "
            f"{equal_method}."
        )

        if check_str:
            # Dump the loaded object as a string
            if use_dump_method:
                re_dumped_str = loaded.dumps()
            else:
                re_dumped_str = dumps(loaded)

            # Assert that both JSON strings are equal. This will not work if some objects were
            # dumped using Skops or pickle (such as a scikit-learn model or a Type object)
            assert isinstance(dumped_str, str) and isinstance(re_dumped_str, str), (
                f"One of the dumped strings is not a string. Got {type(dumped_str)} and "
                f"{type(re_dumped_str)}."
            )
            assert dumped_str == re_dumped_str, (
                "Dumped strings are not equal. This could come from objects that were serialized "
                "in binary strings using Skops or pickle. If this cannot be avoided, please set "
                "`check_str` to False."
            )

        # Define a buffer to test serialization in a file
        with io.StringIO() as buffer:
            buffer.seek(0, 0)

            # Dump the object in a file
            if use_dump_method:
                object_to_serialize.dump(buffer)
            else:
                dump(object_to_serialize, buffer)

            buffer.seek(0, 0)

            # Load the object from the file
            loaded = load(buffer)

            # Assert that the loaded object is equal to the initial one
            assert (
                isinstance(loaded, expected_type) if expected_type is not None else loaded is None
            ), (
                "Loaded object (from file) is not of the expected type. Expected "
                f"{expected_type}, got {type(loaded)}."
            )
            assert equal_method(object_to_serialize, loaded), (
                "Loaded object (from file) is not equal to the initial one, using equal method "
                f"{equal_method}."
            )
