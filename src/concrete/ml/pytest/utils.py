"""Common functions or lists for test files, which can't be put in fixtures."""
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy
import pytest
import torch
from torch import nn

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


def instantiate_model_generic(model_class, **parameters):
    """Instantiate any Concrete ML model type.

    Args:
        model_class (class): The type of the model to instantiate
        parameters (dict): Hyper-parameters for the model instantiation

    Returns:
        model_name (str): The type of the model as a string
        model (object): The model instance
    """

    assert "n_bits" in parameters
    n_bits = parameters["n_bits"]

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

        model = model_class(**extra_kwargs)

    # Else, set the model using n_bits
    else:
        model = model_class(n_bits=n_bits)

    # Seed the model
    model_params = model.get_params()
    if "random_state" in model_params:
        model_params["random_state"] = numpy.random.randint(0, 2**15)

    model.set_params(**model_params)

    return model


def get_torchvision_dataset(
    param: Dict,
    train_set: bool,
):
    """Get train or testing data-set.

    Args:
        param (Dict): Set of hyper-parameters to use based on the selected torchvision data-set.
            It must contain: data-set transformations (torchvision.transforms.Compose), and the
            data-set_size (Optional[int]).
        train_set (bool): Use train data-set if True, else testing data-set

    Returns:
        A torchvision data-sets.
    """

    transform = param["train_transform"] if train_set else param["test_transform"]
    dataset = param["dataset"](download=True, root="./data", train=train_set, transform=transform)

    if param.get("dataset_size", None):
        dataset = torch.utils.data.random_split(
            dataset,
            [param["dataset_size"], len(dataset) - param["dataset_size"]],
        )[0]

    return dataset


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
    n_sample = min(len(data), n_sample)

    # Generates a random sample from a given 1-D array
    random_sample = numpy.random.choice(len(data), n_sample, replace=False)

    if (
        hasattr(data, "__getitem__") and hasattr(data, "__len__") and hasattr(data, "train")
    ) or isinstance(data, torch.utils.data.dataset.Subset):
        assert targets is None, "dataset includes inputs and targets"
        splitted_dataset = list(zip(*data))
        x, y = numpy.stack(splitted_dataset[0]), numpy.array(splitted_dataset[1])

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
