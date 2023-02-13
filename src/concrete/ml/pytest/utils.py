"""Common functions or lists for test files, which can't be put in fixtures."""
from functools import partial

import numpy
import pytest
from torch import nn

from concrete.ml.common.utils import get_model_name, is_model_class_in_a_list
from concrete.ml.sklearn import (
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
)
from concrete.ml.sklearn.base import get_sklearn_neural_net_models

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
        module__n_accum_bits=7,  # Let's stay with 7 bits for test exec time
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

# Get the datasets. The data generation is seeded in load_data.
_classifiers_and_datasets = [
    pytest.param(
        model,
        {
            "dataset": "classification",
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

# Get the datasets. The data generation is seeded in load_data.
# Only LinearRegression supports multi targets
# GammaRegressor, PoissonRegressor and TweedieRegressor only handle positive target values
_regressors_and_datasets = [
    pytest.param(
        model,
        {
            "dataset": "regression",
            "strictly_positive": is_model_class_in_a_list(
                model, [GammaRegressor, PoissonRegressor, TweedieRegressor]
            ),
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

# All scikit-learn models in Concrete-ML
sklearn_models_and_datasets = _classifiers_and_datasets + _regressors_and_datasets


def instantiate_model_generic(model_class, **parameters):
    """Instantiate any Concrete-ML model type.

    Args:
        model_class (class): The type of the model to instantiate
        parameters (dict): Hyper-parameters for the model instantiation

    Returns:
        model_name (str): The type of the model as a string
        model (object): The model instance
    """
    model_name = get_model_name(model_class)

    assert "n_bits" in parameters
    n_bits = parameters["n_bits"]

    extra_kwargs = {}
    if is_model_class_in_a_list(model_class, get_sklearn_neural_net_models()):
        if n_bits > 8:
            extra_kwargs["module__n_w_bits"] = 3
            extra_kwargs["module__n_a_bits"] = 3
            extra_kwargs["module__n_accum_bits"] = 12
        else:
            extra_kwargs["module__n_w_bits"] = 2
            extra_kwargs["module__n_a_bits"] = 2
            extra_kwargs["module__n_accum_bits"] = 7

    # Set the model
    model = model_class(n_bits=n_bits, **extra_kwargs)

    # Seed the model
    model_params = model.get_params()
    if "random_state" in model_params:
        model_params["random_state"] = numpy.random.randint(0, 2**15)

    model.set_params(**model_params)

    return model_name, model
