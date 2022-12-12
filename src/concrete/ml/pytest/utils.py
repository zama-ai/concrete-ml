"""Common functions or lists for test files, which can't be put in fixtures."""
from functools import partial

import numpy
import pytest
from torch import nn

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

regressor_models = [
    XGBRegressor,
    GammaRegressor,
    LinearRegression,
    Lasso,
    Ridge,
    ElasticNet,
    LinearSVR,
    PoissonRegressor,
    TweedieRegressor,
    DecisionTreeRegressor,
    RandomForestRegressor,
    partial(
        NeuralNetRegressor,
        module__n_layers=3,
        module__n_w_bits=2,
        module__n_a_bits=2,
        module__n_accum_bits=7,  # Let's stay with 7 bits for test exec time
        module__n_hidden_neurons_multiplier=1,
        module__n_outputs=1,
        module__input_dim=10,
        module__activation_function=nn.ReLU,
        max_epochs=10,
        verbose=0,
    ),
]

classifier_models = [
    DecisionTreeClassifier,
    RandomForestClassifier,
    XGBClassifier,
    LinearSVC,
    LogisticRegression,
    partial(
        NeuralNetClassifier,
        module__n_layers=3,
        module__n_w_bits=2,
        module__n_a_bits=2,
        module__n_accum_bits=7,  # Let's stay with 7 bits for test exec time.
        module__n_outputs=2,
        module__input_dim=10,
        module__activation_function=nn.ReLU,
        max_epochs=10,
        verbose=0,
    ),
]

# Get the datasets. The data generation is seeded in load_data.
classifiers = [
    pytest.param(
        model,
        {
            "dataset": "classification",
            "n_samples": 1000,
            "n_features": 10,
            "n_classes": 2
            if isinstance(model, partial) and "NeuralNet" in model.func.__name__
            else n_classes,  # FIXME #2402, qnns do not have multiclass yet
            "n_informative": 10,
            "n_redundant": 0,
        },
        id=model.__name__ if not isinstance(model, partial) else model.func.__name__,
    )
    for model in classifier_models
    for n_classes in [2, 4]
]

# Get the datasets. The data generation is seeded in load_data.
# Only LinearRegression supports multi targets
# GammaRegressor, PoissonRegressor and TweedieRegressor only handle positive target values
regressors = [
    pytest.param(
        model,
        {
            "dataset": "regression",
            "strictly_positive": model in [GammaRegressor, PoissonRegressor, TweedieRegressor],
            "n_samples": 200,
            "n_features": 10,
            "n_informative": 10,
            "n_targets": 2 if model == LinearRegression else 1,
            "noise": 0,
        },
        id=model.__name__ if not isinstance(model, partial) else model.func.__name__,
    )
    for model in regressor_models
]


def sanitize_test_and_train_datasets(model, x, y):
    """Sanitize datasets depending on the model type.

    Args:
        model: the model
        x: the first output of load_data, i.e., the inputs
        y: the second output of load_data, i.e., the labels

    Returns:
        Tuple containing sanitized (model_params, x, y, x_train, y_train, x_test)

    """

    x_train = x[:-1]
    y_train = y[:-1]
    x_test = x[-1:]

    if isinstance(model, partial):
        if model.func in [NeuralNetClassifier, NeuralNetRegressor]:
            model_params = model.keywords
            # Change module__input_dim to be the same as the input dimension
            model_params["module__input_dim"] = x_train.shape[1]
            # qnns require float32 as input
            x = x.astype(numpy.float32)
            x_train = x_train.astype(numpy.float32)
            x_test = x_test.astype(numpy.float32)

            if model.func is NeuralNetRegressor:
                # Reshape y_train and y_test if 1d (regression for neural nets)
                if y_train.ndim == 1:
                    y_train = y_train.reshape(-1, 1).astype(numpy.float32)
    elif model in [XGBClassifier, RandomForestClassifier, XGBRegressor]:
        model_params = {
            "n_estimators": 5,
            "max_depth": 2,
            "random_state": numpy.random.randint(0, 2**15),
        }
    elif model is DecisionTreeClassifier:
        model_params = {"max_depth": 2, "random_state": numpy.random.randint(0, 2**15)}
    elif model in [LogisticRegression]:
        model_params = {"random_state": numpy.random.randint(0, 2**15)}
    else:
        model_params = {}

    return model_params, x, y, x_train, y_train, x_test
