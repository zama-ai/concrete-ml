"""Common functions or lists for test files, which can't be put in fixtures."""
from functools import partial

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
# Remark that NeuralNetClassifier is not here because it is particular model for us, needs much more
# parameters
classifiers = [
    pytest.param(
        model,
        {
            "dataset": "classification",
            "n_samples": 1000,
            "n_features": 10,
            "n_classes": 2,  # qnns do not have multiclass yet
            "n_informative": 10,
            "n_redundant": 0,
        },
        id=model.__name__ if not isinstance(model, partial) else None,
    )
    for model in classifier_models
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
        id=model.__name__ if not isinstance(model, partial) else None,
    )
    for model in regressor_models
]
