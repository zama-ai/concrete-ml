"""Common functions or lists for test files, which can't be put in fixtures."""
from functools import partial

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
        module__n_outputs=1,
        module__input_dim=10,
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
            # Remove the following if statement once QNNs support multi-class data sets
            # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2402
            "n_classes": 2 if "NeuralNet" in get_model_name(model) else n_classes,
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
regressors = [
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
