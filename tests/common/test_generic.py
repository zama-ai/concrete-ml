"""Tests for the sklearn decision trees."""
import warnings
from functools import partial
from typing import Any, Dict, List

import numpy
import pytest
from sklearn.base import is_classifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import make_scorer, matthews_corrcoef, mean_squared_error
from sklearn.model_selection import GridSearchCV
from torch import nn

from concrete.ml.pytest.utils import classifiers, regressors
from concrete.ml.sklearn.base import get_sklearn_linear_models, get_sklearn_tree_models


# pylint: disable-next=too-many-arguments
def check_generic(
    model_class,
    parameters,
    use_virtual_lib,
    n_bits,
    load_data,
    default_configuration,
    check_is_good_execution_for_quantized_models,
    test_hyper_parameters=True,
    test_grid_search=True,
    test_double_fit=True,
    test_offset=True,
):
    """Generic tests.

    This function tests:
      - model (with n_bits)
      - virtual_lib or not
      - fit
      - compile
      - grid search
      - hyper parameters
      - offset

    Are currently missing
      - fit_benchmark
      - quantization / r2 score
      - pandas
      - pipeline

    """

    # Get the dataset. The data generation is seeded in load_data.
    model_name = (
        model_class.__name__ if not isinstance(model_class, partial) else model_class.func.__name__
    )
    x, y = load_data(**parameters, model_name=model_name)

    # Testing hyper parameters
    if test_hyper_parameters:
        check_hyper_parameters(model_class)

    # Set the model
    model = model_class(n_bits=n_bits)

    model_params = model.get_params()
    if "random_state" in model_params:
        model_params["random_state"] = numpy.random.randint(0, 2**15)

    model.set_params(**model_params)

    with warnings.catch_warnings():
        # Sometimes, we miss convergence, which is not a problem for our test
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(x, y)

    # Do a single inference in clear
    y_pred = model.predict(x[:1])

    with warnings.catch_warnings():
        # Use virtual lib to not have issues with precision
        model.compile(x, default_configuration, use_virtual_lib=use_virtual_lib)

    # Grid search
    if test_grid_search:
        check_grid_search(model, model_name, model_class, x, y)

    # Double fit
    if test_double_fit:
        check_double_fit(model_class, x, y)

    # Test with offset
    if test_offset:
        check_offset(model_class, x, y)

    # Compare FHE vs non-FHE
    test_execution = use_virtual_lib  # For now, don't check in FHE

    if test_execution:

        if "NeuralNetRegressor" in model_name:
            # FIXME #2403: doesn't work perfectly, to be checked
            # Not even sure why this is just for the regressor
            return

        check_is_good_execution_for_quantized_models(x=x[:5], model_predict=model.predict)
    else:
        # At least, check without execute_in_fhe
        y_pred_fhe = model.predict(x[:1], execute_in_fhe=False)

        # Check that the output shape is correct
        assert y_pred_fhe.shape == y_pred.shape
        assert numpy.array_equal(y_pred_fhe, y_pred)


def check_double_fit(model_class, x, y):
    """Check double fit."""
    model_name = (
        model_class.__name__ if not isinstance(model_class, partial) else model_class.func.__name__
    )

    model = model_class()

    model_params = model.get_params()
    if "random_state" in model_params:
        model_params["random_state"] = numpy.random.randint(0, 2**15)
    model.set_params(**model_params)

    if "NeuralNet" in model_name:
        # FIXME #918: this fails and needs to be fixed
        return

    # Sometimes, we miss convergence, which is not a problem for our test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)

        # First fit
        model.fit(x, y)
        y_pred_one = model.predict(x)

        # Second fit
        model.fit(x, y)
        y_pred_two = model.predict(x)

    assert numpy.array_equal(y_pred_one, y_pred_two)


def check_offset(model_class, x, y):
    """Check offset."""
    model_name = (
        model_class.__name__ if not isinstance(model_class, partial) else model_class.func.__name__
    )

    if "XGB" in model_name:
        # Offsets are not supported by XGBoost
        return

    if "NeuralNet" in model_name:
        # FIXME #2404: check it is not supported
        return

    model = model_class()

    model_params = model.get_params()
    if "random_state" in model_params:
        model_params["random_state"] = numpy.random.randint(0, 2**15)
    model.set_params(**model_params)

    # Sometimes, we miss convergence, which is not a problem for our test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)

        # Add the offset
        y += 3
        model.fit(x, y)
        model.predict(x[:1])

        # Another offset
        y -= 2
        model.fit(x, y)


def check_grid_search(model, model_name, model_class, x, y):
    """Check grid search."""
    if is_classifier(model_class):
        grid_scorer = make_scorer(matthews_corrcoef, greater_is_better=True)
    else:
        grid_scorer = make_scorer(mean_squared_error, greater_is_better=True)

    if "NeuralNet" in model_name:
        param_grid = {
            "module__n_layers": (3, 5),
            "module__activation_function": (nn.Tanh, nn.ReLU6),
        }
    elif model_class in get_sklearn_tree_models(str_in_class_name="DecisionTree"):
        param_grid = {
            "n_bits": [20],
        }
    elif model_class in get_sklearn_tree_models():
        param_grid = {
            "n_bits": [20],
            "max_depth": [2],
            "n_estimators": [5, 10, 50, 100],
            "n_jobs": [1],
        }
    else:
        param_grid = {
            "n_bits": [20],
        }

    with warnings.catch_warnings():
        # Sometimes, we miss convergence, which is not a problem for our test
        warnings.simplefilter("ignore", category=ConvergenceWarning)

        _ = GridSearchCV(
            model, param_grid, cv=5, scoring=grid_scorer, error_score="raise", n_jobs=1
        ).fit(x, y)


def check_hyper_parameters(model_class):
    """Check hyper parameters."""
    hyper_param_combinations: Dict[str, List[Any]]

    if model_class in get_sklearn_linear_models():
        hyper_param_combinations = {"n_bits": [2, 6], "fit_intercept": [0, 1]}
    elif model_class in get_sklearn_tree_models(str_in_class_name="DecisionTree"):
        hyper_param_combinations = {}
    elif model_class in get_sklearn_tree_models(str_in_class_name="RandomForest"):
        hyper_param_combinations = {
            "max_depth": [3, 4, 5, 10],
            "min_samples_split": [2, 3, 4, 5],
            "min_samples_leaf": [1, 2, 3, 4],
            "min_weight_fraction_leaf": [0.0, 0.1, 0.2, 0.3],
            "max_features": ["sqrt", "log2"],
            "max_leaf_nodes": [None, 5, 10, 20],
        }
    elif model_class in get_sklearn_tree_models(str_in_class_name="XGB"):
        hyper_param_combinations = {
            "max_depth": [3, 4, 5, 10],
            "learning_rate": [1, 0.5, 0.1],
            "n_estimators": [1, 50, 100, 1000],
            "tree_method": ["auto", "exact", "approx"],
            "gamma": [0, 0.1, 0.5],
            "min_child_weight": [1, 5, 10],
            "max_delta_step": [0, 0.5, 0.7],
            "subsample": [0.5, 0.9, 1.0],
            "colsample_bytree": [0.5, 0.9, 1.0],
            "colsample_bylevel": [0.5, 0.9, 1.0],
            "colsample_bynode": [0.5, 0.9, 1.0],
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [0, 0.1, 0.5],
            "scale_pos_weight": [0.5, 0.9, 1.0],
            "importance_type": ["weight", "gain"],
            "base_score": [0.5, None],
        }
    else:
        hyper_param_combinations = {
            "n_bits": [2, 6],
            "max_depth": [3, 4, 5, 10],
            "learning_rate": [1, 0.5, 0.1],
            "n_estimators": [1, 50, 100, 1000],
            "tree_method": ["auto", "exact", "approx"],
            "gamma": [0, 0.1, 0.5],
            "min_child_weight": [1, 5, 10],
            "max_delta_step": [0, 0.5, 0.7],
            "subsample": [0.5, 0.9, 1.0],
            "colsample_bytree": [0.5, 0.9, 1.0],
            "colsample_bylevel": [0.5, 0.9, 1.0],
            "colsample_bynode": [0.5, 0.9, 1.0],
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [0, 0.1, 0.5],
            "scale_pos_weight": [0.5, 0.9, 1.0],
            "importance_type": ["weight", "gain"],
            "base_score": [0.5, None],
        }

    # Prepare the list of all hyper parameters
    hyperparameters_list = [
        {key: value} for key, values in hyper_param_combinations.items() for value in values
    ]

    for hyper_parameters in hyperparameters_list:
        _ = model_class(**hyper_parameters)


@pytest.mark.parametrize("model, parameters", classifiers + regressors)
@pytest.mark.parametrize(
    "use_virtual_lib",
    [
        pytest.param(False, id="no_virtual_lib"),
        pytest.param(True, id="use_virtual_lib"),
    ],
)
@pytest.mark.parametrize(
    "n_bits",
    [
        pytest.param(6, id="n_bits_6"),
        pytest.param(2, id="n_bits_2"),
    ],
)
def test_generic(
    model,
    parameters,
    use_virtual_lib,
    n_bits,
    load_data,
    default_configuration,
    check_is_good_execution_for_quantized_models,
):
    """Test in a generic way."""
    check_generic(
        model,
        parameters,
        use_virtual_lib,
        n_bits,
        load_data,
        default_configuration,
        check_is_good_execution_for_quantized_models,
    )
