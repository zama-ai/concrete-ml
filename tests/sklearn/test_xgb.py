"""Tests for the xgboost."""

from typing import Any, Dict, List

import numpy
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.metrics import make_scorer, matthews_corrcoef, mean_squared_error
from sklearn.model_selection import GridSearchCV

from concrete.ml.sklearn import XGBClassifier, XGBRegressor

PARAMS_XGB: Dict[str, List[Any]] = {
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


# Classifier


@pytest.mark.parametrize(
    "hyperparameters",
    [
        pytest.param({key: value}, id=f"{key}={value}")
        for key, values in PARAMS_XGB.items()
        for value in values  # type: ignore
    ],
)
@pytest.mark.parametrize("n_classes", [2, 4])
def test_xgb_classifier_hyperparameters(
    hyperparameters, n_classes, load_data, check_r2_score, check_accuracy
):
    """Test that the hyperparameters are valid."""

    # Get the datasets. The data generation is seeded in load_data.
    x, y = load_data(
        dataset="classification",
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_classes=n_classes,
    )
    model = XGBClassifier(
        **hyperparameters,
        n_bits=20,
        n_jobs=1,
        random_state=numpy.random.randint(0, 2**15),
    )
    model, sklearn_model = model.fit_benchmark(x, y)
    # Check accuracy and r2 score between the two models predictions
    check_accuracy(model.predict(x), sklearn_model.predict(x))
    check_r2_score(model.predict_proba(x), sklearn_model.predict_proba(x))


# Get the datasets. The data generation is seeded in load_data.
# pylint: disable=too-many-arguments
@pytest.mark.parametrize(
    "parameters",
    [
        pytest.param({"dataset": lambda: load_breast_cancer(return_X_y=True)}, id="breast_cancer"),
        pytest.param(
            {
                "dataset": "classification",
                "n_samples": 1000,
                "n_features": 100,
                "n_classes": 2,
            },
            id="make_classification",
        ),
        pytest.param(
            {
                "dataset": "classification",
                "n_samples": 1000,
                "n_features": 100,
                "n_classes": 4,
                "n_informative": 100,
                "n_redundant": 0,
            },
            id="make_multiclassification",
        ),
    ],
)
@pytest.mark.parametrize(
    "use_virtual_lib",
    [
        pytest.param(False, id="no_virtual_lib"),
        pytest.param(True, id="use_virtual_lib"),
    ],
)
@pytest.mark.parametrize(
    "max_depth, n_estimators, skip_if_not_weekly",
    [
        pytest.param(2, 1, False, id="max_depth_2_n_estimators_1"),
        pytest.param(2, 5, False, id="max_depth_2_n_estimators_5"),
        pytest.param(4, 10, True, id="max_depth_4_n_estimators_10"),
    ],
)
@pytest.mark.parametrize(
    "n_bits",
    [
        pytest.param(6, id="n_bits_6"),
        pytest.param(2, id="n_bits_2"),
    ],
)
def test_xgb_classifier(
    parameters,
    max_depth,
    n_estimators,
    skip_if_not_weekly,
    n_bits,
    load_data,
    default_configuration,
    check_is_good_execution_for_quantized_models,
    use_virtual_lib,
    is_weekly_option,
    is_vl_only_option,
):
    """Tests the xgboost.

    WARNING: Increasing the number of trees will increase the compilation / inference
        time and risk of out-of-memory errors.
    """
    if not is_weekly_option:
        if skip_if_not_weekly:
            # Skip long tests
            return

    if not use_virtual_lib and is_vl_only_option:
        print("Warning, skipping non VL tests")
        return

    # Get the dataset
    x, y = load_data(**parameters)

    model = XGBClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        n_bits=n_bits,
        random_state=numpy.random.randint(0, 2**15),
        n_jobs=1,
    )
    model.fit(x, y)

    # Test compilation
    model.compile(x, default_configuration, use_virtual_lib=use_virtual_lib)

    # Compare FHE vs non-FHE
    check_is_good_execution_for_quantized_models(x=x[:5], model_predict=model.predict)


def test_xgb_classifier_grid_search(load_data):
    """Tests xgboost with the gridsearchCV from sklearn."""

    # Get the dataset. The data generation is seeded in load_data.
    x, y = load_data(
        dataset="classification",
        n_samples=1000,
        n_features=100,
        n_classes=2,
    )

    param_grid = {
        "n_bits": [20],
        "max_depth": [2],
        "n_estimators": [5, 10, 50, 100],
        "n_jobs": [1],
    }

    grid_scorer = make_scorer(matthews_corrcoef, greater_is_better=True)

    concrete_clf = XGBClassifier()
    _ = GridSearchCV(
        concrete_clf, param_grid, cv=5, scoring=grid_scorer, error_score="raise", n_jobs=1
    ).fit(x, y)


# Regressor


@pytest.mark.parametrize(
    "hyperparameters",
    [
        pytest.param({key: value}, id=f"{key}={value}")
        for key, values in PARAMS_XGB.items()
        for value in values
        if not (key == "max_delta_step" and value != 0)  # type: ignore
        # R2 score is too low for max_delta_step if different from 0.
    ],
)
def test_xgb_regressor_hyperparameters(hyperparameters, load_data, check_r2_score):
    """Test that the hyperparameters are valid."""

    # Get the dataset. The data generation is seeded in load_data.
    x, y = load_data(
        dataset="regression",
        n_samples=100,
        n_features=10,
        n_informative=5,
    )
    model = XGBRegressor(
        **hyperparameters,
        n_bits=20,
        n_jobs=1,
        random_state=numpy.random.randint(0, 2**15),
    )
    model, sklearn_model = model.fit_benchmark(x, y)
    # Check r2 score between the two models predictions
    check_r2_score(model.predict(x), sklearn_model.predict(x))


# Get the datasets. The data generation is seeded in load_data.
# pylint: disable=too-many-arguments
@pytest.mark.parametrize(
    "parameters",
    [
        pytest.param({"dataset": lambda: load_diabetes(return_X_y=True)}, id="diabetes"),
        pytest.param(
            {
                "dataset": "regression",
                "n_samples": 1000,
                "n_features": 100,
            },
            id="make_regression",
        ),
        pytest.param(
            {
                "dataset": "regression",
                "n_samples": 1000,
                "n_features": 100,
                "n_informative": 100,
            },
            id="make_other_regression",
        ),
    ],
)
@pytest.mark.parametrize(
    "use_virtual_lib",
    [
        pytest.param(False, id="no_virtual_lib"),
        pytest.param(True, id="use_virtual_lib"),
    ],
)
@pytest.mark.parametrize(
    "max_depth, n_estimators, skip_if_not_weekly",
    [
        pytest.param(2, 1, False, id="max_depth_2_n_estimators_1"),
        pytest.param(2, 5, False, id="max_depth_2_n_estimators_5"),
        pytest.param(4, 10, True, id="max_depth_4_n_estimators_10"),
    ],
)
@pytest.mark.parametrize(
    "n_bits",
    [
        pytest.param(6, id="n_bits_6"),
        pytest.param(2, id="n_bits_2"),
    ],
)
def test_xgb_regressor(
    parameters,
    max_depth,
    n_estimators,
    skip_if_not_weekly,
    n_bits,
    load_data,
    default_configuration,
    check_is_good_execution_for_quantized_models,
    use_virtual_lib,
    is_weekly_option,
    is_vl_only_option,
):
    """Tests the xgboost.

    WARNING: Increasing the number of trees will increase the compilation / inference
        time and risk of out-of-memory errors.
    """
    if not is_weekly_option:
        if skip_if_not_weekly:
            # Skip long tests
            return

    if not use_virtual_lib and is_vl_only_option:
        print("Warning, skipping non VL tests")
        return

    # Get the dataset
    x, y = load_data(**parameters)

    model = XGBRegressor(
        max_depth=max_depth,
        n_estimators=n_estimators,
        n_bits=n_bits,
        random_state=numpy.random.randint(0, 2**15),
        n_jobs=1,
    )
    model.fit(x, y)

    # Test compilation
    model.compile(x, default_configuration, use_virtual_lib=use_virtual_lib)

    # Compare FHE vs non-FHE
    check_is_good_execution_for_quantized_models(x=x[:5], model_predict=model.predict)


def test_xgb_regressor_grid_search(load_data):
    """Tests xgboost with the gridsearchCV from sklearn."""

    # Get the dataset. The data generation is seeded in load_data.
    x, y = load_data(
        dataset="regression",
        n_samples=1000,
        n_features=100,
    )

    param_grid = {
        "n_bits": [20],
        "max_depth": [2],
        "n_estimators": [5, 10, 50, 100],
        "n_jobs": [1],
    }

    grid_scorer = make_scorer(mean_squared_error, greater_is_better=True)

    concrete_clf = XGBRegressor()
    _ = GridSearchCV(
        concrete_clf, param_grid, cv=5, scoring=grid_scorer, error_score="raise", n_jobs=1
    ).fit(x, y)
