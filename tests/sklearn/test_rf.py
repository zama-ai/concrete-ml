"""Tests for the random forest."""

import numpy
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.metrics import make_scorer, matthews_corrcoef, mean_squared_error
from sklearn.model_selection import GridSearchCV

from concrete.ml.sklearn import RandomForestClassifier, RandomForestRegressor

PARAMS_RF = {
    "max_depth": [2, 4, 6],
    "n_estimators": [5, 10, 20],
}


@pytest.mark.parametrize(
    "hyperparameters",
    [
        pytest.param({key: value}, id=f"{key}={value}")
        for key, values in PARAMS_RF.items()
        for value in values  # type: ignore
    ],
)
@pytest.mark.parametrize("n_classes", [2, 4])
def test_rf_classifier_hyperparameters(
    hyperparameters, n_classes, load_data, check_r2_score, check_accuracy
):
    """Test that the hyperparameters are valid."""

    # Get the dataset. The data generation is seeded in load_data.
    x, y = load_data(
        dataset="classification",
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_classes=n_classes,
    )
    model = RandomForestClassifier(
        **hyperparameters, n_bits=20, n_jobs=1, random_state=numpy.random.randint(0, 2**15)
    )
    model, sklearn_model = model.fit_benchmark(x, y)
    # Check accuracy and r2 score between the two models predictions
    check_accuracy(model.predict(x), sklearn_model.predict(x))
    check_r2_score(model.predict_proba(x), sklearn_model.predict_proba(x))


# Get the dataset. The data generation is seeded in load_data.
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
        pytest.param(4, 20, True, id="max_depth_4_n_estimators_20"),
    ],
)
@pytest.mark.parametrize(
    "n_bits",
    [
        pytest.param(6, id="n_bits_6"),
        pytest.param(2, id="n_bits_2"),
    ],
)
def test_rf_classifier(
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
    """Tests the random forest."""

    if not is_weekly_option:
        if skip_if_not_weekly:
            # Skip long tests
            return

    if not use_virtual_lib and is_vl_only_option:
        print("Warning, skipping non VL tests")
        return

    # Get the dataset
    x, y = load_data(**parameters)

    model = RandomForestClassifier(
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


def test_rf_classifier_grid_search(load_data):
    """Tests random forest with the gridsearchCV from sklearn."""

    # Get the dataset. The data generation is seeded in load_data.
    x, y = load_data(
        dataset="classification",
        n_samples=1000,
        n_features=100,
        n_classes=2,
    )

    param_grid = {
        "n_bits": [20],
        "max_depth": [15],
        "n_estimators": [5, 10, 50, 100],
    }

    grid_scorer = make_scorer(matthews_corrcoef, greater_is_better=True)

    concrete_clf = RandomForestClassifier()
    _ = GridSearchCV(
        concrete_clf, param_grid, cv=5, scoring=grid_scorer, error_score="raise", n_jobs=1
    ).fit(x, y)


# Regression


@pytest.mark.parametrize(
    "hyperparameters",
    [
        pytest.param({key: value}, id=f"{key}={value}")
        for key, values in PARAMS_RF.items()
        for value in values  # type: ignore
    ],
)
def test_rf_regressor_hyperparameters(hyperparameters, load_data, check_r2_score):
    """Test that the hyperparameters are valid."""
    x, y = load_data(
        dataset="regression",
        n_samples=100,
        n_features=10,
        n_informative=5,
    )
    model = RandomForestRegressor(
        **hyperparameters, n_bits=20, n_jobs=1, random_state=numpy.random.randint(0, 2**15)
    )
    model, sklearn_model = model.fit_benchmark(x, y)
    # Check accuracy and r2 score between the two models predictions
    check_r2_score(model.predict(x), sklearn_model.predict(x))


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
        pytest.param(4, 20, True, id="max_depth_4_n_estimators_20"),
    ],
)
@pytest.mark.parametrize(
    "n_bits",
    [
        pytest.param(6, id="n_bits_6"),
        pytest.param(2, id="n_bits_2"),
    ],
)
def test_rf_regressor(
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
    """Tests the random forest."""

    if not is_weekly_option:
        if skip_if_not_weekly:
            # Skip long tests
            return

    if not use_virtual_lib and is_vl_only_option:
        print("Warning, skipping non VL tests")
        return

    # Get the dataset
    x, y = load_data(**parameters)

    model = RandomForestRegressor(
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


def test_rf_regressor_grid_search(load_data):
    """Tests random forest with the gridsearchCV from sklearn."""
    x, y = load_data(
        dataset="regression",
        n_samples=1000,
        n_features=100,
    )

    param_grid = {
        "n_bits": [20],
        "max_depth": [15],
        "n_estimators": [5, 10, 50, 100],
    }

    grid_scorer = make_scorer(mean_squared_error, greater_is_better=True)

    concrete_clf = RandomForestRegressor()
    _ = GridSearchCV(
        concrete_clf, param_grid, cv=5, scoring=grid_scorer, error_score="raise", n_jobs=1
    ).fit(x, y)
