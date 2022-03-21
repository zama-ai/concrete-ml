"""Tests for the xgboost."""

import numpy
import pytest
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import GridSearchCV

from concrete.ml.sklearn import XGBClassifier

PARAMS_XGB = {
    "max_depth": [3, 4, 5, 10],
    "learning_rate": [1, 0.5, 0.1],
    "n_estimators": [50, 100, 1000],
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


@pytest.mark.parametrize(
    "hyperparameters",
    [
        pytest.param({key: value}, id=f"{key}={value}")
        for key, values in PARAMS_XGB.items()
        for value in values  # type: ignore
    ],
)
def test_xgb_hyperparameters(hyperparameters, check_r2_score, check_accuracy):
    """Test that the hyperparameters are valid."""
    x, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_classes=2,
        random_state=numpy.random.randint(0, 2**15),
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


@pytest.mark.parametrize(
    "load_data",
    [
        pytest.param(lambda: load_breast_cancer(return_X_y=True), id="breast_cancer"),
        pytest.param(
            lambda: make_classification(
                n_samples=1000,
                n_features=100,
                n_classes=2,
                random_state=numpy.random.randint(0, 2**15),
            ),
            id="make_classification",
        ),
    ],
)
def test_xgb_classifier(load_data, check_r2_score, check_accuracy):
    """Tests the xgboost."""

    # Get the dataset
    x, y = load_data()

    model = XGBClassifier(
        max_depth=7,
        n_estimators=100,
        n_bits=7,
        random_state=numpy.random.randint(0, 2**15),
        n_jobs=1,
    )
    model, sklearn_model = model.fit_benchmark(x, y)

    # Check accuracy and r2 score between the two models predictions
    check_accuracy(model.predict(x), sklearn_model.predict(x))
    check_r2_score(model.predict_proba(x), sklearn_model.predict_proba(x))

    # FIXME No FHE yet with XGBoost (see #436)
    # # Test compilation
    # model.compile(x, default_compilation_configuration, use_virtual_lib=use_virtual_lib)

    # # Compare FHE vs non-FHE
    # check_is_good_execution_for_quantized_models(x=x[:5], model_predict=model.predict)


def test_grid_search():
    """Tests xgboost with the gridsearchCV from sklearn."""
    x, y = make_classification(
        n_samples=1000,
        n_features=100,
        n_classes=2,
        random_state=numpy.random.randint(0, 2**15),
    )

    param_grid = {
        "n_bits": [20],
        "max_depth": [2],
        "n_estimators": [5, 10, 50, 100],
        "n_jobs": [1],
    }

    grid_scorer = make_scorer(matthews_corrcoef, greater_is_better=True)

    concrete_clf = XGBClassifier()
    _ = GridSearchCV(concrete_clf, param_grid, cv=5, scoring=grid_scorer, verbose=1, n_jobs=1).fit(
        x, y
    )
