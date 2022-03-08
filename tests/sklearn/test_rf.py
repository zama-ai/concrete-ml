"""Tests for the xgboost."""

import numpy
import pytest
from sklearn.datasets import load_breast_cancer, make_classification

from concrete.ml.sklearn import RandomForestClassifier

PARAMS_RF = {
    "max_depth": [15, 20, 30],
    "n_estimators": [50, 100, 200],
}


@pytest.mark.parametrize(
    "hyperparameters",
    [
        pytest.param({key: value}, id=f"{key}={value}")
        for key, values in PARAMS_RF.items()
        for value in values  # type: ignore
    ],
)
def test_rf_hyperparameters(hyperparameters, check_r2_score):
    """Test that the hyperparameters are valid."""
    x, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_classes=2,
        random_state=numpy.random.randint(0, 2**15),
    )
    model = RandomForestClassifier(
        **hyperparameters, n_bits=20, n_jobs=1, random_state=numpy.random.randint(0, 2**15)
    )
    model, sklearn_model = model.fit_benchmark(x, y)
    # Make sure that model.predict is the same as sklearn_model.predict
    check_r2_score(model.predict(x), sklearn_model.predict(x))
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
def test_rf_classifier(load_data, check_r2_score):
    """Tests the random forest."""

    # Get the dataset
    x, y = load_data()

    model = RandomForestClassifier(
        max_depth=15,
        n_estimators=100,
        n_bits=7,
        random_state=numpy.random.randint(0, 2**15),
        n_jobs=1,
    )
    model, sklearn_model = model.fit_benchmark(x, y)

    # Check that the two models are similar.
    check_r2_score(model.predict_proba(x), sklearn_model.predict_proba(x))

    # FIXME No FHE yet with RandomForest (see #436)
    # # Test compilation
    # model.compile(x, default_compilation_configuration, use_virtual_lib=use_virtual_lib)

    # # Compare FHE vs non-FHE
    # check_is_good_execution_for_quantized_models(x=x[:5], model_predict=model.predict)
