"""Tests for the xgboost."""

import numpy
import pytest
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import GridSearchCV

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
@pytest.mark.parametrize("n_classes", [2, 4])
def test_rf_hyperparameters(hyperparameters, n_classes, check_r2_score, check_accuracy):
    """Test that the hyperparameters are valid."""
    x, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_classes=n_classes,
        random_state=numpy.random.randint(0, 2**15),
    )
    model = RandomForestClassifier(
        **hyperparameters, n_bits=20, n_jobs=1, random_state=numpy.random.randint(0, 2**15)
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
@pytest.mark.parametrize(
    "use_virtual_lib",
    [
        pytest.param(False, id="no_virtual_lib"),
        pytest.param(True, id="use_virtual_lib"),
    ],
)
@pytest.mark.parametrize(
    "max_depth, n_estimators",
    [
        pytest.param(2, 1, id="max_depth_2_n_estimators_1"),
        pytest.param(2, 5, id="max_depth_2_n_estimators_5"),
        pytest.param(2, 10, id="max_depth_2_n_estimators_10"),
        # FIXME add more trees when https://github.com/zama-ai/concrete-ml-internal/issues/572
        # is fixed.
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
    load_data,
    max_depth,
    n_estimators,
    n_bits,
    default_compilation_configuration,
    check_is_good_execution_for_quantized_models,
    use_virtual_lib,
):
    """Tests the random forest."""

    # Get the dataset
    x, y = load_data()

    model = RandomForestClassifier(
        max_depth=max_depth,
        n_estimators=n_estimators,
        n_bits=n_bits,
        random_state=numpy.random.randint(0, 2**15),
        n_jobs=1,
    )
    model.fit(x, y)

    # Test compilation
    model.compile(x, default_compilation_configuration, use_virtual_lib=use_virtual_lib)

    # Compare FHE vs non-FHE
    check_is_good_execution_for_quantized_models(x=x[:5], model_predict=model.predict)


def test_grid_search():
    """Tests random forest with the gridsearchCV from sklearn."""
    x, y = make_classification(
        n_samples=1000,
        n_features=100,
        n_classes=2,
        random_state=numpy.random.randint(0, 2**15),
    )

    param_grid = {
        "n_bits": [20],
        "max_depth": [15],
        "n_estimators": [5, 10, 50, 100],
    }

    grid_scorer = make_scorer(matthews_corrcoef, greater_is_better=True)

    concrete_clf = RandomForestClassifier()
    _ = GridSearchCV(concrete_clf, param_grid, cv=5, scoring=grid_scorer, verbose=1, n_jobs=1).fit(
        x, y
    )
