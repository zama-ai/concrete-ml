"""Tests for the sklearn decision trees."""
import numpy
import pytest
from sklearn.datasets import load_breast_cancer, make_classification

from concrete.ml.sklearn import DecisionTreeClassifier


@pytest.mark.parametrize(
    "load_data",
    [
        pytest.param(lambda: load_breast_cancer(return_X_y=True), id="breast_cancer"),
        pytest.param(
            lambda: make_classification(
                n_samples=100,
                n_features=10,
                n_classes=2,
                random_state=numpy.random.randint(0, 2**15),
            ),
            id="make_classification",
        ),
    ],
)
@pytest.mark.parametrize("use_virtual_lib", [True, False])
def test_decision_tree_classifier(
    load_data,
    default_compilation_configuration,
    check_is_good_execution_for_quantized_models,
    use_virtual_lib,
    check_r2_score,
    check_accuracy,
):
    """Tests the sklearn DecisionTreeClassifier."""

    # Get the dataset
    x, y = load_data()

    model = DecisionTreeClassifier(
        n_bits=6, max_depth=7, random_state=numpy.random.randint(0, 2**15)
    )
    model, sklearn_model = model.fit_benchmark(x, y)

    # Check accuracy and r2 score between the two models predictions
    check_accuracy(model.predict(x), sklearn_model.predict(x))
    check_r2_score(model.predict_proba(x), sklearn_model.predict_proba(x))

    # Test compilation
    model.compile(x, default_compilation_configuration, use_virtual_lib=use_virtual_lib)

    # Compare FHE vs non-FHE
    check_is_good_execution_for_quantized_models(x=x[:5], model_predict=model.predict)


PARAMS_TREE = {
    "max_depth": [3, 4, 5, 10],
    "min_samples_split": [2, 3, 4, 5],
    "min_samples_leaf": [1, 2, 3, 4],
    "min_weight_fraction_leaf": [0.0, 0.1, 0.2, 0.3],
    "max_features": ["auto", "sqrt", "log2"],
    "max_leaf_nodes": [None, 5, 10, 20],
}


@pytest.mark.parametrize(
    "hyperparameters",
    [
        pytest.param({key: value}, id=f"{key}={value}")
        for key, values in PARAMS_TREE.items()
        for value in values  # type: ignore
    ],
)
def test_decision_tree_hyperparameters(hyperparameters, check_r2_score, check_accuracy):
    """Test that the hyperparameters are valid."""
    x, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_classes=2,
        random_state=numpy.random.randint(0, 2**15),
    )
    model = DecisionTreeClassifier(
        **hyperparameters, n_bits=20, random_state=numpy.random.randint(0, 2**15)
    )
    model, sklearn_model = model.fit_benchmark(x, y)
    # Make sure that model.predict is the same as sklearn_model.predict
    check_accuracy(model.predict(x), sklearn_model.predict(x))
    check_r2_score(model.predict_proba(x), sklearn_model.predict_proba(x))
