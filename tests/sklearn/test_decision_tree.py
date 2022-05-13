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
        pytest.param(
            lambda: make_classification(
                n_samples=100,
                n_features=10,
                n_classes=4,
                n_informative=10,
                n_redundant=0,
                random_state=numpy.random.randint(0, 2**15),
            ),
            id="make_classification_multiclass",
        ),
    ],
)
@pytest.mark.parametrize("use_virtual_lib", [True, False])
def test_decision_tree_classifier(
    load_data,
    default_configuration,
    check_is_good_execution_for_quantized_models,
    use_virtual_lib,
    is_vl_only_option,
):
    """Tests the sklearn DecisionTreeClassifier."""
    if not use_virtual_lib and is_vl_only_option:
        print("Warning, skipping non VL tests")
        return

    # Get the dataset
    x, y = load_data()

    model = DecisionTreeClassifier(
        n_bits=6, max_depth=7, random_state=numpy.random.randint(0, 2**15)
    )
    model.fit(x, y)

    # Test compilation
    model.compile(x, default_configuration, use_virtual_lib=use_virtual_lib)

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
@pytest.mark.parametrize("n_classes,", [2, 4])
@pytest.mark.parametrize("offset", [0, 1, 2])
def test_decision_tree_hyperparameters(
    hyperparameters, n_classes, offset, check_accuracy, check_r2_score
):
    """Test that the hyperparameters are valid."""
    x, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_classes=n_classes,
        random_state=numpy.random.randint(0, 2**15),
    )
    y += offset
    model = DecisionTreeClassifier(
        **hyperparameters, n_bits=24, random_state=numpy.random.randint(0, 2**15)
    )
    model, sklearn_model = model.fit_benchmark(x, y)

    # Make sure that model.predict is the same as sklearn_model.predict
    check_accuracy(model.predict(x), sklearn_model.predict(x))
    check_r2_score(model.predict_proba(x), sklearn_model.predict_proba(x))
