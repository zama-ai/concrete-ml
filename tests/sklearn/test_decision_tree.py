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
):
    """Tests the sklearn DecisionTreeClassifier."""

    # Get the dataset
    x, y = load_data()

    model = DecisionTreeClassifier(
        n_bits=6, max_depth=7, random_state=numpy.random.randint(0, 2**15)
    )
    model, sklearn_model = model.fit_benchmark(x, y)

    # Check correlation coefficient between the two models
    check_r2_score(model.predict(x), sklearn_model.predict(x))

    # Test compilation
    model.compile(x, default_compilation_configuration, use_virtual_lib=use_virtual_lib)

    # Compare FHE vs non-FHE
    check_is_good_execution_for_quantized_models(x=x[:5], model_predict=model.predict)
