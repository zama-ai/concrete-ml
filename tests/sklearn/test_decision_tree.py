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
            lambda: make_classification(n_samples=100, n_features=10, n_classes=2),
            id="make_classification",
        ),
    ],
)
def test_decision_tree_classifier(
    load_data, default_compilation_configuration, check_is_good_execution_for_quantized_models
):
    """Tests the sklearn DecisionTreeClassifier."""
    # Get the sklearn model
    x, y = load_data()

    model = DecisionTreeClassifier(n_bits=6, max_depth=7)
    model, sklearn_model = model.fit_benchmark(x, y)

    # Check correlation coefficient between the two models
    assert (
        numpy.corrcoef(model.predict(x), sklearn_model.predict(x))[0, 1] > 0.99
    ), "The correlation coefficient between the two models is too low."

    # Check that the predictions are correct using tensors
    y_pred_tensors = model._predict_with_tensors(x)  # pylint: disable=protected-access
    y_pred = model.predict(x)

    assert numpy.array_equal(y_pred_tensors, y_pred)

    # Test compilation
    model.compile(x, default_compilation_configuration)

    # Compare FHE vs non-FHE
    check_is_good_execution_for_quantized_models(x=x[:5], model_predict=model.predict)
