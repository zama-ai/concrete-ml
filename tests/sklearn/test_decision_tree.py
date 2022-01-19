"""Tests for the sklearn decision trees."""
import numpy
import pytest
from sklearn.datasets import load_breast_cancer

from concrete.ml.sklearn import DecisionTreeClassifier


@pytest.mark.parametrize(
    "load_data",
    [load_breast_cancer],
)
def test_decision_tree_classifier(load_data, default_compilation_configuration):
    """Tests the sklearn DecisionTreeClassifier."""
    # Get the sklearn model
    x, y = load_data(return_X_y=True)

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

    # Predict in FHE for the first 5 examples
    y_pred_fhe = model.predict(x[:5], use_fhe=True)

    # Check that the predictions are correct using FHE
    assert numpy.array_equal(y_pred_fhe, y_pred[:5])
