"""Tests for the sklearn decision trees."""
import numpy
import pytest

from concrete.ml.sklearn.base import get_sklearn_tree_models


@pytest.mark.parametrize("model", get_sklearn_tree_models(regressor=False))
def test_one_class_edge_case(model):
    """Test the assertion for one class in y."""

    model = model()
    x = numpy.random.randint(0, 64, size=(100, 10))
    y = numpy.random.randint(0, 1, size=(100))

    assert len(numpy.unique(y)) == 1, "Wrong numpy randint generation for y."

    with pytest.raises(AssertionError, match="You must provide at least 2 classes in y."):
        model.fit(x, y)
