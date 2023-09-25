"""Tests with Pandas."""
import sys

import numpy
import pandas
import pytest

from concrete.ml.sklearn import (
    _get_sklearn_linear_models,
    _get_sklearn_neighbors_models,
    _get_sklearn_tree_models,
)


# A type error will be raised for NeuralNetworks, which is tested in test_failure_bad_data_types
@pytest.mark.parametrize(
    "model_class",
    _get_sklearn_linear_models() + _get_sklearn_tree_models() + _get_sklearn_neighbors_models(),
)
@pytest.mark.parametrize(
    "bad_value, expected_error",
    [
        (numpy.nan, "Input X contains NaN."),
        (None, "Input X contains NaN."),
        ("this", "could not convert string to float: 'this'"),
    ],
)
def test_failure_bad_param(model_class, bad_value, expected_error):
    """Check our checks see if ever the Pandas data-set is not correct."""

    dic = {
        "Col One": [1, 2, bad_value, 3],
        "Col Two": [4, 5, 6, bad_value],
        "Col Three": [bad_value, 7, 8, 9],
    }

    # Creating a dataframe using dictionary
    x_train = pandas.DataFrame(dic)
    y_train = x_train["Col Three"]

    model = model_class(n_bits=2)

    # The error message changed in one of our dependencies
    assert sys.version_info.major == 3
    if sys.version_info.minor <= 7:
        if expected_error == "Input X contains NaN.":
            expected_error = "Input contains NaN*"

    with pytest.raises(ValueError, match=expected_error):
        model.fit(x_train, y_train)
