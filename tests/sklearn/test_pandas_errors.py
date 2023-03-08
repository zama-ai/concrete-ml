"""Tests with Pandas."""
import sys

import numpy
import pandas
import pytest

from concrete.ml.common.utils import is_model_class_in_a_list
from concrete.ml.pytest.utils import sklearn_models_and_datasets
from concrete.ml.sklearn import get_sklearn_neural_net_models


@pytest.mark.parametrize("model_class", [m[0][0] for m in sklearn_models_and_datasets])
@pytest.mark.parametrize(
    "bad_value, expected_error",
    [
        (numpy.nan, "Input X contains NaN."),
        (None, "Input X contains NaN."),
        ("this", "could not convert string to float: 'this'"),
    ],
)
def test_failure_bad_param(model_class, bad_value, expected_error):
    """Check our checks see if ever the Panda dataset is not correct."""

    # For NeuralNetworks, a type error will be raised, which is tested
    # in test_failure_bad_data_types
    if is_model_class_in_a_list(model_class, get_sklearn_neural_net_models()):
        return

    dic = {
        "Col One": [1, 2, bad_value, 3],
        "Col Two": [4, 5, 6, bad_value],
        "Col Three": [bad_value, 7, 8, 9],
    }

    # Creating a dataframe using dictionary
    x_train = pandas.DataFrame(dic)
    y_train = x_train["Col Three"]

    model = model_class(n_bits=2)

    # The error message changed in one of our dependancies
    assert sys.version_info.major == 3
    if sys.version_info.minor <= 7:
        if expected_error == "Input X contains NaN.":
            expected_error = "Input contains NaN*"

    with pytest.raises(ValueError, match=expected_error):
        model.fit(x_train, y_train)
