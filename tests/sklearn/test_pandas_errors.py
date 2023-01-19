"""Tests with Pandas."""
import numpy
import pandas
import pytest

from concrete.ml.common.utils import is_model_class_in_a_list
from concrete.ml.pytest.utils import classifiers, regressors
from concrete.ml.sklearn.base import get_sklearn_neural_net_models


@pytest.mark.parametrize("model", [m[0][0] for m in classifiers + regressors])
@pytest.mark.parametrize(
    "bad_value, expected_error",
    [
        (numpy.nan, "Input contains NaN*"),
        (None, "Input contains NaN*"),
        ("this", "could not convert string to float: 'this'"),
    ],
)
def test_failure_bad_param(model, bad_value, expected_error):
    """Check our checks see if ever the Panda dataset is not correct."""

    # Works differently for neural nets
    if is_model_class_in_a_list(model, get_sklearn_neural_net_models()):
        return

    dic = {
        "Col One": [1, 2, bad_value, 3],
        "Col Two": [4, 5, 6, bad_value],
        "Col Three": [bad_value, 7, 8, 9],
    }

    # Creating a dataframe using dictionary
    x_train = pandas.DataFrame(dic)
    y_train = x_train["Col Three"]

    model = model(n_bits=2)

    with pytest.raises(ValueError, match=expected_error):
        model.fit(x_train, y_train)
