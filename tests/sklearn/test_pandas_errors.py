"""Tests with Pandas."""
from functools import partial

import numpy
import pandas
import pytest

from concrete.ml.pytest.utils import classifier_models, regressor_models
from concrete.ml.sklearn.base import get_sklearn_neural_net_models


@pytest.mark.parametrize("model", classifier_models + regressor_models)
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
    if isinstance(model, partial):
        # Works differently for neural nets
        if model.func in get_sklearn_neural_net_models():
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
