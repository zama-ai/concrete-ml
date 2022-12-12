"""Tests with Pandas."""
import warnings
from functools import partial

import numpy
import pandas
import pytest
from sklearn.exceptions import ConvergenceWarning

from concrete.ml.pytest.utils import classifier_models, classifiers, regressor_models, regressors
from concrete.ml.sklearn.base import get_sklearn_neural_net_models


@pytest.mark.parametrize("model, parameters", classifiers + regressors)
def test_pandas(model, parameters, load_data):
    """Tests that we can use Pandas for inputs to fit"""
    model_name = model.__name__ if not isinstance(model, partial) else model.func.__name__
    x, y = load_data(**parameters, model_name=model_name)

    # Turn to Pandas
    x = pandas.DataFrame(x)

    if y.ndim == 1:
        y = pandas.Series(y)

    model = model(n_bits=2)

    # Some models use a bit of randomness while fitting under scikit-learn, making the
    # outputs always different after each fit. In order to avoid that problem, their random_state
    # parameter needs to be fixed each time the test is ran.
    model_params = model.get_params()
    if "random_state" in model_params:
        model_params["random_state"] = numpy.random.randint(0, 2**15)
    model.set_params(**model_params)

    # Sometimes, we miss convergence, which is not a problem for our test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)

        if "NeuralNet" in model_name:
            # Pandas data frames are not properly handle yet
            # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2327
            x = x.astype(numpy.float32)
            model.fit(x, y)
            model.predict(x.to_numpy())
        else:
            model.fit(x, y)
            model.predict(x)


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
