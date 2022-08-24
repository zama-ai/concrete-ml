"""Tests common to all sklearn models."""
import warnings
from functools import partial

import numpy
import pytest
from concrete.numpy import MAXIMUM_BIT_WIDTH
from shared import classifiers, regressors
from sklearn.exceptions import ConvergenceWarning
from torch import nn

from concrete.ml.sklearn import NeuralNetClassifier, NeuralNetRegressor


@pytest.mark.parametrize("model, parameters", classifiers + regressors)
def test_double_fit(model, parameters, load_data):
    """Tests that calling fit multiple times gives the same results"""
    if isinstance(model, partial):
        # Works differently for NeuralNetClassifier or NeuralNetRegressor
        if model.func in [NeuralNetClassifier, NeuralNetRegressor]:
            return

    x, y = load_data(**parameters)

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

        # First fit
        model.fit(x, y)
        y_pred_one = model.predict(x)

        # Second fit
        model.fit(x, y)
        y_pred_two = model.predict(x)

    assert numpy.array_equal(y_pred_one, y_pred_two)


def test_double_fit_qnn(load_data):
    """Tests that calling fit multiple times gives the same results"""

    x, y = load_data(
        dataset="classification",
        n_samples=1000,
        n_features=10,
        n_redundant=0,
        n_repeated=0,
        n_informative=5,
        n_classes=2,
        class_sep=2,
        random_state=42,
    )
    x = x.astype(numpy.float32)

    params = {
        "module__n_layers": 3,
        "module__n_w_bits": 2,
        "module__n_a_bits": 2,
        "module__n_accum_bits": MAXIMUM_BIT_WIDTH,
        "module__n_outputs": 2,
        "module__input_dim": 10,
        "module__activation_function": nn.ReLU,
        "max_epochs": 10,
        "verbose": 0,
    }

    model = NeuralNetClassifier(**params)

    # Some models use a bit of randomness while fitting under scikit-learn, making the
    # outputs always different after each fit. In order to avoid that problem, their random_state
    # parameter needs to be fixed each time the test is ran.
    model_params = model.get_params()
    if "random_state" in model_params:
        model_params["random_state"] = numpy.random.randint(0, 2**15)
    model.set_params(**model_params)

    # First fit
    model.fit(x, y)
    # FIXME: this fails and needs to be fixed, #918
    # y_pred_one = model.predict(x)

    # Second fit
    model.fit(x, y)
    # FIXME: this fails and needs to be fixed, #918
    # y_pred_two = model.predict(x)

    # FIXME: this fails and needs to be fixed, #918
    # assert numpy.array_equal(y_pred_one, y_pred_two)
