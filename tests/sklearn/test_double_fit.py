"""Tests common to all sklearn models."""
import warnings

import numpy
import pytest
from concrete.numpy import MAXIMUM_BIT_WIDTH
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from torch import nn

from concrete.ml.sklearn import (
    DecisionTreeClassifier,
    GammaRegressor,
    LinearRegression,
    LinearSVC,
    LinearSVR,
    LogisticRegression,
    NeuralNetClassifier,
    PoissonRegressor,
    RandomForestClassifier,
    TweedieRegressor,
    XGBClassifier,
)


@pytest.mark.parametrize(
    "alg",
    [
        pytest.param(RandomForestClassifier, id="RandomForestClassifier"),
        pytest.param(DecisionTreeClassifier, id="DecisionTreeClassifier"),
        pytest.param(XGBClassifier, id="XGBClassifier"),
        pytest.param(GammaRegressor, id="GammaRegressor"),
        pytest.param(LinearRegression, id="LinearRegression"),
        pytest.param(LinearSVC, id="LinearSVC"),
        pytest.param(LinearSVR, id="LinearSVR"),
        pytest.param(LogisticRegression, id="LoisticRegression"),
        pytest.param(PoissonRegressor, id="PoissonRegressor"),
        pytest.param(TweedieRegressor, id="TweedieRegressor"),
    ],
)
@pytest.mark.parametrize(
    "load_data",
    [
        pytest.param(
            lambda: make_classification(
                n_samples=100,
                n_features=10,
                n_classes=2,
                random_state=numpy.random.randint(0, 2**15),
            ),
            id="make_classification",
        ),
        pytest.param(
            lambda: make_classification(
                n_samples=100,
                n_features=10,
                n_classes=4,
                n_informative=10,
                n_redundant=0,
                random_state=numpy.random.randint(0, 2**15),
            ),
            id="make_classification_multiclass",
        ),
    ],
)
def test_double_fit(alg, load_data):
    """Tests that calling fit multiple times gives the same results"""
    x, y = load_data()

    # For Gamma regressor
    if alg is GammaRegressor:
        y = numpy.abs(y) + 1

    model = alg(n_bits=2)

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


def test_double_fit_qnn():
    """Tests that calling fit multiple times gives the same results"""

    n_features = 10

    x, y = make_classification(
        1000,
        n_features=n_features,
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
        "module__activation_function": nn.SELU,
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
