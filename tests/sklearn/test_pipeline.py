"""Tests pipelines in our models."""
import warnings

import numpy
import pytest
from concrete.numpy import MAXIMUM_BIT_WIDTH
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression as sklearn_make_regression
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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


def make_regression(positive_targets, *args, **kwargs):
    """Generate a random regression problem.

    Sklearn's make_regression() method generates a random regression problem without any domain
    restrictions. However, some models can only handle non negative or (stricly) positive target
    values. This function therefore adapts it in order to make it work for any tested regressors.
    """
    generated_regression = list(sklearn_make_regression(*args, **kwargs))

    # Some regressors can only handle positive target values, often stricly positive.
    if positive_targets:
        generated_regression[1] = numpy.abs(generated_regression[1]) + 1

    return tuple(generated_regression)


list_classifiers = [
    (
        alg,
        make_classification(
            n_samples=1000,
            n_features=100,
            n_classes=4,
            n_informative=100,
            n_redundant=0,
            random_state=numpy.random.randint(0, 2**15),
        ),
    )
    for alg in [
        DecisionTreeClassifier,
        RandomForestClassifier,
        XGBClassifier,
        LinearSVC,
        LinearSVC,
        LogisticRegression,
    ]
]

# Only LinearRegression supports multi targets
# GammaRegressor, PoissonRegressor and TweedieRegressor only handle positive target values
list_regressors = [
    (
        alg,
        make_regression(
            positive_targets=alg in [GammaRegressor, PoissonRegressor, TweedieRegressor],
            n_samples=200,
            n_features=10,
            n_informative=10,
            n_targets=2 if alg == LinearRegression else 1,
            noise=0,
            random_state=numpy.random.randint(0, 2**15),
        ),
    )
    for alg in [
        GammaRegressor,
        LinearRegression,
        LinearSVR,
        PoissonRegressor,
        TweedieRegressor,
    ]
]


@pytest.mark.parametrize("alg, data", list_classifiers + list_regressors)
def test_pipeline_classifiers_regressors(alg, data):
    """Tests that classifiers and regressors work well within sklearn pipelines."""
    x, y = data

    pipe_cv = Pipeline(
        [
            ("pca", PCA(n_components=2)),
            ("scaler", StandardScaler()),
            ("alg", alg()),
        ]
    )
    # Do a grid search to find the best hyperparameters
    param_grid = {
        "alg__n_bits": [2, 3],
    }
    grid_search = GridSearchCV(pipe_cv, param_grid, cv=3)

    # Sometimes, we miss convergence, which is not a problem for our test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        grid_search.fit(x, y)


def test_pipeline_and_cv_qnn():
    """Test whether we can use the quantized NN sklearn wrappers in pipelines and in
    cross-validation"""

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

    # Perform a classic test-train split (deterministic by fixing the seed)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
    )
    params = {
        "module__n_layers": 3,
        "module__n_w_bits": 2,
        "module__n_a_bits": 2,
        "module__n_accum_bits": MAXIMUM_BIT_WIDTH,
        "module__n_outputs": 2,
        "module__input_dim": 2,
        "module__activation_function": nn.SELU,
        "max_epochs": 10,
        "verbose": 0,
    }

    pipe = Pipeline(
        [
            ("pca", PCA(n_components=2)),
            ("scaler", StandardScaler()),
            ("net", NeuralNetClassifier(**params)),
        ]
    )

    pipe.fit(x_train, y_train)
    pipe.score(x_test, y_test)

    pipe_cv = Pipeline(
        [
            ("pca", PCA(n_components=2)),
            ("scaler", StandardScaler()),
            ("net", NeuralNetClassifier(**params)),
        ]
    )

    clf = GridSearchCV(
        pipe_cv,
        {"net__module__n_layers": (3, 5), "net__module__activation_function": (nn.Tanh, nn.ReLU6)},
    )
    clf.fit(x_train, y_train)
