"""Tests pipelines in our models."""
import warnings

import numpy
import pytest
from concrete.numpy import MAXIMUM_BIT_WIDTH
from sklearn.datasets import make_classification
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

list_classifiers = [
    (
        alg,
        lambda: make_classification(
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
    ]
]

list_regressors = [
    (
        alg,
        # FIXME: make it work
        # lambda: make_regression(
        #     n_samples=200,
        #     n_features=10,
        #     n_informative=10,
        #     n_targets=2,
        #     noise=0,
        #     random_state=numpy.random.randint(0, 2**15),
        # ),
        lambda: make_classification(
            n_samples=1000,
            n_features=100,
            n_classes=4,
            n_informative=100,
            n_redundant=0,
            random_state=numpy.random.randint(0, 2**15),
        ),
    )
    for alg in [
        GammaRegressor,
        LinearRegression,
        LinearSVR,
        LogisticRegression,
        PoissonRegressor,
        TweedieRegressor,
    ]
]


@pytest.mark.parametrize("alg, load_data", list_classifiers + list_regressors)
def test_pipeline_classifier(alg, load_data):
    """Tests that the classifier work well within sklearn pipelines."""
    x, y = load_data()

    # For Gamma regressor
    if alg is GammaRegressor:
        y = numpy.abs(y) + 1

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
