"""Tests pipelines in our models."""
import warnings
from functools import partial

import numpy
import pytest
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from concrete.ml.common.utils import MAX_BITWIDTH_BACKWARD_COMPATIBLE
from concrete.ml.pytest.utils import classifiers, regressors
from concrete.ml.sklearn.base import get_sklearn_neural_net_models


@pytest.mark.parametrize("model, parameters", classifiers + regressors)
def test_pipeline_classifiers_regressors(model, parameters, load_data):
    """Tests that classifiers and regressors work well within sklearn pipelines."""
    if isinstance(model, partial):
        # Tested in test_pipeline_and_cv_qnn
        if model.func in get_sklearn_neural_net_models():
            return

    x, y = load_data(**parameters)

    pipe_cv = Pipeline(
        [
            ("pca", PCA(n_components=2, random_state=numpy.random.randint(0, 2**15))),
            ("scaler", StandardScaler()),
            ("model", model()),
        ]
    )
    # Do a grid search to find the best hyperparameters
    param_grid = {
        "model__n_bits": [2, 3],
    }
    grid_search = GridSearchCV(pipe_cv, param_grid, error_score="raise", cv=3)

    # Sometimes, we miss convergence, which is not a problem for our test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        grid_search.fit(x, y)


# Get the dataset. The data generation is seeded in load_data.
qnn = [
    (
        {
            "dataset": "classification",
            "n_samples": 1000,
            "n_features": 10,
            "n_redundant": 0,
            "n_repeated": 0,
            "n_informative": 5,
            "n_classes": 2,
            "class_sep": 2,
        }
    )
]


@pytest.mark.parametrize("model", get_sklearn_neural_net_models(regressor=False))
@pytest.mark.parametrize("parameters", qnn)
def test_pipeline_and_cv_qnn(model, parameters, load_data):
    """Test whether we can use the quantized NN sklearn wrappers in pipelines and in
    cross-validation"""

    x, y = load_data(**parameters)

    x = x.astype(numpy.float32)

    # Perform a classic test-train split (deterministic by fixing the seed)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=numpy.random.randint(0, 2**15),
    )

    # PCA will reduce dataset dimensionality to this number of features
    # When the number is too low, separability could be impacted

    n_dims = 5
    params = {
        "module__n_layers": 3,
        "module__n_w_bits": 5,
        "module__n_a_bits": 5,
        "module__n_accum_bits": 4 * MAX_BITWIDTH_BACKWARD_COMPATIBLE,
        "module__n_outputs": 2,
        "module__input_dim": n_dims,
        "module__activation_function": nn.ReLU,
        "max_epochs": 20,
        "verbose": 0,
    }

    pipe = Pipeline(
        [
            ("pca", PCA(n_components=n_dims, random_state=numpy.random.randint(0, 2**15))),
            ("scaler", StandardScaler()),
            ("net", model(**params)),
        ]
    )

    pipe.fit(x_train, y_train)
    pipe.score(x_test, y_test)

    pipe_cv = Pipeline(
        [
            ("pca", PCA(n_components=n_dims, random_state=numpy.random.randint(0, 2**15))),
            ("scaler", StandardScaler()),
            ("net", model(**params)),
        ]
    )

    clf = GridSearchCV(
        pipe_cv,
        {"net__module__n_layers": (3, 5), "net__module__activation_function": (nn.Tanh, nn.ReLU6)},
        error_score="raise",
    )
    clf.fit(x_train, y_train)
