"""Tests for the FHE sklearn compatible NNs."""
from copy import deepcopy

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from concrete.ml.sklearn.qnn import NeuralNetClassifier


@pytest.mark.parametrize(
    "n_layers",
    [3, 5],
)
@pytest.mark.parametrize("n_bits_w_a", [16])
@pytest.mark.parametrize("n_accum_bits", [32])
@pytest.mark.parametrize(
    "activation_function",
    [
        pytest.param(nn.ReLU),
        pytest.param(nn.ReLU6),
        pytest.param(nn.Sigmoid),
        pytest.param(nn.SELU),
    ],
)
@pytest.mark.parametrize("n_classes", [2])
@pytest.mark.parametrize("input_dim", [10, 100])
def test_nn_classifiers_quant(
    n_layers, n_bits_w_a, n_accum_bits, activation_function, n_classes, input_dim, check_r2_score
):
    """Test the correctness of the results of quantized NN classifiers through the sklearn
    wrapper."""

    x, y = make_classification(
        1000,
        n_features=input_dim,
        n_redundant=0,
        n_repeated=0,
        n_informative=input_dim,
        n_classes=n_classes,
        class_sep=2,
        random_state=42,
    )
    x = x.astype(np.float32)

    # Perform a classic test-train split (deterministic by fixing the seed)
    x_train, x_test, y_train, _ = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
    )

    params = {
        "module__n_layers": n_layers,
        "module__n_w_bits": n_bits_w_a,
        "module__n_a_bits": n_bits_w_a,
        "module__n_accum_bits": n_accum_bits,
        "module__n_classes": n_classes,
        "module__input_dim": x_train.shape[1],
        "module__activation_function": activation_function,
        "max_epochs": 10,
        "verbose": 0,
    }

    concrete_classifier = NeuralNetClassifier(**params)

    # Compute mean/stdev on training set and normalize both train and test sets with them
    normalizer = StandardScaler()
    x_train = normalizer.fit_transform(x_train)
    x_test = normalizer.transform(x_test)

    _, sklearn_classifier = concrete_classifier.fit_benchmark(x_train, y_train)
    y_pred_sk = sklearn_classifier.predict(x_test)
    y_pred = concrete_classifier.predict(x_test)

    check_r2_score(y_pred_sk, y_pred)


def test_parameter_validation():
    """Test that the sklearn quantized NN wrappers validate their parameters"""

    valid_params = {
        "module__n_layers": 3,
        "module__n_w_bits": 2,
        "module__n_a_bits": 2,
        "module__n_accum_bits": 7,
        "module__n_classes": 2,
        "module__input_dim": 10,
        "module__activation_function": nn.ReLU,
        "max_epochs": 10,
        "verbose": 0,
    }

    x, y = make_classification(
        1000,
        n_features=10,
        n_redundant=0,
        n_repeated=0,
        n_informative=10,
        n_classes=2,
        class_sep=2,
        random_state=42,
    )
    x = x.astype(np.float32)

    invalid_params_and_exception_pattern = {
        ("module__n_layers", 0, ".* number of layers.*"),
        ("module__n_w_bits", 0, ".* quantization bitwidth.*"),
        ("module__n_a_bits", 0, ".* quantization bitwidth.*"),
        ("module__n_accum_bits", 0, ".* accumulator bitwidth.*"),
        ("module__n_classes", 1, ".* number of classes.*"),
        ("module__input_dim", 0, ".* number of input dimensions.*"),
    }
    for inv_param in invalid_params_and_exception_pattern:
        params = deepcopy(valid_params)
        params[inv_param[0]] = inv_param[1]

        with pytest.raises(
            ValueError,
            match=inv_param[2],
        ):
            concrete_classifier = NeuralNetClassifier(**params)
            concrete_classifier.fit(x, y)


def test_pipeline_and_cv():
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
    x = x.astype(np.float32)

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
        "module__n_accum_bits": 7,
        "module__n_classes": 2,
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

    # Call .fit twice, this can occur and is valid usage
    # Sometimes it can be done by the user to warm-start, or just to reuse the obj
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


# FIXME: once the HNP frontend compilation speed is improved, test with several activation funcs.
# see: https://github.com/zama-ai/concrete-numpy-internal/issues/1374
def test_compile_and_calib(default_compilation_configuration):
    """Test whether the sklearn quantized NN wrappers compile to FHE and execute well on encrypted
    inputs"""

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
    x = x.astype(np.float32)

    # Perform a classic test-train split (deterministic by fixing the seed)
    x_train, x_test, y_train, _ = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
    )

    # Compute mean/stdev on training set and normalize both train and test sets with them
    # Optimization algorithms for Neural networks work well on 0-centered inputs
    normalizer = StandardScaler()
    x_train = normalizer.fit_transform(x_train)
    x_test = normalizer.transform(x_test)

    # Configure a minimal neural network and train it quickly
    params = {
        "module__n_layers": 1,
        "module__n_w_bits": 2,
        "module__n_a_bits": 2,
        "module__n_accum_bits": 5,
        "module__n_classes": 2,
        "module__input_dim": n_features,
        "module__activation_function": nn.Sigmoid,
        "max_epochs": 10,
        "verbose": 0,
    }

    clf = NeuralNetClassifier(**params)

    # Compiling a model that is not trained should fail
    with pytest.raises(ValueError, match=".* needs to be calibrated .*"):
        clf.compile(x_train, compilation_configuration=default_compilation_configuration)

    # Predicting in FHE with a model that is not trained and calibrated should fail
    with pytest.raises(ValueError, match=".* needs to be calibrated .*"):
        x_test_q = np.zeros((1, n_features), dtype=np.uint8)
        clf.predict(x_test_q, execute_in_fhe=True)

    # Train the model
    clf.fit(x_train, y_train)

    # Predicting with a model that is not compiled should fail
    with pytest.raises(ValueError, match=".* not yet compiled .*"):
        x_test_q = np.zeros((1, n_features), dtype=np.uint8)
        clf.predict(x_test_q, execute_in_fhe=True)

    # Compile the model
    clf.compile(x_train, compilation_configuration=default_compilation_configuration)

    # Execute in FHE, but don't check the value.
    # Since FHE execution introduces some stochastic errors,
    # accuracy of FHE compiled classifiers is measured in the benchmarks
    clf.predict(x_test[0, :], execute_in_fhe=True)
