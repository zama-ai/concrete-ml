"""Tests for the FHE sklearn compatible NNs."""
from copy import deepcopy

import numpy
import pytest
from concrete.numpy import MAXIMUM_BIT_WIDTH
from sklearn.datasets import make_classification, make_regression
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skorch.classifier import NeuralNetClassifier as SKNeuralNetClassifier
from torch import nn

from concrete.ml.sklearn.qnn import (
    NeuralNetClassifier,
    NeuralNetRegressor,
    QuantizedSkorchEstimatorMixin,
)


@pytest.mark.parametrize(
    "n_layers",
    [3],
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
@pytest.mark.parametrize("n_outputs", [1, 5])
@pytest.mark.parametrize("input_dim", [100])
@pytest.mark.parametrize("model", [NeuralNetClassifier, NeuralNetRegressor])
def test_nn_models_quant(
    n_layers,
    n_bits_w_a,
    n_accum_bits,
    activation_function,
    n_outputs,
    input_dim,
    model,
    check_r2_score,
    check_accuracy,
):
    """Test the correctness of the results of quantized NN classifiers through the sklearn
    wrapper."""

    if model is NeuralNetClassifier:
        x, y = make_classification(
            1000,
            n_features=input_dim,
            n_redundant=0,
            n_repeated=0,
            n_informative=input_dim,
            n_classes=n_outputs,
            class_sep=2,
            random_state=42,
        )
    elif model is NeuralNetRegressor:
        x, y, _ = make_regression(
            1000,
            n_features=input_dim,
            n_informative=input_dim,
            n_targets=n_outputs,
            noise=2,
            random_state=42,
            coef=True,
        )
        if y.ndim == 1:
            y = numpy.expand_dims(y, 1)
        y = y.astype(numpy.float32)
    else:
        raise ValueError(f"Data generator not implemented for {str(model)}")

    x = x.astype(numpy.float32)

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
        "module__n_outputs": n_outputs,
        "module__input_dim": x_train.shape[1],
        "module__activation_function": activation_function,
        "max_epochs": 10,
        "verbose": 0,
    }

    if n_outputs == 1 and model is NeuralNetClassifier:
        with pytest.raises(
            ValueError,
            match=".* number of classes.*",
        ):
            concrete_classifier = model(**params)
        return

    concrete_classifier = model(**params)

    # Compute mean/stdev on training set and normalize both train and test sets with them
    normalizer = StandardScaler()
    x_train = normalizer.fit_transform(x_train)
    x_test = normalizer.transform(x_test)

    _, sklearn_classifier = concrete_classifier.fit_benchmark(x_train, y_train)

    if model._estimator_type == "classifier":  # pylint: disable=protected-access
        # Classification models

        y_pred_sk = sklearn_classifier.predict(x_test)
        y_pred = concrete_classifier.predict(x_test)
        check_accuracy(y_pred_sk, y_pred)

        y_pred_sk = sklearn_classifier.predict_proba(x_test)
        y_pred = concrete_classifier.predict_proba(x_test)
    else:
        # Regression models
        y_pred_sk = sklearn_classifier.predict(x_test)
        y_pred = concrete_classifier.predict(x_test)

    check_r2_score(y_pred_sk, y_pred)


@pytest.mark.parametrize("model", [NeuralNetClassifier, NeuralNetRegressor])
def test_parameter_validation(model):
    """Test that the sklearn quantized NN wrappers validate their parameters"""

    valid_params = {
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

    if model is NeuralNetClassifier:
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
    elif model is NeuralNetRegressor:
        x, y, _ = make_regression(
            1000, n_features=10, n_informative=10, noise=2, random_state=42, coef=True
        )
    else:
        raise ValueError(f"Data generator not implemented for {str(model)}")

    x = x.astype(numpy.float32)

    invalid_params_and_exception_pattern = {
        ("module__n_layers", 0, ".* number of layers.*"),
        ("module__n_w_bits", 0, ".* quantization bitwidth.*"),
        ("module__n_a_bits", 0, ".* quantization bitwidth.*"),
        ("module__n_accum_bits", 0, ".* accumulator bitwidth.*"),
        ("module__n_outputs", 0, ".* number of (outputs|classes).*"),
        ("module__input_dim", 0, ".* number of input dimensions.*"),
    }
    for inv_param in invalid_params_and_exception_pattern:
        params = deepcopy(valid_params)
        params[inv_param[0]] = inv_param[1]

        with pytest.raises(
            ValueError,
            match=inv_param[2],
        ):
            concrete_classifier = model(**params)

            with pytest.raises(
                ValueError,
                match=".* must be trained.*",
            ):
                _ = concrete_classifier.n_bits_quant

            concrete_classifier.fit(x, y)


@pytest.mark.parametrize("use_virtual_lib", [True, False])
@pytest.mark.parametrize(
    "activation_function",
    [
        pytest.param(nn.ReLU),
        pytest.param(nn.Sigmoid),
        pytest.param(nn.SELU),
        pytest.param(nn.CELU),
    ],
)
@pytest.mark.parametrize("model", [NeuralNetClassifier, NeuralNetRegressor])
def test_compile_and_calib(activation_function, model, default_configuration, use_virtual_lib):
    """Test whether the sklearn quantized NN wrappers compile to FHE and execute well on encrypted
    inputs"""

    n_features = 10

    if model is NeuralNetClassifier:
        x, y = make_classification(
            1000,
            n_features=n_features,
            n_redundant=0,
            n_repeated=0,
            n_informative=n_features,
            n_classes=2,
            class_sep=2,
            random_state=42,
        )
    elif model is NeuralNetRegressor:
        x, y, _ = make_regression(
            1000,
            n_features=n_features,
            n_informative=n_features,
            n_targets=2,
            noise=2,
            random_state=42,
            coef=True,
        )
        if y.ndim == 1:
            y = numpy.expand_dims(y, 1)
        y = y.astype(numpy.float32)
    else:
        raise ValueError(f"Data generator not implemented for {str(model)}")

    x = x.astype(numpy.float32)

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

    # Setup dummy class weights that will be converted to a tensor
    class_weights = numpy.asarray([1, 1]).reshape((-1,))

    # Configure a minimal neural network and train it quickly
    params = {
        "module__n_layers": 1,
        "module__n_w_bits": 2,
        "module__n_a_bits": 2,
        "module__n_accum_bits": 5,
        "module__n_outputs": 2,
        "module__input_dim": n_features,
        "module__activation_function": activation_function,
        "max_epochs": 10,
        "verbose": 0,
    }

    if model is NeuralNetClassifier:
        params["criterion__weight"] = class_weights

    clf = model(**params)

    # Compiling a model that is not trained should fail
    with pytest.raises(ValueError, match=".* needs to be calibrated .*"):
        clf.compile(
            x_train,
            configuration=default_configuration,
            use_virtual_lib=use_virtual_lib,
        )

    # Predicting in FHE with a model that is not trained and calibrated should fail
    with pytest.raises(ValueError, match=".* needs to be calibrated .*"):
        x_test_q = numpy.zeros((1, n_features), dtype=numpy.uint8)
        clf.predict(x_test_q, execute_in_fhe=True)

    # Train the model
    clf.fit(x_train, y_train)

    # Predicting with a model that is not compiled should fail
    with pytest.raises(ValueError, match=".* not yet compiled .*"):
        x_test_q = numpy.zeros((1, n_features), dtype=numpy.uint8)
        clf.predict(x_test_q, execute_in_fhe=True)

    # Compile the model
    clf.compile(
        x_train,
        configuration=default_configuration,
        use_virtual_lib=use_virtual_lib,
    )

    # Execute in FHE, but don't check the value.
    # Since FHE execution introduces some stochastic errors,
    # accuracy of FHE compiled classifiers is measured in the benchmarks
    clf.predict(x_test[0, :], execute_in_fhe=True)


def test_custom_net_classifier():
    """Tests a wrapped custom network.

    Gives an example how to use our API to train a custom Torch network through the quantized
    sklearn wrapper.
    """

    class MiniNet(nn.Module):
        """Sparse Quantized Neural Network classifier."""

        def __init__(
            self,
        ):
            """Construct mini net"""
            super().__init__()
            self.features = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))

        def forward(self, x):
            """Forward pass."""
            return self.features(x)

    params = {
        "max_epochs": 10,
        "verbose": 0,
    }

    class MiniCustomNeuralNetClassifier(QuantizedSkorchEstimatorMixin, SKNeuralNetClassifier):
        """Sklearn API wrapper class for a custom network that will be quantized.

        Minimal work is needed to implement training of a custom class."""

        @property
        def base_estimator_type(self):
            return SKNeuralNetClassifier

        @property
        def n_bits_quant(self):
            """Return the number of quantization bits"""
            return 2

        def predict(self, X, execute_in_fhe=False):
            # We just need to do argmax on the predicted probabilities
            return self.predict_proba(X, execute_in_fhe=execute_in_fhe).argmax(axis=1)

    clf = MiniCustomNeuralNetClassifier(MiniNet, **params)

    x, y = make_classification(
        1000,
        n_features=2,
        n_redundant=0,
        n_repeated=0,
        n_informative=2,
        n_classes=2,
        class_sep=2,
        random_state=42,
    )

    x = x.astype(numpy.float32)

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

    # Train the model
    clf.fit_benchmark(x_train, y_train)

    # Test the custom network wrapper in a pipeline with grid CV
    # This will clone the skorch estimator
    pipe_cv = Pipeline(
        [
            ("pca", PCA(n_components=2)),
            ("scaler", StandardScaler()),
            ("net", MiniCustomNeuralNetClassifier(MiniNet, **params)),
        ]
    )

    clf = GridSearchCV(
        pipe_cv,
        {"net__lr": [0.01, 0.1]},
    )
    clf.fit(x_train, y_train)
