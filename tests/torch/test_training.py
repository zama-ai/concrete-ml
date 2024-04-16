"""Tests for FHE training."""

import numpy
import torch
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from concrete.ml.pytest.torch_models import ManualLogisticRegressionTraining
from concrete.ml.torch.compile import build_quantized_module


def create_batches(x, y, batch_size, num_iterations):
    """Create batches of data from the given datasets."""
    n_examples = batch_size * num_iterations
    repeated_x = numpy.tile(x, (int(numpy.ceil(n_examples / x.shape[0])), 1))[:n_examples]
    repeated_y = numpy.tile(y, int(numpy.ceil(n_examples / y.size)))[:n_examples]

    x_batches = repeated_x.reshape(-1, batch_size, x.shape[1])
    y_batches = repeated_y.reshape(-1, batch_size, 1)

    return torch.tensor(x_batches, dtype=torch.float32), torch.tensor(
        y_batches, dtype=torch.float32
    )


def initialize_parameters(n_batch, n_features, n_targets, min_val=-3.0, max_val=3.0, seed=50):
    """Initializes weights and bias parameters."""
    torch.manual_seed(seed)
    return (
        (max_val - min_val) * torch.rand(size=(n_batch, n_features, n_targets)) + min_val,
        (max_val - min_val) * torch.rand(size=(n_batch, 1, n_targets)) + min_val,
    )


def train_and_evaluate_model(
    x_train, y_train, x_test, y_test, batch_size, iteration, model, model_type="torch"
):
    """
    Train and evaluate the given model, supporting both torch and quantized models.
    """
    x_train_batches, y_train_batches = create_batches(x_train, y_train, batch_size, iteration)
    x_test_batches, _ = create_batches(x_test, y_test, batch_size, iteration)

    n_features = x_train_batches.shape[2]
    weights, bias = initialize_parameters(1, n_features, 1)

    if model_type == "torch":
        trained_weights = weights
        for i in range(iteration):
            trained_weights = model.forward(
                x_train_batches[[i]], y_train_batches[[i]], trained_weights, bias
            )
        trained_weights = trained_weights.detach().numpy()
    elif model_type == "quantized":
        n_bits = 24

        # Build a compile set for weights and biases
        weights_compile, bias_compile = initialize_parameters(iteration, n_features, 1)
        q_module = build_quantized_module(
            model,
            torch_inputset=(x_train_batches, y_train_batches, weights_compile, bias_compile),
            n_bits=n_bits,
        )
        trained_weights = weights.detach().numpy()
        for i in range(iteration):
            trained_weights = q_module.forward(
                x_train_batches.detach().numpy()[[i]],
                y_train_batches.detach().numpy()[[i]],
                trained_weights,
                bias.detach().numpy(),
            )

    predictions = []
    for i in range(x_test_batches.shape[0]):
        batch_predictions = model.predict(
            x_test_batches[[i]], torch.tensor(trained_weights, dtype=torch.float32), bias
        ).round()
        predictions.append(batch_predictions)
    predictions = torch.cat(predictions).numpy().flatten()

    min_length = min(len(predictions), len(y_test))
    return accuracy_score(y_test[:min_length], predictions[:min_length])


def test_sgd_training_manual():
    """Trains a logistic regression with SGD in torch and quantized."""
    # Train on the bias when multi output is available in concrete
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4131

    # Load and preprocess the dataset
    x, y = datasets.load_breast_cancer(return_X_y=True)
    x_train, x_test, y_train, y_test = train_test_split(
        MinMaxScaler(feature_range=(-1, 1)).fit_transform(x),
        y,
        test_size=0.1,
        random_state=0,
    )

    # Define torch model
    model = ManualLogisticRegressionTraining(learning_rate=1)

    # Define batch size and number of iterations
    batch_size, iteration = 32, 100

    # Train and evaluate custom logistic regression model
    accuracy_torch = train_and_evaluate_model(
        x_train, y_train, x_test, y_test, batch_size, iteration, model, model_type="torch"
    )

    # Train and evaluate sklearn logistic regression model
    sk_model = LogisticRegression(fit_intercept=False).fit(x_train, y_train)
    accuracy_sklearn = accuracy_score(y_test, sk_model.predict(x_test))

    assert (
        abs(accuracy_torch - accuracy_sklearn) < 0.01
    ), "Torch accuracy should be within 1% of sklearn's."

    accuracy_q_module = train_and_evaluate_model(
        x_train, y_train, x_test, y_test, batch_size, iteration, model, model_type="quantized"
    )

    # Quantized accuracy should match torch
    assert (
        abs(accuracy_torch - accuracy_q_module) < 0.01
    ), "SGD Training should be within 1% accuracy from the torch accuracy."
