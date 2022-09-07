"""Tests for the torch to numpy module."""
import random

import numpy
import pytest
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from concrete.ml.sklearn import DecisionTreeClassifier, DecisionTreeRegressor
from concrete.ml.sklearn.qnn import NeuralNetClassifier


def test_seed_1():
    """Test python and numpy seeding"""

    # Python random
    for _ in range(10):
        print(random.randint(0, 1000))

    # Numpy random
    for _ in range(10):
        print(numpy.random.randint(0, 1000))
        print(numpy.random.uniform(-100, 100, size=(3, 3)))


def test_seed_2():
    """Test python and numpy seeding"""

    # Python random
    for _ in range(20):
        print(random.randint(0, 100))

    # Numpy random
    for _ in range(20):
        print(numpy.random.randint(0, 100))
        print(numpy.random.uniform(-10, 100, size=(3, 3)))


@pytest.mark.parametrize("random_inputs", [numpy.random.randint(0, 2**15, size=20)])
def test_seed_3(random_inputs):
    """Test python and numpy seeding for pytest parameters"""

    print("Random inputs", random_inputs)


@pytest.mark.parametrize("n_targets", [2])
@pytest.mark.parametrize("input_dim", [100])
def test_seed_sklearn_regression(n_targets, input_dim, load_data, default_configuration):
    """Test seeding of sklearn regression model"""

    # Get the dataset. The data generation is seeded in load_data.
    x, y = load_data(
        dataset="regression",
        n_samples=1000,
        n_features=input_dim,
        n_informative=input_dim,
        n_targets=n_targets,
    )

    model = DecisionTreeRegressor(
        n_bits=6, max_depth=7, random_state=numpy.random.randint(0, 2**15)
    )
    model, sklearn_model = model.fit_benchmark(x, y)

    print("model", tree.plot_tree(model.sklearn_model), sklearn_model)

    # Test the determinism of our package (even if the bitwidth may be too large)
    try:
        model.compile(x, configuration=default_configuration, show_mlir=True)
    except RuntimeError as err:
        print(err)
    except AssertionError as err:
        print(err)


@pytest.mark.parametrize("n_classes", [2])
@pytest.mark.parametrize("input_dim", [100])
def test_seed_sklearn_classification(n_classes, input_dim, load_data, default_configuration):
    """Test seeding of sklearn classification model"""

    # Get the dataset. The data generation is seeded in load_data.
    x, y = load_data(
        dataset="classification",
        n_samples=1000,
        n_features=input_dim,
        n_redundant=0,
        n_repeated=0,
        n_informative=input_dim,
        n_classes=n_classes,
        class_sep=2,
    )

    model = DecisionTreeClassifier(
        n_bits=6, max_depth=7, random_state=numpy.random.randint(0, 2**15)
    )
    model, sklearn_model = model.fit_benchmark(x, y)

    print("model", tree.plot_tree(model.sklearn_model), sklearn_model)

    # Test the determinism of our package (even if the bitwidth may be too large)
    try:
        model.compile(x, configuration=default_configuration, show_mlir=True)
    except RuntimeError as err:
        print(err)
    except AssertionError as err:
        print(err)


@pytest.mark.parametrize(
    "n_layers",
    [3],
)
@pytest.mark.parametrize("n_bits_w_a", [16])
@pytest.mark.parametrize("n_classes", [2])
@pytest.mark.parametrize("n_accum_bits", [32])
@pytest.mark.parametrize("activation_function", [pytest.param(nn.ReLU)])
@pytest.mark.parametrize("input_dim", [10])
def test_seed_torch(
    n_layers,
    n_bits_w_a,
    n_accum_bits,
    activation_function,
    n_classes,
    input_dim,
    load_data,
    default_configuration,
):
    """Test seeding of torch function"""

    # Get the dataset. The data generation is seeded in load_data.
    x, y = load_data(
        dataset="classification",
        n_samples=1000,
        n_features=input_dim,
        n_redundant=0,
        n_repeated=0,
        n_informative=input_dim,
        n_classes=n_classes,
        class_sep=2,
    )

    x = x.astype(numpy.float32)

    # Perform a classic test-train split (deterministic by fixing the seed)
    x_train, x_test, y_train, _ = train_test_split(
        x,
        y,
        test_size=0.95,
        random_state=numpy.random.randint(0, 2**15),
    )

    params = {
        "module__n_layers": n_layers,
        "module__n_w_bits": n_bits_w_a,
        "module__n_a_bits": n_bits_w_a,
        "module__n_accum_bits": n_accum_bits,
        "module__n_outputs": n_classes,
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

    for name, param in sklearn_classifier.module.named_parameters():
        print(name, param.detach().numpy())

    # Test the determinism of our package (even if the bitwidth may be too large)
    try:
        concrete_classifier.compile(x_train, configuration=default_configuration, show_mlir=True)
    except RuntimeError as err:
        print(err)
    except AssertionError as err:
        print(err)
