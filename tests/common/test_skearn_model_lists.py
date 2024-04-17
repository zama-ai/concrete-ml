"""Tests lists of models in Concrete ML."""

from concrete.ml.pytest.utils import MODELS_AND_DATASETS, UNIQUE_MODELS_AND_DATASETS
from concrete.ml.sklearn import (
    _get_sklearn_all_models,
    _get_sklearn_linear_models,
    _get_sklearn_neighbors_models,
    _get_sklearn_neural_net_models,
    _get_sklearn_tree_models,
)
from concrete.ml.sklearn.glm import GammaRegressor, PoissonRegressor, TweedieRegressor
from concrete.ml.sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDClassifier,
    SGDRegressor,
)
from concrete.ml.sklearn.neighbors import KNeighborsClassifier
from concrete.ml.sklearn.qnn import NeuralNetClassifier, NeuralNetRegressor
from concrete.ml.sklearn.rf import RandomForestClassifier, RandomForestRegressor
from concrete.ml.sklearn.svm import LinearSVC, LinearSVR
from concrete.ml.sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from concrete.ml.sklearn.xgb import XGBClassifier, XGBRegressor


def test_get_sklearn_models():
    """List all available models in Concrete ML."""
    all_models = _get_sklearn_all_models()
    linear_models = _get_sklearn_linear_models()
    tree_models = _get_sklearn_tree_models()
    neural_network_models = _get_sklearn_neural_net_models()
    neighbor_models = _get_sklearn_neighbors_models()

    print("All models: ")
    for m in all_models:
        print(f"     {m}")

    print("Linear models: ")
    for m in linear_models:
        print(f"     {m}")

    print("Tree models: ")
    for m in tree_models:
        print(f"     {m}")

    print("Neural net models: ")
    for m in neural_network_models:
        print(f"     {m}")

    print("Neighbors models: ")
    for m in neighbor_models:
        print(f"     {m}")

    # Check values
    expected_neural_network_models = {NeuralNetClassifier, NeuralNetRegressor}
    assert (
        set(neural_network_models) == expected_neural_network_models
    ), "Please change the expected number of models if neural network models have been added"

    expected_tree_models = {
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
        XGBClassifier,
        XGBRegressor,
    }
    assert (
        set(tree_models) == expected_tree_models
    ), "Please change the expected number of tree models if new models have been added"

    expected_linear_models = {
        ElasticNet,
        GammaRegressor,
        Lasso,
        LinearRegression,
        LinearSVC,
        LinearSVR,
        LogisticRegression,
        PoissonRegressor,
        Ridge,
        SGDRegressor,
        SGDClassifier,
        TweedieRegressor,
    }

    assert (
        set(linear_models) == expected_linear_models
    ), "Please change the expected number of linear models if new models have been added"

    expected_neighbor_models = {KNeighborsClassifier}

    # Check number
    expected = set(all_models)
    obtained = (
        expected_linear_models
        | expected_tree_models
        | expected_neural_network_models
        | expected_neighbor_models
    )
    assert expected == obtained, (
        "Please change the expected number of models if new models have been added\n"
        f"Missing: {expected ^ obtained}"
    )


def test_models_and_datasets():
    """Check that the tested model's configuration lists remain fixed."""

    assert len(MODELS_AND_DATASETS) == 32
    assert len(UNIQUE_MODELS_AND_DATASETS) == 21
