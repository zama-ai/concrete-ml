"""Tests lists of models in Concrete ML."""
from concrete.ml.sklearn import get_sklearn_models
from concrete.ml.sklearn.glm import GammaRegressor, PoissonRegressor, TweedieRegressor
from concrete.ml.sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from concrete.ml.sklearn.qnn import NeuralNetClassifier, NeuralNetRegressor
from concrete.ml.sklearn.rf import RandomForestClassifier, RandomForestRegressor
from concrete.ml.sklearn.svm import LinearSVC, LinearSVR
from concrete.ml.sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from concrete.ml.sklearn.xgb import XGBClassifier, XGBRegressor


def test_get_sklearn_models():
    """List all available models in Concrete ML."""
    dic = get_sklearn_models()
    cml_list = dic["all"]
    linear_list = dic["linear"]
    tree_list = dic["tree"]
    neuralnet_list = dic["neural_net"]

    print("All models: ")
    for m in cml_list:
        print(f"     {m}")

    print("Linear models: ")
    for m in linear_list:
        print(f"     {m}")

    print("Tree models: ")
    for m in tree_list:
        print(f"     {m}")

    print("Neural net models: ")
    for m in neuralnet_list:
        print(f"     {m}")

    # Check values
    expected_neuralnet_list = [NeuralNetClassifier, NeuralNetRegressor]
    assert (
        neuralnet_list == expected_neuralnet_list
    ), "Please change the expected number of models if you add new models"

    expected_tree_list = [
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        RandomForestClassifier,
        RandomForestRegressor,
        XGBClassifier,
        XGBRegressor,
    ]
    assert (
        tree_list == expected_tree_list
    ), "Please change the expected number of models if you add new models"

    expected_linear_list = [
        ElasticNet,
        GammaRegressor,
        Lasso,
        LinearRegression,
        LinearSVC,
        LinearSVR,
        LogisticRegression,
        PoissonRegressor,
        Ridge,
        TweedieRegressor,
    ]
    assert (
        linear_list == expected_linear_list
    ), "Please change the expected number of models if you add new models"

    # Check number
    assert cml_list == sorted(
        expected_linear_list + expected_neuralnet_list + expected_tree_list,
        key=lambda m: m.__name__,
    ), "Please change the expected number of models if you add new models"
