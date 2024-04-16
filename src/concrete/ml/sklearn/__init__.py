"""Import sklearn models."""

from typing import Dict, List, Optional, Union

from ..common.debugging.custom_assert import assert_true
from ..common.utils import (
    get_model_name,
    is_classifier_or_partial_classifier,
    is_regressor_or_partial_regressor,
)
from .base import (
    _ALL_SKLEARN_MODELS,
    _LINEAR_MODELS,
    _NEIGHBORS_MODELS,
    _NEURALNET_MODELS,
    _TREE_MODELS,
)
from .glm import GammaRegressor, PoissonRegressor, TweedieRegressor
from .linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDClassifier,
    SGDRegressor,
)
from .neighbors import KNeighborsClassifier
from .qnn import NeuralNetClassifier, NeuralNetRegressor
from .rf import RandomForestClassifier, RandomForestRegressor
from .svm import LinearSVC, LinearSVR
from .tree import DecisionTreeClassifier, DecisionTreeRegressor
from .xgb import XGBClassifier, XGBRegressor


def _get_sklearn_models() -> Dict[str, List]:
    """Return the list of available scikit-learn models in Concrete ML.

    Returns:
        sklearn_models (Dict[str, List]): The lists of scikit-learn models available in Concrete ML.
    """

    # Import anything in sklearn, just to force the import, to populate _ALL_SKLEARN_MODELS list
    # pylint: disable-next=unused-import, import-outside-toplevel, cyclic-import, redefined-outer-name  # noqa: E501
    from .linear_model import LinearRegression  # noqa: F401, F811

    # We return sorted lists such that it is ordered, to avoid notably issues when it is used
    # in @pytest.mark.parametrize
    sklearn_models = {
        "all": sorted(list(_ALL_SKLEARN_MODELS), key=lambda m: m.__name__),
        "linear": sorted(list(_LINEAR_MODELS), key=lambda m: m.__name__),
        "tree": sorted(list(_TREE_MODELS), key=lambda m: m.__name__),
        "neural_net": sorted(list(_NEURALNET_MODELS), key=lambda m: m.__name__),
        "neighbors": sorted(list(_NEIGHBORS_MODELS), key=lambda m: m.__name__),
    }
    return sklearn_models


def _filter_models(
    models,
    classifier: bool = True,
    regressor: bool = True,
    select: Optional[Union[str, List[str]]] = None,
    ignore: Optional[Union[str, List[str]]] = None,
):
    """Return a list of models filtered by the given conditions, sorted by name.

    Args:
        models: The list of models to consider.
        classifier (bool): If classifiers should be considered. Default to True.
        regressor (bool): If regressors should be considered. Default to True.
        select (Optional[Union[str, List[str]]]): If not None, only return models which names (or
            a part of it) match the given string or list of strings. Default to None.
        ignore (Optional[Union[str, List[str]]]): If not None, only return models which names (or
            a part of it) do not match the given string or list of strings. Default to None.

    Returns:
        The filtered list of models available in Concrete ML, sorted by name.

    """
    assert_true(classifier or regressor, "Please set at least one option")

    selected_models = []

    if classifier:
        selected_models += [model for model in models if is_classifier_or_partial_classifier(model)]

    if regressor:
        selected_models += [model for model in models if is_regressor_or_partial_regressor(model)]

    if select is not None:
        if isinstance(select, str):
            select = [select]

        # Filter the selected models: only select the ones which name matches the given string
        # (or at least one of the given strings if it is a list)
        selected_models = [
            model for name in select for model in selected_models if name in get_model_name(model)
        ]

    if ignore is not None:
        if isinstance(ignore, str):
            ignore = [ignore]

        # Filter the selected models: remove the ones which name matches the given string (or at
        # least one of the given strings if it is a list)
        selected_models = [
            model
            for name in ignore
            for model in selected_models
            if name not in get_model_name(model)
        ]

    # Return a sorted list in order to avoid issues when used in @pytest.mark.parametrize
    return sorted(selected_models, key=lambda m: m.__name__)


def _get_sklearn_linear_models(
    classifier: bool = True,
    regressor: bool = True,
    select: Optional[Union[str, List[str]]] = None,
    ignore: Optional[Union[str, List[str]]] = None,
):
    """Return a list of available linear models in Concrete ML.

    The list is sorted by name and can be filtered using the given conditions.

    Args:
        classifier (bool): If classifiers should be considered. Default to True.
        regressor (bool): If regressors should be considered. Default to True.
        select (Optional[Union[str, List[str]]]): If not None, only return models which
            names (or a part of it) match the given string or list of strings. Default to None.
        ignore (Optional[Union[str, List[str]]]): If not None, only return models
            which names (or a part of it) do not match the given string or list of strings. Default
            to None.

    Returns:
        The filtered list of linear models available in Concrete ML, sorted by name.
    """
    linear_models = _get_sklearn_models()["linear"]
    return _filter_models(
        linear_models,
        classifier=classifier,
        regressor=regressor,
        select=select,
        ignore=ignore,
    )


def _get_sklearn_tree_models(
    classifier: bool = True,
    regressor: bool = True,
    select: Optional[Union[str, List[str]]] = None,
    ignore: Optional[Union[str, List[str]]] = None,
):
    """Return the list of available tree-based models in Concrete ML.

    The list is sorted by name and can be filtered using the given conditions.

    Args:
        classifier (bool): If classifiers should be considered. Default to True.
        regressor (bool): If regressors should be considered. Default to True.
        select (Union[str, List[str]]): If not None, only return models which names (or
            a part of it) match the given string or list of strings. Default to None.
        ignore (Union[str, List[str]]): If not None, only return models which names
            (or a part of it) do not match the given string or list of strings. Default to None.

    Returns:
        The filtered list of tree-based models available in Concrete ML, sorted by name.
    """
    tree_models = _get_sklearn_models()["tree"]
    return _filter_models(tree_models, classifier, regressor, select=select, ignore=ignore)


def _get_sklearn_neural_net_models(
    classifier: bool = True,
    regressor: bool = True,
    select: Optional[Union[str, List[str]]] = None,
    ignore: Optional[Union[str, List[str]]] = None,
):
    """Return the list of available neural network models in Concrete ML.

    The list is sorted by name and can be filtered using the given conditions.

    Args:
        classifier (bool): If classifiers should be considered. Default to True.
        regressor (bool): If regressors should be considered. Default to True.
        select (Optional[Union[str, List[str]]]): If not None, only return models which
            names (or a part of it) match the given string or list of strings. Default to None.
        ignore (Optional[Union[str, List[str]]]): If not None, only return models
            which names (or a part of it) do not match the given string or list of strings. Default
            to None.

    Returns:
        The filtered list of neural network models available in Concrete ML, sorted by name.
    """
    neural_network_models = _get_sklearn_models()["neural_net"]
    return _filter_models(
        neural_network_models,
        classifier=classifier,
        regressor=regressor,
        select=select,
        ignore=ignore,
    )


def _get_sklearn_neighbors_models(
    classifier: bool = True,
    regressor: bool = True,
    select: Optional[Union[str, List[str]]] = None,
    ignore: Optional[Union[str, List[str]]] = None,
):
    """Return the list of available neighbor models in Concrete ML.

    The list is sorted by name and can be filtered using the given conditions.

    Args:
        classifier (bool): If classifiers should be considered. Default to True.
        regressor (bool): If regressors should be considered. Default to True.
        select (Optional[Union[str, List[str]]]): If not None, only return models which
            names (or a part of it) match the given string or list of strings. Default to None.
        ignore (Optional[Union[str, List[str]]]): If not None, only return models
            which names (or a part of it) do not match the given string or list of strings. Default
            to None.

    Returns:
        The filtered list of neighbor models available in Concrete ML, sorted by name.
    """
    neighbor_models = _get_sklearn_models()["neighbors"]
    return _filter_models(
        neighbor_models,
        classifier=classifier,
        regressor=regressor,
        select=select,
        ignore=ignore,
    )


def _get_sklearn_all_models(
    classifier: bool = True,
    regressor: bool = True,
    select: Optional[Union[str, List[str]]] = None,
    ignore: Optional[Union[str, List[str]]] = None,
):
    """Return the list of all available models in Concrete ML.

    The list is sorted by name and can be filtered using the given conditions.

    Args:
        classifier (bool): If classifiers should be considered.
        regressor (bool): If regressors should be considered.
        select (Optional[Union[str, List[str]]]): If not None, only return models which
            names (or a part of it) match the given string or list of strings. Default to None.
        ignore (Optional[Union[str, List[str]]]): If not None, only return models
            which names (or a part of it) do not match the given string or list of strings. Default
            to None.

    Returns:
        The filtered list of all models available in Concrete ML, sorted by name.
    """
    all_models = _get_sklearn_models()["all"]
    return _filter_models(
        all_models,
        classifier=classifier,
        regressor=regressor,
        select=select,
        ignore=ignore,
    )
