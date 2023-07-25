"""Import sklearn models."""
from typing import List

from ..common.debugging.custom_assert import assert_true
from ..common.utils import is_classifier_or_partial_classifier, is_regressor_or_partial_regressor
from .base import _ALL_SKLEARN_MODELS, _LINEAR_MODELS, _NEURALNET_MODELS, _TREE_MODELS
from .glm import GammaRegressor, PoissonRegressor, TweedieRegressor
from .linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from .qnn import NeuralNetClassifier, NeuralNetRegressor
from .rf import RandomForestClassifier, RandomForestRegressor
from .svm import LinearSVC, LinearSVR
from .tree import DecisionTreeClassifier, DecisionTreeRegressor
from .xgb import XGBClassifier, XGBRegressor


def get_sklearn_models():
    """Return the list of available models in Concrete ML.

    Returns:
        the lists of models in Concrete ML
    """

    # Import anything in sklearn, just to force the import, to populate _ALL_SKLEARN_MODELS list
    # pylint: disable-next=unused-import, import-outside-toplevel, cyclic-import, redefined-outer-name  # noqa: E501
    from .linear_model import LinearRegression  # noqa: F401, F811

    # We return sorted lists such that it is ordered, to avoid notably issues when it is used
    # in @pytest.mark.parametrize
    ans = {
        "all": sorted(list(_ALL_SKLEARN_MODELS), key=lambda m: m.__name__),
        "linear": sorted(list(_LINEAR_MODELS), key=lambda m: m.__name__),
        "tree": sorted(list(_TREE_MODELS), key=lambda m: m.__name__),
        "neural_net": sorted(list(_NEURALNET_MODELS), key=lambda m: m.__name__),
    }
    return ans


def _filter_models(prelist, classifier: bool, regressor: bool, str_in_class_name: List[str] = None):
    """Return the models which are in prelist and follow (classifier, regressor) conditions.

    Args:
        prelist: list of models
        classifier (bool): whether you want classifiers or not
        regressor (bool): whether you want regressors or not
        str_in_class_name (List[str]): if not None, only return models with the given string or
            list of strings as a substring in their class name

    Returns:
        the sublist which fulfills the (classifier, regressor, str_in_class_name) conditions.

    """
    assert_true(classifier or regressor, "Please set at least one option")

    answer = []

    if classifier:
        answer += [m for m in prelist if is_classifier_or_partial_classifier(m)]

    if regressor:
        answer += [m for m in prelist if is_regressor_or_partial_regressor(m)]

    if str_in_class_name is not None:
        if isinstance(str_in_class_name, str):
            str_in_class_name = [str_in_class_name]

        for name in str_in_class_name:
            answer += [m for m in answer if name in m.__name__]

    # We return a sorted list such that it is ordered, to avoid notably issues when it is used
    # in @pytest.mark.parametrize
    return sorted(answer, key=lambda m: m.__name__)


def get_sklearn_linear_models(
    classifier: bool = True, regressor: bool = True, str_in_class_name: List[str] = None
):
    """Return the list of available linear models in Concrete ML.

    Args:
        classifier (bool): whether you want classifiers or not
        regressor (bool): whether you want regressors or not
        str_in_class_name (List[str]): if not None, only return models with the given string or
            list of strings as a substring in their class name

    Returns:
        the lists of linear models in Concrete ML
    """
    prelist = get_sklearn_models()["linear"]
    return _filter_models(prelist, classifier, regressor, str_in_class_name)


def get_sklearn_tree_models(
    classifier: bool = True, regressor: bool = True, str_in_class_name: List[str] = None
):
    """Return the list of available tree models in Concrete ML.

    Args:
        classifier (bool): whether you want classifiers or not
        regressor (bool): whether you want regressors or not
        str_in_class_name (List[str]): if not None, only return models with the given string or
            list of strings as a substring in their class name

    Returns:
        the lists of tree models in Concrete ML
    """
    prelist = get_sklearn_models()["tree"]
    return _filter_models(prelist, classifier, regressor, str_in_class_name)


def get_sklearn_neural_net_models(
    classifier: bool = True, regressor: bool = True, str_in_class_name: List[str] = None
):
    """Return the list of available neural net models in Concrete ML.

    Args:
        classifier (bool): whether you want classifiers or not
        regressor (bool): whether you want regressors or not
        str_in_class_name (List[str]): if not None, only return models with the given string or
            list of strings as a substring in their class name

    Returns:
        the lists of neural net models in Concrete ML
    """
    prelist = get_sklearn_models()["neural_net"]
    return _filter_models(prelist, classifier, regressor, str_in_class_name)
