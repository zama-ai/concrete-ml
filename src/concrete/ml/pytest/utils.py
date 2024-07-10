"""Common functions or lists for test files, which can't be put in fixtures."""

import copy
import io
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Type, Union

import numpy
import pandas
import pytest
import torch
from numpy.random import RandomState
from torch import nn

from concrete.ml.sklearn.linear_model import SGDClassifier

from ..common.serialization.dumpers import dump, dumps
from ..common.serialization.loaders import load, loads
from ..common.utils import (
    get_model_class,
    get_model_name,
    is_classifier_or_partial_classifier,
    is_model_class_in_a_list,
    is_pandas_type,
    is_regressor_or_partial_regressor,
)
from ..sklearn import (
    KNeighborsClassifier,
    LinearRegression,
    NeuralNetClassifier,
    NeuralNetRegressor,
    TweedieRegressor,
    _get_sklearn_linear_models,
    _get_sklearn_neighbors_models,
    _get_sklearn_neural_net_models,
    _get_sklearn_tree_models,
)


def _get_pytest_param_regressor(model):
    """Get the pytest parameters to use for testing the regression model.

    The pytest parameters selects the model itself, the parameters to use for generating the
    regression data-set and the test identifier (the model's name).

    Args:
        model: The regression model to consider.

    Returns:
        The pytest parameters to use for testing the regression model.
    """

    # We only test LinearRegression models for multiple-targets support
    return pytest.param(
        model,
        {
            "n_samples": 200,
            "n_features": 10,
            "n_informative": 10,
            "n_targets": 2 if get_model_class(model) == LinearRegression else 1,
            "noise": 0,
        },
        id=get_model_name(model),
    )


def _get_pytest_param_classifier(model, n_classes: int):
    """Get the pytest parameters to use for testing the classification model.

    The pytest parameters selects the model itself, the parameters to use for generating the
    classification data-set and the test identifier (the model's name).

    Args:
        model: The classification model to consider.
        n_classes (int): The number of classes to consider when generating the dataset.

    Returns:
        The pytest parameters to use for testing the classification model.
    """
    if get_model_class(model) == KNeighborsClassifier:
        dataset_params = {
            "n_samples": 6,
            "n_features": 2,
            "n_classes": n_classes,
            "n_informative": 2,
            "n_redundant": 0,
        }
    else:
        dataset_params = {
            "n_samples": 100,
            "n_features": 10,
            "n_classes": n_classes,
            "n_informative": 10,
            "n_redundant": 0,
        }

    return pytest.param(
        model,
        dataset_params,
        id=get_model_name(model),
    )


def _get_sklearn_models_and_datasets(model_classes: List, unique_models: bool = False) -> List:
    """Get the pytest parameters to use for testing the given models.

    Args:
        model_classes (List): The models to consider.
        unique_models (bool): If each models should be represented only once.

    Returns:
        models_and_datasets (List): The pytest parameters to use for testing the given models.

    Raises:
        ValueError: If one of the given model is neither considered a regressor nor a classifier.
    """
    models_and_datasets = []

    for model_class in model_classes:
        if is_regressor_or_partial_regressor(model_class):
            models_and_datasets.append(_get_pytest_param_regressor(model_class))

        elif is_classifier_or_partial_classifier(model_class):

            # Unless each models should be represented only once, we test classifier models for both
            # binary and multiclass classification
            # Also, only consider 2 classes for the KNeighborsClassifier model in order to decrease
            # the test execution timings
            n_classes_to_test = (
                [2]
                if unique_models
                or get_model_class(model_class) == KNeighborsClassifier
                or (
                    isinstance(model_class, partial)
                    and model_class.func == SGDClassifier
                    and model_class.keywords.get("fit_encrypted", False)
                )
                else [2, 4]
            )

            for n_classes in n_classes_to_test:
                models_and_datasets.append(
                    _get_pytest_param_classifier(model_class, n_classes=n_classes)
                )

        else:
            raise ValueError(  # pragma: no cover
                f"Model class {model_class} is neither a regressor nor a classifier."
            )

    return models_and_datasets


def get_sklearn_linear_models_and_datasets(
    regressor: bool = True,
    classifier: bool = True,
    unique_models: bool = False,
    select: Optional[Union[str, List[str]]] = None,
    ignore: Optional[Union[str, List[str]]] = None,
) -> List:
    """Get the pytest parameters to use for testing linear models.

    Args:
        regressor (bool): If regressors should be selected.
        classifier (bool): If classifiers should be selected.
        unique_models (bool): If each models should be represented only once.
        select (Optional[Union[str, List[str]]]): If not None, only return models which names (or
            a part of it) match the given string or list of strings. Default to None.
        ignore (Optional[Union[str, List[str]]]): If not None, only return models which names (or
            a part of it) do not match the given string or list of strings. Default to None.

    Returns:
        List: The pytest parameters to use for testing linear models.
    """

    # Get all linear model classes currently available in Concrete ML
    linear_classes = _get_sklearn_linear_models(
        regressor=regressor,
        classifier=classifier,
        select=select,
        ignore=ignore,
    )

    # If the TweedieRegressor has been selected and is allowed to be represented more than once,
    # add a few more testing configuration
    if not unique_models and is_model_class_in_a_list(TweedieRegressor, linear_classes):
        linear_classes += [
            partial(TweedieRegressor, link="auto", power=0.0),
            partial(TweedieRegressor, link="auto", power=2.8),
            partial(TweedieRegressor, link="log", power=1.0),
            partial(TweedieRegressor, link="identity", power=0.0),
        ]

    # If the linear model is SGDClassifier,
    # we need to handle the training parameters
    if is_model_class_in_a_list(SGDClassifier, linear_classes):
        linear_classes += [
            partial(SGDClassifier, fit_encrypted=False),
            # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4460
            # partial(SGDClassifier, fit_encrypted=True, parameters_range=(-1, 1)),
        ]

    return _get_sklearn_models_and_datasets(linear_classes, unique_models=unique_models)


def get_sklearn_tree_models_and_datasets(
    regressor: bool = True,
    classifier: bool = True,
    unique_models: bool = False,
    select: Optional[Union[str, List[str]]] = None,
    ignore: Optional[Union[str, List[str]]] = None,
) -> List:
    """Get the pytest parameters to use for testing tree-based models.

    Args:
        regressor (bool): If regressors should be selected.
        classifier (bool): If classifiers should be selected.
        unique_models (bool): If each models should be represented only once.
        select (Optional[Union[str, List[str]]]): If not None, only return models which names (or
            a part of it) match the given string or list of strings. Default to None.
        ignore (Optional[Union[str, List[str]]]): If not None, only return models which names (or
            a part of it) do not match the given string or list of strings. Default to None.

    Returns:
        List: The pytest parameters to use for testing tree-based models.
    """
    # Get all tree-based model classes currently available in Concrete ML
    tree_classes = _get_sklearn_tree_models(
        regressor=regressor,
        classifier=classifier,
        select=select,
        ignore=ignore,
    )

    return _get_sklearn_models_and_datasets(tree_classes, unique_models=unique_models)


def get_sklearn_neural_net_models_and_datasets(
    regressor: bool = True,
    classifier: bool = True,
    unique_models: bool = False,
    select: Optional[Union[str, List[str]]] = None,
    ignore: Optional[Union[str, List[str]]] = None,
) -> List:
    """Get the pytest parameters to use for testing neural network models.

    Args:
        regressor (bool): If regressors should be selected.
        classifier (bool): If classifiers should be selected.
        unique_models (bool): If each models should be represented only once.
        select (Optional[Union[str, List[str]]]): If not None, only return models which names (or
            a part of it) match the given string or list of strings. Default to None.
        ignore (Optional[Union[str, List[str]]]): If not None, only return models which names (or
            a part of it) do not match the given string or list of strings. Default to None.

    Returns:
        List: The pytest parameters to use for testing neural network models.
    """

    # Get all neural-network model classes currently available in Concrete ML
    selected_neural_net_classes = _get_sklearn_neural_net_models(
        regressor=regressor,
        classifier=classifier,
        select=select,
        ignore=ignore,
    )

    neural_net_classes = []

    # If the NeuralNetRegressor has been selected, configure its initialization parameters
    if is_model_class_in_a_list(NeuralNetRegressor, selected_neural_net_classes):
        neural_net_classes.append(
            partial(
                NeuralNetRegressor,
                module__n_layers=3,
                module__n_w_bits=2,
                module__n_a_bits=2,
                module__n_accum_bits=7,  # Stay with 7 bits for test exec time
                module__n_hidden_neurons_multiplier=1,
                module__activation_function=nn.ReLU,
                max_epochs=10,
                verbose=0,
                callbacks="disable",
            )
        )

    # If the NeuralNetClassifier has been selected, configure its initialization parameters
    if is_model_class_in_a_list(NeuralNetClassifier, selected_neural_net_classes):
        neural_net_classes.append(
            partial(
                NeuralNetClassifier,
                module__n_layers=3,
                module__activation_function=nn.ReLU,
                max_epochs=10,
                verbose=0,
                callbacks="disable",
            )
        )
    return _get_sklearn_models_and_datasets(neural_net_classes, unique_models=unique_models)


def get_sklearn_neighbors_models_and_datasets(
    regressor: bool = True,
    classifier: bool = True,
    unique_models: bool = False,
    select: Optional[Union[str, List[str]]] = None,
    ignore: Optional[Union[str, List[str]]] = None,
) -> List:
    """Get the pytest parameters to use for testing neighbor models.

    Args:
        regressor (bool): If regressors should be selected.
        classifier (bool): If classifiers should be selected.
        unique_models (bool): If each models should be represented only once.
        select (Optional[Union[str, List[str]]]): If not None, only return models which names (or
            a part of it) match the given string or list of strings. Default to None.
        ignore (Optional[Union[str, List[str]]]): If not None, only return models which names (or
            a part of it) do not match the given string or list of strings. Default to None.

    Returns:
        List: The pytest parameters to use for testing neighbor models.
    """
    # Get all neighbor model classes currently available in Concrete ML
    neighbor_classes = _get_sklearn_neighbors_models(
        regressor=regressor,
        classifier=classifier,
        select=select,
        ignore=ignore,
    )

    return _get_sklearn_models_and_datasets(neighbor_classes, unique_models=unique_models)


def get_sklearn_all_models_and_datasets(
    regressor: bool = True,
    classifier: bool = True,
    unique_models: bool = False,
    select: Optional[Union[str, List[str]]] = None,
    ignore: Optional[Union[str, List[str]]] = None,
) -> List:
    """Get the pytest parameters to use for testing all models available in Concrete ML.

    Args:
        regressor (bool): If regressors should be selected.
        classifier (bool): If classifiers should be selected.
        unique_models (bool): If each models should be represented only once.
        select (Optional[Union[str, List[str]]]): If not None, only return models which names (or
            a part of it) match the given string or list of strings. Default to None.
        ignore (Optional[Union[str, List[str]]]): If not None, only return models which names (or
            a part of it) do not match the given string or list of strings. Default to None.

    Returns:
        List: The pytest parameters to use for testing all models available in Concrete ML.
    """
    return (
        get_sklearn_linear_models_and_datasets(
            regressor=regressor,
            classifier=classifier,
            unique_models=unique_models,
            select=select,
            ignore=ignore,
        )
        + get_sklearn_tree_models_and_datasets(
            regressor=regressor,
            classifier=classifier,
            unique_models=unique_models,
            select=select,
            ignore=ignore,
        )
        + get_sklearn_neural_net_models_and_datasets(
            regressor=regressor,
            classifier=classifier,
            unique_models=unique_models,
            select=select,
            ignore=ignore,
        )
        + get_sklearn_neighbors_models_and_datasets(
            regressor=regressor,
            classifier=classifier,
            unique_models=unique_models,
            select=select,
            ignore=ignore,
        )
    )


# All scikit-learn models available in Concrete ML to test and their associated dataset parameters
MODELS_AND_DATASETS = get_sklearn_all_models_and_datasets(regressor=True, classifier=True)

# All unique scikit-learn models available in Concrete ML and their associated dataset parameters
UNIQUE_MODELS_AND_DATASETS = get_sklearn_all_models_and_datasets(
    regressor=True, classifier=True, unique_models=True
)


def instantiate_model_generic(model_class, n_bits, **parameters):
    """Instantiate any Concrete ML model type.

    Args:
        model_class (class): The type of the model to instantiate.
        n_bits (int): The number of quantization to use when initializing the model. For QNNs,
            default parameters are used based on whether `n_bits` is greater or smaller than 8.
        parameters (dict): Hyper-parameters for the model instantiation. For QNNs, these parameters
            will override the matching default ones.

    Returns:
        model_name (str): The type of the model as a string.
        model (object): The model instance.
    """
    # Force multi threaded models to be single threaded as it is not working
    # properly with pytest multi-threading
    if "n_jobs" in model_class().get_params():
        parameters["n_jobs"] = 1

    # If the model is a QNN, set the model using appropriate bit-widths
    if is_model_class_in_a_list(model_class, _get_sklearn_neural_net_models()):
        extra_kwargs = {}
        if n_bits > 8:
            extra_kwargs["module__n_w_bits"] = 3
            extra_kwargs["module__n_a_bits"] = 3
            extra_kwargs["module__n_accum_bits"] = 12
        else:
            extra_kwargs["module__n_w_bits"] = 2
            extra_kwargs["module__n_a_bits"] = 2
            extra_kwargs["module__n_accum_bits"] = 7

        # Disable power-of-two since it sets the input bitwidth to 8
        # and thus increases bitwidth too much for a test
        extra_kwargs["module__power_of_two_scaling"] = False
        extra_kwargs.update(parameters)
        model = model_class(**extra_kwargs)

    # Else, set the model using n_bits
    else:
        model = model_class(n_bits=n_bits, **parameters)

    # Seed the model if it handles "random_state" as a parameter
    model_params = model.get_params()
    if "random_state" in model_params:
        model_params["random_state"] = numpy.random.randint(0, 2**15)

        model.set_params(**model_params)

    return model


def data_calibration_processing(data, n_sample: int, targets=None):
    """Reduce size of the given data-set.

    Args:
        data: The input container to consider
        n_sample (int): Number of samples to keep if the given data-set
        targets: If `dataset` is a `torch.utils.data.Dataset`, it typically contains both the data
            and the corresponding targets. In this case, `targets` must be set to `None`.
            If `data` is instance of `torch.Tensor` or 'numpy.ndarray`, `targets` is expected.

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]: The input data and the target (respectively x and y).

    Raises:
        TypeError: If the 'data-set' does not match any expected type.
    """
    assert n_sample >= 1, "`n_sample` must be greater than or equal to `1`"

    if isinstance(data, torch.utils.data.dataloader.DataLoader):
        n_elements = 0
        all_y, all_x = [], []
        # Iterate over n batches, to apply any necessary torch transformations to the data-set
        for x_batch, y_batch in data:
            all_x.append(x_batch.numpy())
            all_y.append(y_batch.numpy())
            # Number of elements per batch
            n_elements += y_batch.shape[0]
            # If the number of elements within n batches is >= `n_element`, break the loop
            # `n_sample` is reached.
            if n_sample <= n_elements:
                break

        x, y = numpy.concatenate(all_x), numpy.concatenate(all_y)
    elif targets is not None and is_pandas_type(data) and is_pandas_type(targets):
        x = data.to_numpy()
        y = targets.to_numpy()

    elif (
        targets is not None
        and isinstance(targets, (List, numpy.ndarray, torch.Tensor))  # type: ignore[arg-type]
        and isinstance(data, (numpy.ndarray, torch.Tensor))
    ):
        x = numpy.array(data)
        y = numpy.array(targets)
    else:
        raise TypeError(
            "Only numpy arrays, torch tensors and torchvision data-sets are supported. "
            f"Got `{type(data)}` as input type and `{type(targets)}` as target type"
        )

    n_sample = min(len(x), n_sample)

    # Generates a random sample from a given 1-D array
    random_sample = numpy.random.choice(len(x), n_sample, replace=False)

    x = x[random_sample]
    y = y[random_sample]

    return x, y


def load_torch_model(
    model_class: torch.nn.Module,
    state_dict_or_path: Optional[Union[str, Path, Dict[str, Any]]],
    params: Dict,
    device: str = "cpu",
) -> torch.nn.Module:
    """Load an object saved with torch.save() from a file or dict.

    Args:
        model_class (torch.nn.Module): A PyTorch or Brevitas network.
        state_dict_or_path (Optional[Union[str, Path, Dict[str, Any]]]): Path or state_dict
        params (Dict): Model's parameters
        device (str):  Device type.

    Returns:
        torch.nn.Module: A PyTorch or Brevitas network.
    """
    model = model_class(**params)

    if state_dict_or_path is not None:
        if isinstance(state_dict_or_path, (str, Path)):
            state_dict = torch.load(state_dict_or_path, map_location=device)
        else:
            state_dict = state_dict_or_path
        model.load_state_dict(state_dict)

    return model


# pylint: disable-next=too-many-return-statements
def values_are_equal(value_1: Any, value_2: Any) -> bool:
    """Indicate if two values are equal.

    This method takes into account objects of type None, numpy.ndarray, numpy.floating,
    numpy.integer, numpy.random.RandomState or any instance that provides a `__eq__` method.

    Args:
        value_2 (Any): The first value to consider.
        value_1 (Any): The second value to consider.

    Returns:
        bool: If the two values are equal.
    """
    if value_1 is None:
        return value_2 is None

    if isinstance(value_1, numpy.ndarray):
        return (
            isinstance(value_2, numpy.ndarray)
            and numpy.array_equal(value_1, value_2)
            and value_1.dtype == value_2.dtype
        )

    if isinstance(value_1, (numpy.floating, numpy.integer)):
        return (
            isinstance(value_2, (numpy.floating, numpy.integer))
            and value_1 == value_2
            and value_1.dtype == value_2.dtype
        )

    if isinstance(value_1, RandomState):
        if isinstance(value_2, RandomState):
            state_1, state_2 = value_1.get_state(), value_2.get_state()

            # Check that values from both states are equal
            for elt_1, elt_2 in zip(state_1, state_2):
                if not numpy.array_equal(elt_1, elt_2):
                    return False
            return True
        return False

    return value_1 == value_2


def check_serialization(
    object_to_serialize: Any,
    expected_type: Type,
    equal_method: Optional[Callable] = None,
    check_str: bool = True,
):
    """Check that the given object can properly be serialized.

    This function serializes all objects using the `dump`, `dumps`, `load` and `loads` functions
    from Concrete ML. If the given object provides a `dump` and `dumps` method, they are also
    serialized using these.

    Args:
        object_to_serialize (Any): The object to serialize.
        expected_type (Type): The object's expected type.
        equal_method (Optional[Callable]): The function to use to compare the two loaded objects.
            Default to `values_are_equal`.
        check_str (bool): If the JSON strings should also be checked. Default to True.
    """
    # apidocs does not work properly when the function is directly in the default value.
    if equal_method is None:
        equal_method = values_are_equal

    assert (
        isinstance(object_to_serialize, expected_type)
        if expected_type is not None
        else object_to_serialize is None
    )

    dump_method_to_test = [False]

    # If the given object provides a `dump`, `dumps` `dump_dict` method (which indicates that they
    # are Concrete ML serializable classes) and are instantiated, run the check using these as well
    if (
        hasattr(object_to_serialize, "dump")
        and hasattr(object_to_serialize, "dumps")
        and hasattr(object_to_serialize, "dump_dict")
        and not isinstance(object_to_serialize, type)
    ):
        dump_method_to_test.append(True)

    for use_dump_method in dump_method_to_test:

        # Dump the object as a string
        if use_dump_method:
            dumped_str = object_to_serialize.dumps()
        else:
            dumped_str = dumps(object_to_serialize)

        # Load the object using a string
        loaded = loads(dumped_str)

        # Assert that the loaded object is equal to the initial one
        assert isinstance(loaded, expected_type) if expected_type is not None else loaded is None, (
            f"Loaded object (from string) is not of the expected type. Expected {expected_type}, "
            f"got {type(loaded)}."
        )
        assert equal_method(object_to_serialize, loaded), (
            "Loaded object (from string) is not equal to the initial one, using equal method "
            f"{equal_method}."
        )

        if check_str:
            # Dump the loaded object as a string
            if use_dump_method:
                re_dumped_str = loaded.dumps()
            else:
                re_dumped_str = dumps(loaded)

            # Assert that both JSON strings are equal. This will not work if some objects were
            # dumped using Skops or pickle (such as a scikit-learn model or a Type object)
            assert isinstance(dumped_str, str) and isinstance(re_dumped_str, str), (
                f"One of the dumped strings is not a string. Got {type(dumped_str)} and "
                f"{type(re_dumped_str)}."
            )
            assert dumped_str == re_dumped_str, (
                "Dumped strings are not equal. This could come from objects that were serialized "
                "in binary strings using Skops or pickle. If this cannot be avoided, please set "
                "`check_str` to False."
            )

        # Define a buffer to test serialization in a file
        with io.StringIO() as buffer:
            buffer.seek(0, 0)

            # Dump the object in a file
            if use_dump_method:
                object_to_serialize.dump(buffer)
            else:
                dump(object_to_serialize, buffer)

            buffer.seek(0, 0)

            # Load the object from the file
            loaded = load(buffer)

            # Assert that the loaded object is equal to the initial one
            assert (
                isinstance(loaded, expected_type) if expected_type is not None else loaded is None
            ), (
                "Loaded object (from file) is not of the expected type. Expected "
                f"{expected_type}, got {type(loaded)}."
            )
            assert equal_method(object_to_serialize, loaded), (
                "Loaded object (from file) is not equal to the initial one, using equal method "
                f"{equal_method}."
            )


def get_random_samples(x: numpy.ndarray, n_sample: int) -> numpy.ndarray:
    """Select `n_sample` random elements from a 2D NumPy array.

    Args:
        x (numpy.ndarray): The 2D NumPy array from which random rows will be selected.
        n_sample (int): The number of rows to randomly select.

    Returns:
        numpy.ndarray: A new 2D NumPy array containing the randomly selected rows.

    Raises:
        AssertionError: If `n_sample` is not within the range (0, x.shape[0]) or
            if `x` is not a 2D array.
    """

    # Sanity checks
    assert 0 < n_sample < x.shape[0]
    assert len(x.shape) == 2

    random_rows_indices = numpy.random.choice(x.shape[0], size=n_sample, replace=False)
    return x[random_rows_indices]


def pandas_dataframe_are_equal(
    df_1: pandas.DataFrame,
    df_2: pandas.DataFrame,
    float_rtol: float = 1.0e-5,
    float_atol: float = 1.0e-8,
    equal_nan: bool = False,
):
    """Determine if both data-frames are identical.

    Args:
        df_1 (pandas.DataFrame): The first data-frame to consider.
        df_2 (pandas.DataFrame): The second data-frame to consider.
        float_rtol (float): Numpy's relative tolerance parameter to use when comparing columns with
            floating point values. Default to 1.e-5.
        float_atol (float): Numpy's absolute tolerance parameter to use when comparing columns with
            floating point values. Default to 1.e-8.
        equal_nan (bool):  Whether to compare NaN values as equal. Default to False.

    Returns:
        Bool: Wether both data-frames are equal.
    """
    df_1 = copy.copy(df_1)
    df_2 = copy.copy(df_2)

    # Select columns with floating point values
    float_columns = df_1.select_dtypes(include="float").columns

    # Check if the float columns contain the same values
    float_equal = numpy.isclose(
        df_1[float_columns],
        df_2[float_columns],
        rtol=float_rtol,
        atol=float_atol,
        equal_nan=equal_nan,
    ).all()

    # Select other columns (integers, objects, ...)
    non_float_columns = df_1.select_dtypes(exclude="float").columns

    # In case NaN values must be considered equal, replace them by a custom placeholder before
    # comparing the data-frames
    if equal_nan:
        placeholder = "<NA>"

        # Make sure this placeholder does not already exist in the data-frames
        assert (
            not df_1[non_float_columns].isin([placeholder]).any().any()
            or not df_2[non_float_columns].isin([placeholder]).any().any()
        ), (
            f"The placeholder value '{placeholder}' already exists in the string columns and thus "
            "cannot be used for comparing the data-frames."
        )

        df_1 = df_1[non_float_columns].fillna(placeholder)
        df_2 = df_2[non_float_columns].fillna(placeholder)

    # Check if non-float columns contain the same values
    string_equal = df_1.eq(df_2).all().all()

    return float_equal and string_equal
