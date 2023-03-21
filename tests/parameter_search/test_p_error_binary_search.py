"""Tests 'p_error' parameter search on a quantized NN with Brevitas for classition tasks."""

from pathlib import Path

import numpy
import pytest
from sklearn.datasets import make_classification
from tensorflow import keras
from torchvision import datasets, transforms

from concrete.ml.pytest.torch_models import QNNFashionMNIST, QuantCustomModel, TorchCustomModel
from concrete.ml.pytest.utils import (
    data_calibration_processing,
    get_torchvision_dataset,
    load_torch_model,
)
from concrete.ml.search_parameters import BinarySearch

DATASETS_ARGS = {
    "FashionMNIST": {
        "dataset": datasets.FashionMNIST,
        "train_transform": transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(0.2859, 0.3530)]
        ),
        "test_transform": transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(0.2859, 0.3530)]
        ),
        "dataset_size": 60000,
    },
    "CustomModel": {
        "param": {
            "n_samples": 1000,
            "n_classes": 3,
            "n_features": 6,
            "n_informative": 4,
            "n_redundant": 2,
            "random_state": 42,
        }
    },
}

MODELS_ARGS = {
    "FashionMNIST": {
        "quant": {
            "model_class": QNNFashionMNIST,
            "path": Path(__file__).parent / "FashionMNIST_quant_state_dict.pt",
            "params": {"n_bits": 4},
        }
    },
    "CustomModel": {
        "quant": {
            "model_class": QuantCustomModel,
            "path": Path(__file__).parent / "custom_data_quant_state_dict.pt",
            "params": {"n_bits": 4, "input_shape": 6, "hidden_shape": 100, "output_shape": 3},
        },
        "fp32": {
            "model_class": TorchCustomModel,
            "path": Path(__file__).parent / "custom_data_fp32_state_dict.pt",
            "params": {"input_shape": 6, "hidden_shape": 100, "output_shape": 3},
        },
    },
}


@pytest.mark.parametrize("attr, value", [("undefined_attr", 1)])
@pytest.mark.parametrize("model_name, training", [("CustomModel", "quant")])
def test_update_unvalid_attr_method(attr, value, model_name, training, load_data):
    """Check if the _update_attr method raises an exception when an undefined attribute is given."""

    model = load_torch_model(
        MODELS_ARGS[model_name][training]["model_class"],
        MODELS_ARGS[model_name][training]["path"],
        MODELS_ARGS[model_name][training]["params"],
    )

    x, y = load_data(model, **DATASETS_ARGS[model_name]["param"])
    x_calib, y = data_calibration_processing(data=x, targets=y, n_sample=1)

    search = BinarySearch(estimator=model)

    with pytest.raises(AttributeError, match=".* does not belong to this class"):
        _ = search.run(x=x_calib, ground_truth=y, strategy=all, **{attr: value})


@pytest.mark.parametrize("attr, value", [("max_iter", 1)])
@pytest.mark.parametrize("model_name, training", [("CustomModel", "quant")])
def test_update_valid_attr_method(attr, value, model_name, training, load_data):
    """Check if the _update_attr method raises an exception when an undefined attribute is given."""

    model = load_torch_model(
        MODELS_ARGS[model_name][training]["model_class"],
        MODELS_ARGS[model_name][training]["path"],
        MODELS_ARGS[model_name][training]["params"],
    )

    x, y = load_data(model, **DATASETS_ARGS[model_name]["param"])
    x_calib, y = data_calibration_processing(data=x, targets=y, n_sample=1)

    search = BinarySearch(estimator=model)

    _ = search.run(x=x_calib, ground_truth=y, strategy=all, **{attr: value})


@pytest.mark.parametrize("max_metric_loss, max_iter, n_simulation", [(0.00001, 1, 1)])
@pytest.mark.parametrize("model_name, training", [("FashionMNIST", "quant")])
def test_non_convergence(model_name, training, max_metric_loss, max_iter, n_simulation):
    """Check that a binary search that does not converge raises a user warning."""

    dataset = get_torchvision_dataset(DATASETS_ARGS[model_name], train_set=True)
    x_calib, y = data_calibration_processing(dataset, n_sample=3)

    model = load_torch_model(
        MODELS_ARGS[model_name][training]["model_class"],
        MODELS_ARGS[model_name][training]["path"],
        MODELS_ARGS[model_name][training]["params"],
    )

    search = BinarySearch(
        estimator=model,
        max_metric_loss=max_metric_loss,
        max_iter=max_iter,
        n_simulation=n_simulation,
    )

    with pytest.warns(UserWarning, match="The convergence is not reached. .*"):
        _ = search.run(x=x_calib, ground_truth=y, strategy=all)


@pytest.mark.parametrize(
    "strategy",
    [
        all,
        any,
        lambda all_matches: numpy.mean(all_matches) >= 0.5,
        lambda all_matches: numpy.median(all_matches) == 1,
    ],
)
@pytest.mark.parametrize(
    "model_name, training", [("FashionMNIST", "quant"), ("CustomModel", "quant")]
)
def test_valid_strategy_method(strategy, model_name, training):
    """Check if the get_strategy method raises an exception when un undefined function is given."""

    model = load_torch_model(
        MODELS_ARGS[model_name][training]["model_class"],
        MODELS_ARGS[model_name][training]["path"],
        MODELS_ARGS[model_name][training]["params"],
    )

    search = BinarySearch(estimator=model)

    _ = search.eval_match(strategy, all_match=[True, True, False, True])


@pytest.mark.parametrize("strategy", ["undefined_function", lambda x, y: x + y])
@pytest.mark.parametrize(
    "model_name, training", [("FashionMNIST", "quant"), ("CustomModel", "quant")]
)
def test_invalid_strategy_method(strategy, model_name, training):
    """Check if the get_strategy method raises an exception when un undefined function is given."""

    model = load_torch_model(
        MODELS_ARGS[model_name][training]["model_class"],
        MODELS_ARGS[model_name][training]["path"],
        MODELS_ARGS[model_name][training]["params"],
    )

    search = BinarySearch(estimator=model)

    with pytest.raises(TypeError):
        _ = search.eval_match(strategy, all_match=[True, True, False, True])


@pytest.mark.parametrize("max_metric_loss", [0.01])
@pytest.mark.parametrize(
    "strategy_function",
    [
        all,
        lambda all_matches: numpy.mean(all_matches) >= 0.5,
        lambda all_matches: numpy.median(all_matches) == 1,
    ],
)
@pytest.mark.parametrize(
    "model_name, training", [("FashionMNIST", "quant"), ("CustomModel", "quant")]
)
def test_binary_search(model_name, training, strategy_function, max_metric_loss, load_data):
    """Check if the returned `p_error` is valid."""

    # Load pretrained NN
    model = load_torch_model(
        MODELS_ARGS[model_name][training]["model_class"],
        MODELS_ARGS[model_name][training]["path"],
        MODELS_ARGS[model_name][training]["params"],
    )

    # Load data-set
    if model_name == "FashionMNIST":
        pytest.skip(reason="This test is too long.")
    else:
        x, y = load_data(model, **DATASETS_ARGS[model_name]["param"])
        x_calib, y = data_calibration_processing(data=x, targets=y, n_sample=100)

    search = BinarySearch(
        estimator=model,
        save=True,
        verbose=True,
        n_simulation=5,
        max_metric_loss=max_metric_loss,
        delta_tolerence=1e-4,
    )

    largest_perror = search.run(x=x_calib, ground_truth=y, strategy=strategy_function, max_iter=100)

    assert 1.0 > largest_perror > 0.0
    assert round(search.history[-1]["accuracy_difference"], 5) <= max_metric_loss


@pytest.mark.parametrize(
    "is_qat, model_name, training", [(True, "CustomModel", "quant"), (False, "CustomModel", "fp32")]
)
def test_valid_estimator_with_compile_function(is_qat, model_name, training):
    """Check the supported models according our build functions."""

    # Load pretrained NN
    model = load_torch_model(
        MODELS_ARGS[model_name][training]["model_class"],
        MODELS_ARGS[model_name][training]["path"],
        MODELS_ARGS[model_name][training]["params"],
    )

    x, y = make_classification(**DATASETS_ARGS[model_name]["param"])

    x_calib, y = data_calibration_processing(data=x, targets=y, n_sample=100)

    search = BinarySearch(
        estimator=model,
        save=True,
        verbose=True,
        n_simulation=5,
        is_qat=is_qat,
        delta_tolerence=1e-4,
        max_metric_loss=0.1,
    )

    _ = search.run(x=x_calib, ground_truth=y, strategy=all, max_iter=1)


@pytest.mark.parametrize("is_qat", [False, True])
def test_invalid_estimator_with_compile_function(is_qat):
    """Check the supported models according our build functions."""

    model = keras.Sequential(
        [
            keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
            keras.layers.Dense(units=256, activation="relu"),
            keras.layers.Dense(units=192, activation="relu"),
            keras.layers.Dense(units=128, activation="relu"),
            keras.layers.Dense(units=10, activation="softmax"),
        ]
    )

    dataset = get_torchvision_dataset(DATASETS_ARGS["FashionMNIST"], train_set=True)
    x_calib, y = data_calibration_processing(dataset, n_sample=3)

    search = BinarySearch(
        estimator=model,
        save=True,
        verbose=True,
        n_simulation=5,
        is_qat=is_qat,
        delta_tolerence=1e-4,
    )

    with pytest.raises(ValueError, match=".* is not supported. .*"):
        _ = search.run(x=x_calib, ground_truth=y, strategy=all, max_iter=1)
