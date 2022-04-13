import os
import time
from functools import partial

import numpy as np
import py_progress_tracker as progress
from common import BENCHMARK_CONFIGURATION, run_and_report_regression_metrics
from sklearn.datasets import fetch_openml, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from concrete.ml.sklearn import LinearRegression, LinearSVR, NeuralNetRegressor

N_MAX_COMPILE_FHE = int(os.environ.get("N_MAX_COMPILE_FHE", 1000))
N_MAX_RUN_FHE = int(os.environ.get("N_MAX_RUN_FHE", 100))


openml_dataset_names = [
    "pol",
    "house_16H",
    "tecator",
    "boston",
    "socmob",
    "wine_quality",
    "abalone",
    "us_crime",
    "Brazilian_houses",
    "Moneyball",
    "SAT11-HAND-runtime-regression",
    "Santander_transaction_value",
    "house_prices_nominal",
    "Yolanda",
    "house_sales",
    "Buzzinsocialmedia_Twitter",
]

dataset_versions = {
    "abalone": 5,
    "us_crime": 2,
    "Brazilian_houses": 4,
    "Moneyball": 2,
    "Yolanda": 2,
    "quake": 2,
    "house_sales": 3,
}


def return_one_bias(i):
    bias_list = [0, 0.0001, 0.1, 100, 10000]
    return bias_list[i % len(bias_list)]


def return_one_noise(i):
    noise_list = [0, 0.001, 0.1, 10, 1000]
    return noise_list[i % len(noise_list)]


def return_one_n_targets(i):
    n_targets_list = [1, 2, 3, 4, 5]
    return n_targets_list[i % len(n_targets_list)]


def return_one_n_features(i):
    n_features_list = [1, 2, 5, 14, 100]
    return n_features_list[i % len(n_features_list)]


# Make the bias and noise not synchronized, ie not always the same bias with the same noise
# 3, 7 and 11 are prime with 5, which is the length of bias_list and noise_list
# Not all couples will be done, however
synthetic_dataset_params = [
    (
        42 + i,
        {
            "bias": return_one_bias(i),
            "noise": return_one_noise((i + 1) * 3),
            "n_targets": return_one_n_targets((i + 2) * 7),
            "n_features": return_one_n_features((i + 3) * 11),
        },
    )
    for i in range(10)
]


def make_synthetic_name(i, bias, noise, n_targets, n_features):
    return f"SyntheticReg_{i}_{bias}_{noise}_{n_targets}_{n_features}"


def prepare_openml_dataset(dataset_name):
    # Sometimes we want a specific version of a dataset, otherwise just get the 'active' one
    version = dataset_versions.get(dataset_name, "active")
    x_all, y_all = fetch_openml(
        name=dataset_name, version=version, as_frame=False, cache=True, return_X_y=True
    )
    if y_all.ndim == 1:
        y_all = np.expand_dims(y_all, 1)
    return x_all, y_all, None


synthetic_dataset_names = [make_synthetic_name(i, **p) for i, p in synthetic_dataset_params]

synthetic_datasets = {
    name: partial(make_regression, n_samples=200, coef=True, random_state=seed, **params)
    for name, (seed, params) in zip(synthetic_dataset_names, synthetic_dataset_params)
}

openml_datasets = {name: partial(prepare_openml_dataset, name) for name in openml_dataset_names}

datasets = {**openml_datasets, **synthetic_datasets}

# Will contain all the regressors we can support
regressors = [NeuralNetRegressor, LinearRegression, LinearSVR]

benchmark_params = {
    LinearRegression: [{"n_bits": n_bits} for n_bits in range(2, 11)],
    LinearSVR: [{"n_bits": n_bits} for n_bits in range(2, 11)],
    NeuralNetRegressor: [
        # An FHE compatible config
        {
            "module__n_layers": 3,
            "module__n_w_bits": 2,
            "module__n_a_bits": 2,
            "module__n_accum_bits": 7,
            "module__n_hidden_neurons_multiplier": 1,
            "max_epochs": 200,
            "verbose": 0,
            "lr": 0.001,
        }
    ]
    + [
        # Pruned configurations that have approx. the same number of active neurons as the
        # FHE compatible config. This evaluates the accuracy that can be attained
        # for different accumulator bitwidths
        {
            "module__n_layers": 3,
            "module__n_w_bits": n_b,
            "module__n_a_bits": n_b,
            "module__n_accum_bits": n_b_acc,
            "module__n_hidden_neurons_multiplier": 4,
            "max_epochs": 200,
            "verbose": 0,
            "lr": 0.001,
        }
        for (n_b, n_b_acc) in [
            (2, 7),
            (3, 11),
            (4, 12),
            (5, 14),
            (6, 16),
            (7, 18),
            (8, 20),
            (9, 22),
            (10, 24),
        ]
    ]
    + [
        # Configs with all neurons active, to evaluate the accuracy of quantization of weights
        # and biases only
        {
            "module__n_layers": 3,
            "module__n_w_bits": n_b,
            "module__n_a_bits": n_b,
            "module__n_accum_bits": 32,
            "module__n_hidden_neurons_multiplier": 4,
            "max_epochs": 200,
            "verbose": 0,
            "lr": 0.001,
        }
        for n_b in range(2, 10)
    ],
}


# pylint: disable=too-many-return-statements
def should_test_config_in_fhe(regressor, config, n_features):
    """Determine whether a benchmark config for a regressor should be tested in FHE"""

    assert config is not None

    # System override to disable FHE benchmarks (useful for debugging)
    if os.environ.get("BENCHMARK_NO_FHE", "0") == "1":
        return False

    if regressor is LinearRegression:

        if config["n_bits"] == 2 and n_features <= 14:
            return True

        if config["n_bits"] == 3 and n_features <= 2:
            return True

        return False

    if regressor is LinearSVR:

        if config["n_bits"] == 2 and n_features <= 14:
            return True

        if config["n_bits"] == 3 and n_features <= 2:
            return True

        return False
    if regressor is NeuralNetRegressor:
        # For NNs only 7 bit accumulators with few neurons should be compiled to FHE
        return (
            config["module__n_accum_bits"] <= 7
            and config["module__n_hidden_neurons_multiplier"] == 1
        )
    raise ValueError(f"Regressor {str(regressor)} configurations not yet setup for FHE")


def benchmark_generator():
    # Iterate over the dataset names and the dataset generator functions
    for dataset_str, dataset_fun in datasets.items():
        for regressor in regressors:
            for config in benchmark_params[regressor]:
                yield (dataset_str, dataset_fun, regressor, config)


def benchmark_name_generator(dataset_str, regressor, config, joiner):
    if regressor is LinearRegression:
        config_str = f"_{config['n_bits']}"
    elif regressor is LinearSVR:
        config_str = f"_{config['n_bits']}"
    elif regressor is NeuralNetRegressor:
        config_str = f"_{config['module__n_a_bits']}_{config['module__n_accum_bits']}"
    else:
        raise ValueError

    return regressor.__name__ + config_str + joiner + dataset_str


# We run all the regressor that we want to benchmark over all datasets listed
@progress.track(
    [
        {
            "id": benchmark_name_generator(dataset_str, regressor, config, "_"),
            "name": benchmark_name_generator(dataset_str, regressor, config, " on "),
            "parameters": {
                "regressor": regressor,
                "dataset_str": dataset_str,
                "dataset_fun": dataset_fun,
                "config": config,
            },
            "samples": 1,
        }
        for (dataset_str, dataset_fun, regressor, config) in benchmark_generator()
    ]
)
def main(regressor, dataset_str, dataset_fun, config):
    """
    Our regressor benchmark. Use some synthetic data to train a regressor model,
    then fit a model with sklearn. We quantize the sklearn model and compile it to FHE.
    We compute the training loss for the quantized and FHE models and compare them. We also
    predict on a test set and compare FHE results to predictions from the quantized model
    """
    X, y, _ = dataset_fun()

    if regressor is NeuralNetRegressor:
        # Cast to a type that works for both sklearn and Torch
        X = X.astype(np.float32)
        y = y.astype(np.float32)

    # Split it into train/test and sort the sets for nicer visualization
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    if regressor is NeuralNetRegressor:
        normalizer = StandardScaler()
        # Compute mean/stdev on training set and normalize both train and test sets with them
        x_train = normalizer.fit_transform(x_train)
        x_test = normalizer.transform(x_test)

        config["module__input_dim"] = x_train.shape[1]
        config["module__n_outputs"] = y_train.shape[1] if y_train.ndim == 2 else 1

    print("Dataset:", dataset_str)
    print("Config:", config)

    concrete_regressor = regressor(**config)

    # We call fit_benchmark to both fit our Concrete ML regressors but also to return the sklearn
    # one that we would use if we were not using FHE. This regressor will be our baseline
    concrete_regressor, sklearn_regressor = concrete_regressor.fit_benchmark(x_train, y_train)

    # Predict with the sklearn regressor and compute goodness of fit
    y_pred_sklearn = sklearn_regressor.predict(x_test)
    run_and_report_regression_metrics(y_test, y_pred_sklearn, "sklearn", "Sklearn")

    # Now predict with our regressor and report its goodness of fit
    y_pred_q = concrete_regressor.predict(x_test, execute_in_fhe=False)
    run_and_report_regression_metrics(y_test, y_pred_q, "quantized-clear", "Quantized Clear")

    n_features = X.shape[1] if X.ndim == 2 else 1

    if should_test_config_in_fhe(regressor, config, n_features):
        x_test_comp = x_test[0:N_MAX_COMPILE_FHE, :]

        # Compile and report compilation time
        t_start = time.time()
        concrete_regressor.compile(x_test_comp, compilation_configuration=BENCHMARK_CONFIGURATION)
        duration = time.time() - t_start
        progress.measure(id="fhe-compile-time", label="FHE Compile Time", value=duration)

        # To keep the test short and to fit in RAM we limit the number of test samples
        x_test = x_test[0:N_MAX_RUN_FHE, :]
        y_test = y_test[0:N_MAX_RUN_FHE]

        # Now predict with our regressor and report its goodness of fit. We also measure
        # execution time per test sample
        t_start = time.time()
        y_pred_c = concrete_regressor.predict(x_test, execute_in_fhe=True)
        duration = time.time() - t_start

        run_and_report_regression_metrics(y_test, y_pred_c, "fhe", "FHE")

        run_and_report_regression_metrics(
            y_test, y_pred_q[0:N_MAX_RUN_FHE], "quant-clear-fhe-set", "Quantized Clear on FHE set"
        )

        progress.measure(
            id="fhe-inference_time",
            label="FHE Inference Time per sample",
            value=duration / x_test.shape[0] if x_test.shape[0] > 0 else 0,
        )
