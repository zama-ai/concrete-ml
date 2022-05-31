import argparse
import json
import os
import random
import time

import numpy as np
import py_progress_tracker as progress
from common import BENCHMARK_CONFIGURATION, run_and_report_regression_metrics, seed_everything
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from concrete.ml.sklearn import LinearRegression, LinearSVR, NeuralNetRegressor

possible_datasets = [
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

# Will contain all the regressors we can support
possible_regressors = [NeuralNetRegressor, LinearRegression, LinearSVR]
regressors_string_to_class = {c.__name__: c for c in possible_regressors}

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
def should_test_config_in_fhe(regressor, config, n_features, local_args):
    """Determine whether a benchmark config for a regressor should be tested in FHE"""

    if local_args.execute_in_fhe != "auto":
        return local_args.execute_in_fhe

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


def benchmark_generator(local_args):
    # Iterate over the dataset names and the dataset generator functions
    for dataset in local_args.datasets:
        for regressor in local_args.regressors:
            if local_args.configs is None:
                for config in benchmark_params[regressor]:
                    yield (dataset, regressor, config)
            else:
                for config in local_args.configs:
                    yield (dataset, regressor, config)


def benchmark_name_generator(dataset, regressor, config, joiner):
    if regressor is LinearRegression:
        config_str = f"_{config['n_bits']}"
    elif regressor is LinearSVR:
        config_str = f"_{config['n_bits']}"
    elif regressor is NeuralNetRegressor:
        config_str = f"_{config['module__n_a_bits']}_{config['module__n_accum_bits']}"
    else:
        raise ValueError

    return regressor.__name__ + config_str + joiner + dataset


def train_and_test_on_dataset(regressor, dataset, config, local_args):

    # Could be changed but not very useful
    size_of_compilation_dataset = 1000

    if local_args.verbose:
        print("Start")
        time_current = time.time()

    version = dataset_versions.get(dataset, "active")
    X, y = fetch_openml(name=dataset, version=version, as_frame=False, cache=True, return_X_y=True)
    if y.ndim == 1:
        y = np.expand_dims(y, 1)

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

    concrete_regressor = regressor(**config)

    if local_args.verbose:
        print(f"  -- Done in {time.time() - time_current} seconds")
        time_current = time.time()
        print("Fit")

    # We call fit_benchmark to both fit our Concrete ML regressors but also to return the sklearn
    # one that we would use if we were not using FHE. This regressor will be our baseline
    concrete_regressor, sklearn_regressor = concrete_regressor.fit_benchmark(x_train, y_train)

    if local_args.verbose:
        print(f"  -- Done in {time.time() - time_current} seconds")
        time_current = time.time()
        print("Predict with scikit-learn")

    # Predict with the sklearn regressor and compute goodness of fit
    y_pred_sklearn = sklearn_regressor.predict(x_test)
    run_and_report_regression_metrics(y_test, y_pred_sklearn, "sklearn", "Sklearn")

    if local_args.verbose:
        print(f"  -- Done in {time.time() - time_current} seconds")
        time_current = time.time()
        print("Predict in clear")

    # Now predict with our regressor and report its goodness of fit
    y_pred_q = concrete_regressor.predict(x_test, execute_in_fhe=False)
    run_and_report_regression_metrics(y_test, y_pred_q, "quantized-clear", "Quantized Clear")

    n_features = X.shape[1] if X.ndim == 2 else 1

    if should_test_config_in_fhe(regressor, config, n_features, local_args):

        if local_args.verbose:
            print(f"  -- Done in {time.time() - time_current} seconds")
            time_current = time.time()
            print("Compile")

        x_test_comp = x_test[0:size_of_compilation_dataset, :]

        # Compile and report compilation time
        t_start = time.time()
        forward_fhe = concrete_regressor.compile(x_test_comp, configuration=BENCHMARK_CONFIGURATION)
        duration = time.time() - t_start
        progress.measure(id="fhe-compile-time", label="FHE Compile Time", value=duration)

        if local_args.verbose:
            print(f"  -- Done in {time.time() - time_current} seconds")
            time_current = time.time()
            print("Key generation")

        t_start = time.time()
        forward_fhe.keygen()
        duration = time.time() - t_start
        progress.measure(id="fhe-keygen-time", label="FHE Key Generation Time", value=duration)

        if local_args.verbose:
            print(f"  -- Done in {time.time() - time_current} seconds")
            time_current = time.time()
            print(f"Predict in FHE ({local_args.fhe_samples} samples)")

        # To keep the test short and to fit in RAM we limit the number of test samples
        x_test = x_test[0 : local_args.fhe_samples, :]
        y_test = y_test[0 : local_args.fhe_samples]

        # Now predict with our regressor and report its goodness of fit. We also measure
        # execution time per test sample
        t_start = time.time()
        y_pred_c = concrete_regressor.predict(x_test, execute_in_fhe=True)
        duration = time.time() - t_start

        run_and_report_regression_metrics(y_test, y_pred_c, "fhe", "FHE")

        run_and_report_regression_metrics(
            y_test,
            y_pred_q[0 : local_args.fhe_samples],
            "quant-clear-fhe-set",
            "Quantized Clear on FHE set",
        )

        progress.measure(
            id="fhe-inference_time",
            label="FHE Inference Time per sample",
            value=duration / x_test.shape[0] if x_test.shape[0] > 0 else 0,
        )

    if local_args.verbose:
        print(f"  -- Done in {time.time() - time_current} seconds")
        time_current = time.time()
        print("End")


def argument_manager():
    # Manage arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true", help="show more information on stdio")
    parser.add_argument(
        "--datasets",
        choices=possible_datasets,
        type=str,
        nargs="+",
        default=None,
        help="dataset(s) to use",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=random.randint(0, 2**32 - 1),
        help="set the seed for reproducibility",
    )
    parser.add_argument(
        "--regressors",
        choices=regressors_string_to_class.keys(),
        nargs="+",
        default=None,
        help="regressors(s) to use",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        type=json.loads,
        default=None,
        help="config(s) to use",
    )
    parser.add_argument(
        "--model_samples",
        type=int,
        default=1,
        help="number of model samples (ie, overwrite PROGRESS_SAMPLES)",
    )
    parser.add_argument(
        "--fhe_samples", type=int, default=1, help="number of FHE samples on which to predict"
    )
    parser.add_argument(
        "--execute_in_fhe",
        action="store_true",
        default="auto",
        help="force to execute in FHE (default is to use should_test_config_in_fhe function)",
    )
    parser.add_argument(
        "--dont_execute_in_fhe",
        action="store_true",
        help="force to not execute in FHE (default is to use should_test_config_in_fhe function)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="just list the different tasks and stop",
    )

    args = parser.parse_args()

    if args.dont_execute_in_fhe:
        assert args.execute_in_fhe == "auto"
        args.execute_in_fhe = False

    if args.datasets is None:
        args.datasets = possible_datasets
    if args.regressors is None:
        args.regressors = possible_regressors
    else:
        args.regressors = [regressors_string_to_class[c] for c in args.regressors]

    return args


def main():

    # Parameters by the user
    args = argument_manager()

    # Seed everything we can
    seed_everything(args.seed)

    all_tasks = list(benchmark_generator(args))

    if args.list:
        print("\nList of equivalent individual calls:\n")

        for (dataset_i, regressor_i, config_i) in all_tasks:
            config_n = str(config_i).replace("'", '"')
            print(
                f"--regressors {regressor_i.__name__} --datasets {dataset_i}"
                f" --configs '{config_n}'"
            )
        return

    print(f"Will perform benchmarks on {len(list(all_tasks))} test cases")
    print(f"Using --seed {args.seed}")

    # We run all the regressor that we want to benchmark over all datasets listed
    @progress.track(
        [
            {
                "id": benchmark_name_generator(dataset, regressor, config, "_"),
                "name": benchmark_name_generator(dataset, regressor, config, " on "),
                "parameters": {"regressor": regressor, "dataset": dataset, "config": config},
                "samples": args.model_samples,
            }
            for (dataset, regressor, config) in all_tasks
        ]
    )
    def perform_benchmark(regressor, dataset, config):
        """
        Our regressor benchmark. Use some synthetic data to train a regressor model,
        then fit a model with sklearn. We quantize the sklearn model and compile it to FHE.
        We compute the training loss for the quantized and FHE models and compare them. We also
        predict on a test set and compare FHE results to predictions from the quantized model
        """
        train_and_test_on_dataset(regressor, dataset, config, args)


if __name__ == "__main__":
    main()
