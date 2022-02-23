import os
import time

import py_progress_tracker as progress
from common import BENCHMARK_CONFIGURATION
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from concrete.ml.sklearn import LinearRegression as ConcreteLinearRegression

N_MAX_COMPILE_FHE = int(os.environ.get("N_MAX_COMPILE_FHE", 1000))
N_MAX_RUN_FHE = int(os.environ.get("N_MAX_RUN_FHE", 100))


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


# Make the bias and noise not synchronised, ie not always the same bias with the same noise
# 3, 7 and 11 are prime with 5, which is the length of bias_list and noise_list
# Not all couples will be done, however
list_params = [
    (
        42 + i,
        return_one_bias(i),
        return_one_noise((i + 1) * 3),
        return_one_n_targets((i + 2) * 7),
        return_one_n_features((i + 3) * 11),
    )
    for i in range(10)
]

datasets = {
    f"SyntheticReg_{i}_{bias}_{noise}_{n_targets}_{n_features}": make_regression(
        n_samples=200,
        n_features=n_features,
        n_targets=n_targets,
        bias=bias,
        noise=noise,
        random_state=random_state,
        coef=True,
    )
    for i, (random_state, bias, noise, n_targets, n_features) in enumerate(list_params)
}

# Will contain all the regressors we can support
regressors = [ConcreteLinearRegression]

benchmark_params = {
    ConcreteLinearRegression: [{"n_bits": n_bits} for n_bits in range(2, 11)],
}


def should_test_config_in_fhe(regressor, config, n_features):
    """Determine whether a benchmark config for a regressor should be tested in FHE"""

    assert config is not None

    # System override to disable FHE benchmarks (useful for debugging)
    if os.environ.get("BENCHMARK_NO_FHE", "0") == "1":
        return False

    if regressor is ConcreteLinearRegression:

        if config["n_bits"] == 2 and n_features <= 14:
            return True

        if config["n_bits"] == 3 and n_features <= 2:
            return True

        return False
    raise ValueError(f"Regressor {str(regressor)} configurations not yet setup for FHE")


def run_and_report_metric(y_gt, y_pred, metric, metric_id, metric_label):
    """Run a single metric and report results to progress tracker"""
    value = metric(y_gt, y_pred) if y_gt.size > 0 else 0
    progress.measure(
        id=metric_id,
        label=metric_label,
        value=value,
    )


def run_and_report_all_metrics(y_gt, y_pred, metric_id_prefix, metric_label_prefix):
    """Run several metrics and report results to progress tracker with computed name and id"""

    metric_info = [(r2_score, "r2_score", "R2Score"), (mean_squared_error, "MSE", "MSE")]
    for (metric, metric_id, metric_label) in metric_info:
        run_and_report_metric(
            y_gt,
            y_pred,
            metric,
            "_".join((metric_id_prefix, metric_id)),
            " ".join((metric_label_prefix, metric_label)),
        )


def benchmark_generator():
    for dataset_str, dataset_fun in datasets.items():
        for regressor in regressors:
            for config in benchmark_params[regressor]:
                yield (dataset_str, dataset_fun, regressor, config)


def benchmark_name_generator(dataset_str, regressor, config, joiner):
    if regressor is ConcreteLinearRegression:
        config_str = f"_{config['n_bits']}"
    else:
        raise ValueError

    return regressor.__name__ + config_str + joiner + dataset_str


# We run all the regressorthat we want to benchmark over all datasets listed
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
    X, y, _ = dataset_fun

    # Split it into train/test and sort the sets for nicer visualization
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    print("Dataset:", dataset_str)
    print("Config:", config)

    concrete_regressor = regressor(**config)

    # We call fit_benchmark to both fit our Concrete ML regressors but also to return the sklearn
    # one that we would use if we were not using FHE. This regressor will be our baseline
    concrete_regressor, sklearn_regressor = concrete_regressor.fit_benchmark(x_train, y_train)

    # Predict with the sklearn regressor and compute goodness of fit
    y_pred_sklearn = sklearn_regressor.predict(x_test)
    run_and_report_all_metrics(y_test, y_pred_sklearn, "sklearn", "Sklearn")

    # Now predict with our regressor and report its goodness of fit
    y_pred_q = concrete_regressor.predict(x_test, execute_in_fhe=False)
    run_and_report_all_metrics(y_test, y_pred_q, "quantized-clear", "Quantized Clear")

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

        run_and_report_all_metrics(y_test, y_pred_c, "fhe", "FHE")

        run_and_report_all_metrics(
            y_test, y_pred_q[0:N_MAX_RUN_FHE], "quant-clear-fhe-set", "Quantized Clear on FHE set"
        )

        progress.measure(
            id="fhe-inference_time",
            label="FHE Inference Time per sample",
            value=duration / x_test.shape[0] if x_test.shape[0] > 0 else 0,
        )
