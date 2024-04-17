import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas
import py_progress_tracker as progress
from common import (
    BENCHMARK_CONFIGURATION,
    GLMS_STRING_TO_CLASS,
    benchmark_name_generator,
    compute_number_of_components,
    run_and_report_regression_metrics,
    seed_everything,
)
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.metrics import mean_tweedie_deviance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    StandardScaler,
)


# pylint: disable=redefined-outer-name
def get_data() -> Tuple[pandas.DataFrame, str]:
    """Fetch, merge and clean the GLM data-set."""

    # Getting the original data-set containing the risk features
    # Link: https://www.openml.org/d/41214
    risks_data, _ = fetch_openml(
        data_id=41214, as_frame=True, cache=True, data_home="~/.cache/sklearn", return_X_y=True
    )

    # Getting the data-set containing claims amount
    # Link: https://www.openml.org/d/41215
    claims_data, _ = fetch_openml(
        data_id=41215, as_frame=True, cache=True, data_home="~/.cache/sklearn", return_X_y=True
    )

    # Set IDpol as index
    risks_data["IDpol"] = risks_data["IDpol"].astype(int)
    risks_data.set_index("IDpol", inplace=True)

    # Grouping claims mounts together if they are associated with the same policy
    claims_data = claims_data.groupby("IDpol").sum()

    # Merging the two sets over policy IDs
    data = risks_data.join(claims_data, how="left")

    # Only keeping part of the data for faster running time
    data = data.head(40000)

    # Filtering out unknown claim amounts
    data["ClaimAmount"].fillna(0, inplace=True)

    # Filtering out claims with zero amount, as the severity (gamma) model
    # requires strictly positive target values
    data.loc[(data["ClaimAmount"] == 0) & (data["ClaimNb"] >= 1), "ClaimNb"] = 0

    # Removing unreasonable outliers
    data["ClaimNb"] = data["ClaimNb"].clip(upper=4)
    data["Exposure"] = data["Exposure"].clip(upper=1)
    data["ClaimAmount"] = data["ClaimAmount"].clip(upper=200000)

    return data, "freMTPL2freq"


def create_target_values(data: pandas.DataFrame) -> None:
    """Create the targets values for each GLMs."""

    # Creating target values for Poisson
    data["Frequency"] = data["ClaimNb"] / data["Exposure"]

    # Creating target values for Gamma
    data["AvgClaimAmount"] = data["ClaimAmount"] / np.fmax(data["ClaimNb"], 1)

    # Creating target values for Tweedie
    # Insurances companies are interested in modeling the Pure Premium, that is the expected total
    # claim amount per unit of exposure for each policyholder in their portfolio
    data["PurePremium"] = data["ClaimAmount"] / data["Exposure"]


def get_preprocessor() -> ColumnTransformer:
    """Instantiate the data preprocessor."""
    log_scale_transformer = make_pipeline(
        FunctionTransformer(np.log, validate=False), StandardScaler()
    )

    preprocessor = ColumnTransformer(
        [
            ("passthrough_numeric", "passthrough", ["BonusMalus"]),
            ("binned_numeric", KBinsDiscretizer(n_bins=10), ["VehAge", "DrivAge"]),
            ("log_scaled_numeric", log_scale_transformer, ["Density"]),
            (
                "onehot_categorical",
                OneHotEncoder(sparse=False),
                ["VehBrand", "VehPower", "VehGas", "Region", "Area"],
            ),
        ],
        remainder="drop",
    )
    return preprocessor


def get_train_test_data(data: pandas.DataFrame) -> Tuple[pandas.DataFrame, pandas.DataFrame]:
    """Split the data into a train and test set."""
    (
        train_data,
        test_data,
    ) = train_test_split(
        data,
        test_size=0.2,
        random_state=0,
    )

    # The test set is reduced for faster FHE runs.
    (
        _,
        test_data,
    ) = train_test_split(
        test_data,
        test_size=500,
        random_state=0,
    )

    return train_data, test_data


def get_parameters_glms(config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    # Fetching the data and initializing the parameters
    data, dataset_name = get_data()
    create_target_values(data)
    train_data, test_data = get_train_test_data(data)

    # GammaRegressor only handles strictly positive target values.
    gamma_mask_train = train_data["ClaimAmount"] > 0
    gamma_mask_test = test_data["ClaimAmount"] > 0

    parameters_glms = {
        "PoissonRegressor": {
            "regressor": GLMS_STRING_TO_CLASS["PoissonRegressor"],
            "dataset_name": dataset_name,
            "init_parameters": {
                "alpha": config["PoissonRegressor"]["alpha"],
                "max_iter": config["PoissonRegressor"]["max_iter"],
            },
            "fit_parameters": {
                "X": train_data,
                "y": train_data["Frequency"],
                "regressor__sample_weight": train_data["Exposure"],
            },
            "x_test": test_data,
            "score_parameters": {
                "y_true": test_data["Frequency"],
                "sample_weight": test_data["Exposure"],
                "power": config["PoissonRegressor"]["power"],
            },
        },
        "GammaRegressor": {
            "regressor": GLMS_STRING_TO_CLASS["GammaRegressor"],
            "dataset_name": dataset_name,
            "init_parameters": {
                "alpha": config["GammaRegressor"]["alpha"],
                "max_iter": config["GammaRegressor"]["max_iter"],
            },
            "fit_parameters": {
                "X": train_data[gamma_mask_train],
                "y": train_data[gamma_mask_train]["AvgClaimAmount"],
                "regressor__sample_weight": train_data[gamma_mask_train]["ClaimNb"],
            },
            "x_test": test_data[gamma_mask_test],
            "score_parameters": {
                "y_true": test_data[gamma_mask_test]["AvgClaimAmount"],
                "sample_weight": test_data[gamma_mask_test]["ClaimNb"],
                "power": config["GammaRegressor"]["power"],
            },
        },
        "TweedieRegressor": {
            "regressor": GLMS_STRING_TO_CLASS["TweedieRegressor"],
            "dataset_name": dataset_name,
            "init_parameters": {
                "power": config["TweedieRegressor"]["power"],
                "alpha": config["TweedieRegressor"]["alpha"],
                "max_iter": config["TweedieRegressor"]["max_iter"],
            },
            "fit_parameters": {
                "X": train_data,
                "y": train_data["PurePremium"],
                "regressor__sample_weight": train_data["Exposure"],
            },
            "x_test": test_data,
            "score_parameters": {
                "y_true": test_data["PurePremium"],
                "sample_weight": test_data["Exposure"],
                "power": config["TweedieRegressor"]["power"],
            },
        },
    }

    return parameters_glms


def get_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Fix the GLM parameters used for initialization, fitting, prediction and score evaluation."""

    if args.n_bits is not None:
        n_bits_list = args.n_bits
    else:  # Default
        try:  # For backward compatibility
            # pylint: disable-next=unused-import,import-outside-toplevel
            # flake8: noqa: F401,C0415
            from concrete.ml.quantization.base_quantized_op import DEFAULT_MODEL_BITS

            n_bits_list = [
                {"model_inputs": 6, "op_inputs": 2, "op_weights": 2, "model_outputs": 6},
                {"model_inputs": 6, "op_inputs": 3, "op_weights": 2, "model_outputs": 6},
                {"model_inputs": 6, "op_inputs": 3, "op_weights": 3, "model_outputs": 6},
            ]
        except (ImportError, AttributeError):
            n_bits_list = [
                {"net_inputs": 6, "op_inputs": 2, "op_weights": 2, "net_outputs": 6},
                {"net_inputs": 6, "op_inputs": 3, "op_weights": 2, "net_outputs": 6},
                {"net_inputs": 6, "op_inputs": 3, "op_weights": 3, "net_outputs": 6},
            ]

    config = {
        "PoissonRegressor": {
            "n_bits_list": n_bits_list,
            "alpha": 1e-3,
            "max_iter": 400,
            "power": 1,
        },
        "GammaRegressor": {
            "n_bits_list": n_bits_list,
            "alpha": 1e-3,
            "max_iter": 300,
            "power": 2,
        },
        "TweedieRegressor": {
            "n_bits_list": n_bits_list,
            "alpha": 0.1,
            "max_iter": 1000,
            "power": 1.9,
        },
    }

    # Retrieving the number of bits to benchmark on
    if args.configs is not None:
        # The configuration parameters should be in the same order as the input regressor list
        assert len(args.configs) == len(args.regressors)
        for i, regressor in enumerate(args.regressors):
            config[regressor].update(args.configs[i])

    return config


def score_estimator(
    y_pred: np.ndarray,
    y_true: pandas.Series,
    sample_weight: pandas.Series,
    power: float,
) -> Optional[float]:
    """Evaluate the score of a GLM using its predictions."""

    # Ignore non-positive predictions, as they are invalid for the Tweedie deviance. We want to
    # issue a warning if for some reason (e.g., low quantization, user error), the regressor
    # predictions are negative.

    # Concrete predictions' shape is (n, 1) but mean_tweedie_deviance only accepts arrays
    # of shape (n,)
    y_pred = np.squeeze(y_pred, axis=1)

    # Find all strictly positive values
    mask = y_pred > 0

    # If any non-positive values are found, issue a warning
    number_of_negative_values = mask.shape[0] - mask.sum()

    if number_of_negative_values > 0:
        print(
            "WARNING: Estimator yields invalid, non-positive predictions "
            f"for {number_of_negative_values} samples out of {mask.shape[0]}. These predictions "
            "are ignored when computing the deviance score."
        )

    # If all or at least 50% of the y_preds are non-positive, set this benchmark as a failure.
    if y_pred[mask].shape == (0,) or number_of_negative_values / mask.shape[0] > 0.5:
        return None

    # mean_tweedie_deviance with power=1 is equivalent to mean_poisson_deviance
    # mean_tweedie_deviance with power=2 is equivalent to mean_gamma_deviance
    return mean_tweedie_deviance(
        y_true=y_true[mask],
        y_pred=y_pred[mask],
        sample_weight=sample_weight[mask],
        power=power,
    )


def argument_manager() -> argparse.Namespace:
    """Manage input arguments from the user."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--mlir_only", type=int, help="Only dump MLIR (no inference)")
    parser.add_argument("--verbose", action="store_true", help="show more information on stdio")
    parser.add_argument(
        "--seed",
        type=int,
        default=random.randint(0, 2**32 - 1),
        help="set the seed for reproducibility",
    )
    parser.add_argument(
        "--regressors",
        choices=GLMS_STRING_TO_CLASS.keys(),
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
        "--n_bits",
        nargs="+",
        type=json.loads,
        default=None,
        help="Bit-width config(s) to use",
    )
    parser.add_argument(
        "--model_samples",
        type=int,
        default=1,
        help="number of model samples (i.e., overwrite PROGRESS_SAMPLES)",
    )
    parser.add_argument(
        "--fhe_samples", type=int, default=1, help="number of FHE samples on which to predict"
    )
    parser.add_argument(
        "--long_list",
        action="store_true",
        help="just list the different tasks and stop",
    )
    parser.add_argument(
        "--short_list",
        action="store_true",
        help="just list the different tasks (one per model type) and stop",
    )

    args = parser.parse_args()

    if args.regressors is None:
        args.regressors = GLMS_STRING_TO_CLASS.keys()

    return args


def main():
    # Parameters from the user
    args = argument_manager()

    # Seed everything possible
    seed_everything(args.seed)

    config = get_config(args)

    # Generate list of commands
    if args.long_list or args.short_list:
        already_done_models = {}

        for regressor in args.regressors:
            for n_bits in config[regressor]["n_bits_list"]:
                print_configs = config[regressor].copy()
                print_configs.pop("n_bits_list")
                print_configs = json.dumps(print_configs).replace("'", '"')
                n_bits = str(n_bits).replace("'", '"')

                if not args.short_list or regressor not in already_done_models:
                    print(
                        f"--regressors {regressor} --n_bits '{n_bits}' "
                        f"--configs '{print_configs}' --fhe_samples {args.fhe_samples}"
                    )
                    already_done_models[regressor] = 1

        return

    number_of_test_cases = sum(
        len(config[regressor]["n_bits_list"]) for regressor in args.regressors
    )
    print(f"Will perform benchmarks on {number_of_test_cases} test cases")
    print(f"Using --seed {args.seed}")

    parameters_glms = get_parameters_glms(config)

    # pylint: disable=undefined-loop-variable, too-many-branches
    @progress.track(
        [
            {
                "id": benchmark_name_generator(
                    model=GLMS_STRING_TO_CLASS[regressor],
                    dataset_name=parameters_glms[regressor]["dataset_name"],
                    config={"n_bits": n_bits},
                ),
                "name": regressor + "_" + str(n_bits),
                "parameters": {"parameters": parameters_glms[regressor], "n_bits": n_bits},
                "samples": args.model_samples,
            }
            for regressor in args.regressors
            for n_bits in config[regressor]["n_bits_list"]
        ]
    )
    def perform_benchmark(parameters: Dict, n_bits: Union[Dict, int]) -> None:
        """
        This is our main benchmark function. It gets the data and trains the available GLM models
        in four different ways:
        - using scikit-learn's model, in clear
        - using scikit-learn's model on PCA reduced features, in clear
        - using Concrete ML's model on PCA reduced features, in clear
        - using Concrete ML's model on PCA reduced features, in FHE
        The corresponding deviance scores are then computed in order to determine the impact of
        quantization on the models' performances as well as to verify that executing them in FHE
        correctly gives the same outputs as its equivalent model compiled 'in clear'.
        """
        if args.verbose:
            print("Start")
            time_current = time.time()
        dataset_name = parameters["dataset_name"]

        ids_to_convert_in_seconds = []

        regressor = parameters["regressor"]
        init_parameters = parameters["init_parameters"]

        preprocessor = get_preprocessor()

        # Compute the maximum number of PCA components possible for executing the model in FHE
        n_components = compute_number_of_components(n_bits)

        # If the n_bits input is too high, the model could overflow the max precision bit-width
        # currently available
        if n_components == 0:
            raise ValueError(f"n_bits = {n_bits} is too high. Please lower its value(s).")

        # Instantiate the pipelines
        model_pca = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "pca",
                    PCA(
                        n_components=n_components,
                        whiten=True,
                        random_state=random.randint(0, 2**32 - 1),
                    ),
                ),
                ("regressor", regressor(n_bits=n_bits, **init_parameters)),
            ]
        )

        if args.verbose:
            print(f"  -- Done in {time.time() - time_current:.4f} seconds")
            time_current = time.time()
            print("Fit")

        # Fitting the models
        fit_parameters = parameters["fit_parameters"]
        model_pca.fit(**fit_parameters)

        # limit the number of test samples in order to fit in the RAM
        x_test = parameters["x_test"].head(args.fhe_samples)

        score_parameters = parameters["score_parameters"]
        y_test = score_parameters["y_true"].head(args.fhe_samples)

        # Compute the predictions for both sklearn and Concrete ML clear models
        if args.verbose:
            print(f"  -- Done in {time.time() - time_current} seconds")
            time_current = time.time()
            print("Predict with scikit-learn")

        x_test_sklearn = model_pca["pca"].transform(model_pca["preprocessor"].transform(x_test))
        sklearn_model_pca = model_pca["regressor"].sklearn_model  # pylint: disable=no-member
        sklearn_predictions = sklearn_model_pca.predict(x_test_sklearn)
        run_and_report_regression_metrics(y_test, sklearn_predictions, "sklearn", "Sklearn")

        if args.verbose:
            print(f"  -- Done in {time.time() - time_current:.4f} seconds")
            time_current = time.time()
            print("Predict in clear")

        predictions = model_pca.predict(x_test, fhe="disable")
        run_and_report_regression_metrics(y_test, predictions, "quantized-clear", "Quantized Clear")

        if args.verbose:
            print(f"  -- Done in {time.time() - time_current:.4f} seconds")
            time_current = time.time()
            print("Compile")

        # Compile the Concrete ML model for FHE, which needs some preprocessed data in order to run
        x_train_subset_pca = model_pca["pca"].transform(
            model_pca["preprocessor"].transform(fit_parameters["X"].head(100))
        )

        fhe_compile_time_id = "fhe-compile-time"
        ids_to_convert_in_seconds.append(fhe_compile_time_id)

        with progress.measure(id=fhe_compile_time_id, label="FHE Compile Time"):
            fhe_circuit = model_pca["regressor"].compile(  # pylint: disable=no-member
                x_train_subset_pca,
                configuration=BENCHMARK_CONFIGURATION,
                show_mlir=False,
            )

            # Dump MLIR
            if args.mlir_only:
                mlirs_dir: Path = Path(__file__).parents[1] / "MLIRs"
                benchmark_name = benchmark_name_generator(
                    dataset_name=dataset_name,
                    model=model_pca["regressor"],
                    config={"n_bits": n_bits},
                )
                mlirs_dir.mkdir(parents=True, exist_ok=True)
                with open(mlirs_dir / f"{benchmark_name}.mlir", "w", encoding="utf-8") as file:
                    file.write(fhe_circuit.mlir)
                return

        if args.verbose:
            print(f"  -- Done in {time.time() - time_current} seconds")
            time_current = time.time()
            print("Key generation")

        fhe_keygen_time_id = "fhe-keygen-time"
        ids_to_convert_in_seconds.append(fhe_keygen_time_id)

        with progress.measure(id=fhe_keygen_time_id, label="FHE Key Generation Time"):
            fhe_circuit.keygen()

        if args.verbose:
            print(f"  -- Done in {time.time() - time_current:.4f} seconds")
            time_current = time.time()
            print(f"Predict in FHE ({args.fhe_samples} samples)")

        # Compute the predictions in FHE and measure its running time (ms)
        fhe_inference_time_id = "fhe-inference_time"
        ids_to_convert_in_seconds.append(fhe_inference_time_id)

        with progress.measure(id=fhe_inference_time_id, label="FHE Inference Time per sample"):
            fhe_predictions = model_pca.predict(x_test, fhe="execute")

        run_and_report_regression_metrics(y_test, fhe_predictions, "fhe", "FHE")

        run_and_report_regression_metrics(
            y_test,
            predictions[0 : args.fhe_samples],
            "quant-clear-fhe-set",
            "Quantized Clear on FHE set",
        )

        # Modify the prediction running time to consider the number of samples
        if x_test.shape[0] > 0:
            progress.state.MEASUREMENTS[fhe_inference_time_id][-1] /= x_test.shape[0]
        else:
            progress.state.MEASUREMENTS[fhe_inference_time_id][-1] = 0

        if args.verbose:
            print(f"  -- Done in {time.time() - time_current:.4f} seconds")
            time_current = time.time()
            print("Computing scores")

        # Compute the deviance scores
        score_parameters["y_true"] = score_parameters["y_true"].head(args.fhe_samples)
        score_parameters["sample_weight"] = score_parameters["sample_weight"].head(args.fhe_samples)

        score = score_estimator(y_pred=predictions, **score_parameters)
        fhe_score = score_estimator(y_pred=fhe_predictions, **score_parameters)

        # In case all samples are invalid. This can happen with low bit quantization on a very few
        # FHE samples
        if score is None or fhe_score is None:
            raise ValueError(
                "Too many predictions were found set to 0, which is not supported for "
                f"computing {regressor.__name__}'s deviance score."
            )

        # Check what prediction performance we lose due to PCA
        if args.verbose:
            print("Evaluation in clear with PCA transformation (Concrete ML):", score)
            print("Evaluation in FHE with PCA transformation (Concrete ML):", fhe_score)

        progress.measure(
            id="quantized-deviance-score",
            label="Quantized Deviance Score",
            value=score,
        )

        progress.measure(
            id="homomorphic-deviance-score",
            label="Homomorphic Deviance Score",
            value=fhe_score,
        )

        # Convert time values from milliseconds to seconds
        for id_to_convert_in_seconds in ids_to_convert_in_seconds:
            progress.state.MEASUREMENTS[id_to_convert_in_seconds][-1] /= 1000

        if args.verbose:
            print(f"  -- Done in {time.time() - time_current:.4f} seconds")
            print("End")


if __name__ == "__main__":
    main()
