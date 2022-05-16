import math
from typing import Dict, Tuple, Union

import numpy as np
import pandas
import py_progress_tracker as progress
from common import BENCHMARK_CONFIGURATION
from concrete.numpy import MAXIMUM_BIT_WIDTH
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.linear_model import GammaRegressor as SklearnGammaRegressor
from sklearn.linear_model import PoissonRegressor as SklearnPoissonRegressor
from sklearn.linear_model import TweedieRegressor as SklearnTweedieRegressor
from sklearn.metrics import mean_tweedie_deviance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    StandardScaler,
)

from concrete.ml.sklearn import GammaRegressor as ConcreteGammaRegressor
from concrete.ml.sklearn import PoissonRegressor as ConcretePoissonRegressor
from concrete.ml.sklearn import TweedieRegressor as ConcreteTweedieRegressor


# pylint: disable=redefined-outer-name
def get_data() -> pandas.DataFrame:
    """Fetch, merge and clean the desired datasets."""

    # Getting the original data set containing the risk features
    # Link: https://www.openml.org/d/41214
    risks_data, _ = fetch_openml(
        data_id=41214, as_frame=True, cache=True, data_home="~/.cache/sklean", return_X_y=True
    )

    # Getting the data set containing claims amount
    # Link: https://www.openml.org/d/41215
    claims_data, _ = fetch_openml(
        data_id=41215, as_frame=True, cache=True, data_home="~/.cache/sklean", return_X_y=True
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

    return data


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
    train_data, test_data, = train_test_split(
        data,
        test_size=0.2,
        random_state=0,
    )

    # The test set is reduced for faster FHE runs.
    _, test_data, = train_test_split(
        test_data,
        test_size=500,
        random_state=0,
    )

    return train_data, test_data


def get_config() -> Tuple[list, Dict]:
    """Fix the GLM parameters used for initialization, fitting, prediction and score evaluation."""

    n_bits_to_tests = [
        {"inputs": 2, "weights": 2, "outputs": 8},
        {"inputs": 3, "weights": 2, "outputs": 8},
        {"inputs": 3, "weights": 3, "outputs": 8},
    ]

    # Fetching the data and initializing the parameters
    data = get_data()
    create_target_values(data)
    train_data, test_data = get_train_test_data(data)

    # GammaRegressor only handles stricly positive target values.
    gamma_mask_train = train_data["ClaimAmount"] > 0
    gamma_mask_test = test_data["ClaimAmount"] > 0

    parameters_glms = {
        "Poisson": {
            "sklearn": SklearnPoissonRegressor,
            "concrete": ConcretePoissonRegressor,
            "init_parameters": {
                "alpha": 1e-3,
                "max_iter": 400,
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
                "power": 1,
            },
        },
        "Gamma": {
            "sklearn": SklearnGammaRegressor,
            "concrete": ConcreteGammaRegressor,
            "init_parameters": {
                "alpha": 1e-3,
                "max_iter": 300,
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
                "power": 2,
            },
        },
        "Tweedie": {
            "sklearn": SklearnTweedieRegressor,
            "concrete": ConcreteTweedieRegressor,
            "init_parameters": {
                "power": 1.9,
                "alpha": 0.1,
                "max_iter": 1000,
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
                "power": 1.9,
            },
        },
    }
    return n_bits_to_tests, parameters_glms


def compute_number_of_components(n_bits: Union[Dict, int]) -> int:
    """Computes the maximum number of PCA components possible for executing a model in FHE."""
    if isinstance(n_bits, int):
        n_bits_inputs = n_bits
        n_bits_weights = n_bits
    else:
        n_bits_inputs = n_bits["inputs"]
        n_bits_weights = n_bits["weights"]

    n_components = math.floor(
        (2**MAXIMUM_BIT_WIDTH - 1) / ((2**n_bits_inputs - 1) * (2**n_bits_weights - 1))
    )
    return n_components


def score_estimator(
    y_pred: np.ndarray,
    y_true: pandas.Series,
    sample_weight: pandas.Series,
    power: float,
) -> float:
    """Evaluate the score of a GLM using its predictions."""

    # Ignore non-positive predictions, as they are invalid for the Tweedie deviance. We want to
    # issue a warning if for some reason (e.g. low quantization, user error), the regressor
    # predictions are negative.

    # Concrete predictions' shape is (n, 1) but mean_tweedie_deviance only accepts arrays
    # of shape (n,)
    y_pred = np.squeeze(y_pred)

    # Find all strictly positive values
    mask = y_pred > 0

    # If any non-positive values are found, issue a warning
    number_of_negative_values = mask.shape[0] - mask.sum()
    if number_of_negative_values > 0:
        print(
            "WARNING: Estimator yields invalid, non-positive predictions "
            f"for {number_of_negative_values} samples out of {mask.shape[0]}. These predictions "
            "are ignored when computing the Poisson deviance."
        )

    # mean_tweedie_deviance with power=1 is equivalent to mean_poisson_deviance
    # mean_tweedie_deviance with power=2 is equivalent to mean_gamma_deviance
    return mean_tweedie_deviance(
        y_true=y_true[mask],
        y_pred=y_pred[mask],
        sample_weight=sample_weight[mask],
        power=power,
    )


def get_benchmark_id(n_bits) -> str:
    """Create the suffix for benchmark ids."""
    if isinstance(n_bits, int):
        n_bits_inputs = n_bits
        n_bits_weights = n_bits
    else:
        n_bits_inputs = n_bits["inputs"]
        n_bits_weights = n_bits["weights"]

    pca_n_components = compute_number_of_components(n_bits)

    return (
        f"Regressor_n_bits_i_{n_bits_inputs}_n_bits_w_{n_bits_weights}_"
        f"n_compotent_{pca_n_components}"
    )


n_bits_to_tests, parameters_glms = get_config()


@progress.track(
    [
        {
            "id": glm + get_benchmark_id(n_bits),
            "name": glm + " Regressor",
            "parameters": {"glm": glm, "parameters": parameters, "n_bits": n_bits},
            "samples": 1,
        }
        for glm, parameters in parameters_glms.items()
        for n_bits in n_bits_to_tests
    ]
)
def main(glm: str, parameters: Dict, n_bits: Union[Dict, int]) -> None:
    """
    This is our main benchmark function. It gets the datas and trains the available GLM models in
    four different ways:
    - using scikit-learn's model, in clear
    - using scikit-learn's model on PCA reduced features, in clear
    - using Concrete-ML's model on PCA reduced features, in clear
    - using Concrete-ML's model on PCA reduced features, in FHE
    The corresponding deviance scores are then computed in order to determine the impact of
    quantization on the models' performances as well as to verify that executing them in FHE
    correctly gives the same outputs as its equivalent model compiled 'in clear'.
    """
    concrete_regressor = parameters["concrete"]
    init_parameters = parameters["init_parameters"]

    preprocessor = get_preprocessor()

    # Compute the maximum number of PCA components possible for executing the model in FHE
    n_components = compute_number_of_components(n_bits)

    # Let's instantiate the pipelines
    concrete_model_pca = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("pca", PCA(n_components=n_components, whiten=True)),
            ("regressor", concrete_regressor(n_bits=n_bits, **init_parameters)),
        ]
    )

    # Fitting the models
    fit_parameters = parameters["fit_parameters"]
    concrete_model_pca.fit(**fit_parameters)

    # Compute the predictions
    x_test = parameters["x_test"]
    concrete_predictions = concrete_model_pca.predict(x_test, execute_in_fhe=False)

    # Compile the Concrete-ML model for FHE, which needs some preprocessed data in order to run
    x_train_subset_pca = concrete_model_pca["pca"].transform(
        concrete_model_pca["preprocessor"].transform(fit_parameters["X"].head(100))
    )
    concrete_model_pca["regressor"].compile(  # pylint: disable=no-member
        x_train_subset_pca,
        use_virtual_lib=False,
        configuration=BENCHMARK_CONFIGURATION,
        show_mlir=False,
    )

    # Compute the predictions in FHE and measure its running time (ms)
    measure_id = "fhe-predict-evaluation-time-ms"
    with progress.measure(id=measure_id, label="FHE Inference Time (ms)"):
        concrete_fhe_predictions = concrete_model_pca.predict(x_test, execute_in_fhe=True)

    # Add a measurement of running time (ms) per sample
    progress.measure(
        id="fhe-inference-time-ms-per-sample",
        label="FHE Inference Time per sample (ms)",
        value=(progress.state.MEASUREMENTS[measure_id][-1] / x_test.shape[0])
        if x_test.shape[0] > 0
        else 0,
    )

    # Compute the deviance scores
    score_parameters = parameters["score_parameters"]
    concrete_score = score_estimator(y_pred=concrete_predictions, **score_parameters)
    concrete_fhe_score = score_estimator(y_pred=concrete_fhe_predictions, **score_parameters)

    # Let's check what prediction performance we lose due to PCA
    print(glm, ":")
    print("Evaluation in clear with PCA transformation (Concrete-ML):", concrete_score)
    progress.measure(
        id="non-homomorphic-deviance-score",
        label="Non Homomorphic Deviance Score",
        value=concrete_score,
    )

    print("Evaluation in FHE with PCA transformation (Concrete-ML):", concrete_fhe_score)
    progress.measure(
        id="homomorphic-deviance-score",
        label="Homomorphic Deviance Score",
        value=concrete_fhe_score,
    )

    # Let's make sure both models executed in clear and FHE output the same deviance score
    if concrete_fhe_score > 0.001:
        score_difference = abs(concrete_score - concrete_fhe_score) * 100 / concrete_fhe_score
    else:
        score_difference = 0

    print(f"Percentage difference: {score_difference}%")
    progress.measure(
        id="relative-deviance-score-difference-percent",
        label="Relative Deviance Score Difference (%)",
        value=score_difference,
        alert=(">", 7.5),
    )
