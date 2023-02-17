"""Tests for the FHE sklearn compatible NNs."""
from copy import deepcopy
from itertools import product

import brevitas.nn as qnn
import numpy
import pandas
import pytest
import torch
from sklearn.base import is_classifier, is_regressor
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from skorch.classifier import NeuralNetClassifier as SKNeuralNetClassifier
from skorch.exceptions import NotInitializedError
from torch import nn

from concrete.ml.common.utils import (
    MAX_BITWIDTH_BACKWARD_COMPATIBLE,
    check_dtype_and_cast,
    get_model_name,
)
from concrete.ml.sklearn.base import get_sklearn_neural_net_models
from concrete.ml.sklearn.qnn import (
    NeuralNetClassifier,
    NeuralNetRegressor,
    QuantizedSkorchEstimatorMixin,
    SparseQuantNeuralNetImpl,
)


@pytest.mark.parametrize("model", get_sklearn_neural_net_models())
def test_parameter_validation(model, load_data):
    """Test that the sklearn quantized NN wrappers validate their parameters"""

    valid_params = {
        "module__n_layers": 3,
        "module__n_w_bits": 2,
        "module__n_a_bits": 2,
        "module__n_accum_bits": MAX_BITWIDTH_BACKWARD_COMPATIBLE,
        "module__activation_function": nn.ReLU,
        "max_epochs": 10,
        "verbose": 0,
    }

    # Get the dataset. The data generation is seeded in load_data.
    if is_classifier(model):
        x, y = load_data(
            dataset="classification",
            n_samples=1000,
            n_features=10,
            n_redundant=0,
            n_repeated=0,
            n_informative=10,
            n_classes=2,
            class_sep=2,
        )

        valid_params["criterion__weight"] = [1, 1]

    # Get the dataset. The data generation is seeded in load_data.
    elif is_regressor(model):
        x, y, _ = load_data(
            dataset="regression",
            n_samples=1000,
            n_features=10,
            n_informative=10,
            noise=2,
            coef=True,
        )
    else:
        raise ValueError(f"Data generator not implemented for {str(model)}")

    invalid_params_and_exception_pattern = {
        "init": [
            ("module__n_outputs", 0, ".*module__n_outputs.*"),
            ("module__input_dim", 0, ".*module__input_dim.*"),
        ],
        "fit": [
            ("module__n_layers", 0, ".* number of layers.*"),
            ("module__n_w_bits", 0, ".* quantization bitwidth.*"),
            ("module__n_a_bits", 0, ".* quantization bitwidth.*"),
            ("module__n_accum_bits", 0, ".* accumulator bitwidth.*"),
        ],
    }
    for inv_param in invalid_params_and_exception_pattern["init"]:
        params = deepcopy(valid_params)
        params[inv_param[0]] = inv_param[1]

        with pytest.raises(
            ValueError,
            match=inv_param[2],
        ):
            concrete_classifier = model(**params)

    for inv_param in invalid_params_and_exception_pattern["fit"]:
        params = deepcopy(valid_params)
        params[inv_param[0]] = inv_param[1]

        concrete_classifier = model(**params)

        with pytest.raises(
            NotInitializedError,
            match=".* Call 'initialize' or 'fit' with appropriate arguments .*",
        ):
            _ = concrete_classifier.n_bits_quant

        with pytest.raises(
            ValueError,
            match=inv_param[2],
        ):
            concrete_classifier.fit(x, y)


@pytest.mark.parametrize("use_virtual_lib", [True, False])
@pytest.mark.parametrize(
    "activation_function",
    [
        pytest.param(nn.ReLU),
        pytest.param(nn.Sigmoid),
        pytest.param(nn.SELU),
        pytest.param(nn.CELU),
    ],
)
@pytest.mark.parametrize("model", get_sklearn_neural_net_models())
def test_compile_and_calib(
    activation_function, model, load_data, default_configuration, use_virtual_lib, is_vl_only_option
):
    """Test whether the sklearn quantized NN wrappers compile to FHE and execute well on encrypted
    inputs"""
    if not use_virtual_lib and is_vl_only_option:
        print("Warning, skipping non VL tests")
        return

    n_features = 10

    # Get the dataset. The data generation is seeded in load_data.
    if is_classifier(model):
        x, y = load_data(
            dataset="classification",
            n_samples=1000,
            n_features=n_features,
            n_redundant=0,
            n_repeated=0,
            n_informative=n_features,
            n_classes=2,
            class_sep=2,
        )

        # Add an offset to the labels to check that it is supported
        y += 10

    # Get the dataset. The data generation is seeded in load_data.
    elif is_regressor(model):
        x, y, _ = load_data(
            dataset="regression",
            n_samples=1000,
            n_features=n_features,
            n_informative=n_features,
            n_targets=2,
            noise=2,
            coef=True,
        )
        if y.ndim == 1:
            y = numpy.expand_dims(y, 1)
    else:
        raise ValueError(f"Data generator not implemented for {str(model)}")

    # Perform a classic test-train split (deterministic by fixing the seed)
    x_train, x_test, y_train, _ = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=numpy.random.randint(0, 2**15),
    )

    # Compute mean/stdev on training set and normalize both train and test sets with them
    # Optimization algorithms for Neural networks work well on 0-centered inputs
    normalizer = StandardScaler()
    x_train = normalizer.fit_transform(x_train)
    x_test = normalizer.transform(x_test)

    # Setup dummy class weights that will be converted to a tensor
    class_weights = numpy.asarray([1, 1]).reshape((-1,))

    # Configure a minimal neural network and train it quickly
    params = {
        "module__n_layers": 2,
        "module__n_w_bits": 2,
        "module__n_a_bits": 2,
        "module__n_accum_bits": 5,
        "module__activation_function": activation_function,
        "max_epochs": 10,
        "verbose": 0,
    }

    if is_classifier(model):
        params["criterion__weight"] = class_weights

    clf = model(**params)

    # Train the model
    # Needed for coverage
    if is_regressor(model):
        for x_d_type, y_d_type in product(
            [numpy.float32, numpy.float64], [numpy.float32, numpy.float64]
        ):
            clf.fit(x_train.astype(x_d_type), y_train.astype(y_d_type))
    elif is_classifier(model):
        for x_d_type, y_d_type in product(
            [numpy.float32, numpy.float64], [numpy.int32, numpy.int64]
        ):
            clf.fit(x_train.astype(x_d_type), y_train.astype(y_d_type))

    # Train normally
    clf.fit(x_train, y_train)

    if is_classifier(model):
        y_pred_clear = clf.predict(x_train, execute_in_fhe=False)
        # Check that the predicted classes are all contained in the model class list
        assert set(numpy.unique(y_pred_clear)).issubset(set(clf.classes_))

    # Compile the model
    clf.compile(
        x_train,
        configuration=default_configuration,
        use_virtual_lib=use_virtual_lib,
    )

    # Execute in FHE, but don't check the value.
    # Since FHE execution introduces some stochastic errors,
    # accuracy of FHE compiled classifiers and regressors is measured in the benchmarks
    clf.predict(x_test[0, :], execute_in_fhe=True)


def test_custom_net_classifier(load_data):
    """Tests a wrapped custom network.

    Gives an example how to use our API to train a custom Torch network through the quantized
    sklearn wrapper.
    """

    class MiniNet(nn.Module):
        """Sparse Quantized Neural Network classifier."""

        def __init__(
            self,
        ):
            """Construct mini net"""
            super().__init__()
            self.features = nn.Sequential(
                qnn.QuantIdentity(return_quant_tensor=True),
                qnn.QuantLinear(2, 4, weight_bit_width=3, bias=True),
                nn.ReLU(),
                qnn.QuantIdentity(return_quant_tensor=True),
                qnn.QuantLinear(4, 2, weight_bit_width=3, bias=True),
            )

        def forward(self, x):
            """Forward pass."""
            return self.features(x)

    params = {
        "max_epochs": 10,
        "verbose": 0,
    }

    class MiniCustomNeuralNetClassifier(QuantizedSkorchEstimatorMixin, SKNeuralNetClassifier):
        """Sklearn API wrapper class for a custom network that will be quantized.

        Minimal work is needed to implement training of a custom class."""

        def __init__(self, *args, **kwargs):
            super().__init__()
            SKNeuralNetClassifier.__init__(self, *args, **kwargs)

        @property
        def base_estimator_type(self):
            return SKNeuralNetClassifier

        @property
        def n_bits_quant(self):
            """Return the number of quantization bits"""
            return 2

        def fit(self, X, y, **fit_params):
            # We probably can't handle all cases since per Skorch documentation they handle:
            #  * numpy arrays
            #  * torch tensors
            #  * pandas DataFrame or Series
            #  * scipy sparse CSR matrices
            #  * a dictionary of the former three
            #  * a list/tuple of the former three
            #  * a Dataset
            # which is a bit much since they don't necessarily
            # have the same interfaces to handle types
            X = check_dtype_and_cast(
                X, "float32", error_information="MiniCustomNeuralNetClassifier input"
            )
            y = check_dtype_and_cast(
                y, "int64", error_information="MiniCustomNeuralNetClassifier target"
            )
            return super().fit(X, y, **fit_params)

        def predict(self, X, execute_in_fhe=False):
            # We just need to do argmax on the predicted probabilities
            return self.predict_proba(X, execute_in_fhe=execute_in_fhe).argmax(axis=1)

    clf = MiniCustomNeuralNetClassifier(MiniNet, **params)

    # Get the dataset. The data generation is seeded in load_data.
    x, y = load_data(
        dataset="classification",
        n_samples=1000,
        n_features=2,
        n_redundant=0,
        n_repeated=0,
        n_informative=2,
        n_classes=2,
        class_sep=2,
    )

    # Perform a classic test-train split (deterministic by fixing the seed)
    x_train, x_test, y_train, _ = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=numpy.random.randint(0, 2**15),
    )

    # Compute mean/stdev on training set and normalize both train and test sets with them
    # Optimization algorithms for Neural networks work well on 0-centered inputs
    normalizer = StandardScaler()
    x_train = normalizer.fit_transform(x_train)
    x_test = normalizer.transform(x_test)

    # Train the model
    clf.fit(x_train, y_train)

    # Test the custom network wrapper in a pipeline with grid CV
    # This will clone the skorch estimator
    pipe_cv = Pipeline(
        [
            ("pca", PCA(n_components=2, random_state=numpy.random.randint(0, 2**15))),
            ("scaler", StandardScaler()),
            ("net", MiniCustomNeuralNetClassifier(MiniNet, **params)),
        ]
    )

    clf = GridSearchCV(
        pipe_cv,
        {"net__lr": [0.01, 0.1]},
        error_score="raise",
    )
    clf.fit(x_train, y_train)


@pytest.mark.parametrize(
    "model_classes, bad_types, expected_error",
    [
        pytest.param(
            [NeuralNetClassifier],
            ("float32", "uint8"),
            "Neural Network classifier target dtype .* int64 .*",
            id="NeuralNetClassifier-target-uint8",
        ),
        pytest.param(
            [NeuralNetRegressor],
            ("float64", "complex64"),
            "Neural Network regressor target dtype .* float32 .*",
            id="NeuralNetRegressor-target-complex64",
        ),
        pytest.param(
            get_sklearn_neural_net_models(),
            ("complex64", "int64"),
            "Neural Network .* input dtype .* float32 .*",
            id="NeuralNets-input-complex64",
        ),
        pytest.param(
            get_sklearn_neural_net_models(),
            ("int64", "int64"),
            "Neural Network .* input dtype .* float32 .*",
            id="NeuralNets-input-int64",
        ),
        pytest.param(
            get_sklearn_neural_net_models(),
            ("str", "int64"),
            "Neural Network .* input dtype .* float32 .*",
            id="NeuralNets-input-str",
        ),
        pytest.param(
            get_sklearn_neural_net_models(),
            ("float32", "str"),
            "Neural Network .* target dtype .* (float32|int64) .*",
            id="NeuralNets-target-str",
        ),
    ],
)
@pytest.mark.parametrize("container", ["numpy", "pandas", "torch", "list"])
def test_failure_bad_data_types(model_classes, container, bad_types, expected_error, load_data):
    """Check that training using data with unsupported dtypes raises an expected error."""
    for model_class in model_classes:
        # Generate the data
        model_name = get_model_name(model_class)
        dataset = "regression" if is_regressor(model_class) else "classification"
        x, y = load_data(dataset, model_name=model_name)

        x, y = x.astype(bad_types[0]), y.astype(bad_types[1])

        if container == "torch":
            # Convert input and target to Torch tensors if the values are not string, as torch
            # Tensors only handles numerical values
            if "str" not in bad_types:
                x, y = torch.tensor(x), torch.tensor(y)

        elif container == "pandas":
            # Convert input to a Pandas DataFrame or Pandas Series
            x, y = pandas.DataFrame(x), pandas.DataFrame(y)

        # Instantiate the model
        model = model_class()

        # Train the model, which should raise the expected error
        with pytest.raises(ValueError, match=expected_error):
            model.fit(x, y)


@pytest.mark.parametrize("activation_function", [pytest.param(nn.ReLU)])
@pytest.mark.parametrize("model", get_sklearn_neural_net_models())
def test_structured_pruning(activation_function, model, load_data, default_configuration):
    """Test whether the sklearn quantized NN wrappers compile to FHE and execute well on encrypted
    inputs"""
    n_features = 10

    # Get the dataset. The data generation is seeded in load_data.
    if is_classifier(model):
        x, y = load_data(
            dataset="classification",
            n_samples=1000,
            n_features=n_features,
            n_redundant=0,
            n_repeated=0,
            n_informative=n_features,
            n_classes=2,
            class_sep=2,
        )

        # Add an offset to the labels to check that it is supported
        y += 10

    # Get the dataset. The data generation is seeded in load_data.
    elif is_regressor(model):
        x, y, _ = load_data(
            dataset="regression",
            n_samples=1000,
            n_features=n_features,
            n_informative=n_features,
            n_targets=2,
            noise=2,
            coef=True,
        )
        if y.ndim == 1:
            y = numpy.expand_dims(y, 1)
    else:
        raise ValueError(f"Data generator not implemented for {str(model)}")

    # Perform a classic test-train split (deterministic by fixing the seed)
    x_train, x_test, y_train, _ = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=numpy.random.randint(0, 2**15),
    )

    # Compute mean/stdev on training set and normalize both train and test sets with them
    # Optimization algorithms for Neural networks work well on 0-centered inputs
    normalizer = StandardScaler()
    x_train = normalizer.fit_transform(x_train)
    x_test = normalizer.transform(x_test)

    # Setup dummy class weights that will be converted to a tensor
    class_weights = numpy.asarray([1, 1]).reshape((-1,))

    # Configure a minimal neural network and train it quickly
    params = {
        "module__n_layers": 2,
        "module__n_w_bits": 2,
        "module__n_a_bits": 2,
        "module__n_accum_bits": 5,
        "module__activation_function": activation_function,
        "max_epochs": 10,
        "verbose": 0,
    }

    if is_classifier(model):
        params["criterion__weight"] = class_weights

    clf = model(**params)

    with pytest.raises(ValueError, match=""):
        clf.prune(x_train, y_train, 0.5)

    clf.fit(x_train, y_train)

    with pytest.raises(ValueError, match=".*Valid values.*"):
        clf.prune(x_train, y_train, -0.1)

    with pytest.raises(ValueError, match=".*Valid values.*"):
        clf.prune(x_train, y_train, 1.0)

    pruned_clf = clf.prune(x_train, y_train, 0.5)

    def _get_number_of_neurons(module: SparseQuantNeuralNetImpl):
        neurons = {}
        idx = 0
        for layer in module.features:
            if not isinstance(layer, nn.Linear) and not isinstance(layer, qnn.QuantLinear):
                continue
            neurons[idx] = layer.weight.shape[0]
            idx += 1
        return neurons

    neurons_orig = _get_number_of_neurons(clf.module_)
    neurons_pruned = _get_number_of_neurons(pruned_clf.module_)

    # Compile the model
    clf.compile(
        x_train,
        configuration=default_configuration,
        use_virtual_lib=True,
    )

    # Compile the pruned model, this will also perform ONNX export and calibration
    pruned_clf.compile(
        x_train,
        configuration=default_configuration,
        use_virtual_lib=True,
    )

    with pytest.raises(ValueError, match=".*Cannot apply.*"):
        pruned_clf.prune(x_train, y_train, 0.5)

    # Test prediction with QuantizedModule
    pruned_clf.predict(x_test)

    assert neurons_pruned[0] < neurons_orig[0]
