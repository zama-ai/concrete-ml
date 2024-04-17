"""Tests for the FHE sklearn compatible NNs."""

from copy import deepcopy
from itertools import product

import brevitas.nn as qnn
import numpy
import pandas
import pytest
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn

from concrete.ml.common import utils
from concrete.ml.common.utils import (
    MAX_BITWIDTH_BACKWARD_COMPATIBLE,
    is_classifier_or_partial_classifier,
    is_regressor_or_partial_regressor,
)
from concrete.ml.quantization.base_quantized_op import QuantizedMixingOp
from concrete.ml.quantization.post_training import PowerOfTwoScalingRoundPBSAdapter
from concrete.ml.sklearn import _get_sklearn_neural_net_models
from concrete.ml.sklearn.qnn_module import SparseQuantNeuralNetwork


@pytest.mark.parametrize("model_class", _get_sklearn_neural_net_models())
def test_parameter_validation(model_class, load_data):
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

    # Get the data-set. The data generation is seeded in load_data.
    if is_classifier_or_partial_classifier(model_class):
        x, y = load_data(
            model_class,
            n_samples=1000,
            n_features=10,
            n_redundant=0,
            n_repeated=0,
            n_informative=10,
            n_classes=2,
            class_sep=2,
        )

        valid_params["criterion__weight"] = [1, 1]

    # Get the data-set. The data generation is seeded in load_data.
    elif is_regressor_or_partial_regressor(model_class):
        x, y, _ = load_data(
            model_class,
            n_samples=1000,
            n_features=10,
            n_informative=10,
            noise=2,
            coef=True,
        )
    else:
        raise ValueError(f"Data generator not implemented for {str(model_class)}")

    init_module = SparseQuantNeuralNetwork(
        input_dim=1,
        n_layers=1,
        n_outputs=1,
    )

    invalid_params_and_exception_pattern = {
        "init": [
            ("module__n_outputs", 0, ".*module__n_outputs.*"),
            ("module__input_dim", 0, ".*module__input_dim.*"),
            ("n_bits", 4, "Setting `n_bits` in Quantized Neural Networks is not possible.*"),
            ("module", init_module, "Setting `module` manually is forbidden..*"),
        ],
        "fit": [
            ("module__n_layers", 0, ".* number of layers.*"),
            ("module__n_w_bits", 0, ".* quantization bit-width.*"),
            ("module__n_a_bits", 0, ".* quantization bit-width.*"),
            ("module__n_accum_bits", 0, ".* accumulator bit-width.*"),
        ],
    }

    params = deepcopy(valid_params)

    for inv_param in invalid_params_and_exception_pattern["init"]:
        params = deepcopy(valid_params)
        params[inv_param[0]] = inv_param[1]

        with pytest.raises(
            ValueError,
            match=inv_param[2],
        ):
            model = model_class(**params)

    for inv_param in invalid_params_and_exception_pattern["fit"]:
        params = deepcopy(valid_params)
        params[inv_param[0]] = inv_param[1]

        model = model_class(**params)

        with pytest.raises(AssertionError, match=".*The underlying model.*"):
            _ = model.base_module

        with pytest.raises(ValueError, match=inv_param[2]):
            model.fit(x, y)


@pytest.mark.parametrize(
    "activation_function",
    [
        pytest.param(nn.ReLU),
        pytest.param(nn.Sigmoid),
        pytest.param(nn.SELU),
        pytest.param(nn.CELU),
    ],
)
@pytest.mark.parametrize("model_class", _get_sklearn_neural_net_models())
def test_compile_and_calib(
    activation_function,
    model_class,
    load_data,
    default_configuration,
):
    """Test whether the sklearn quantized NN wrappers compile to FHE and execute well on encrypted
    inputs"""

    n_features = 10

    # Get the data-set. The data generation is seeded in load_data.
    if is_classifier_or_partial_classifier(model_class):
        x, y = load_data(
            model_class,
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

    # Get the data-set. The data generation is seeded in load_data.
    elif is_regressor_or_partial_regressor(model_class):
        x, y, _ = load_data(
            model_class,
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
        raise ValueError(f"Data generator not implemented for {str(model_class)}")

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
        "module__power_of_two_scaling": False,
        "max_epochs": 10,
        "verbose": 0,
    }

    if is_classifier_or_partial_classifier(model_class):
        params["criterion__weight"] = class_weights

    model = model_class(**params)

    # Train the model
    # Needed for coverage
    if is_regressor_or_partial_regressor(model_class):
        for x_d_type_regressor, y_d_type_regressor in product(
            [numpy.float32, numpy.float64], [numpy.float32, numpy.float64]
        ):
            model.fit(x_train.astype(x_d_type_regressor), y_train.astype(y_d_type_regressor))

    elif is_classifier_or_partial_classifier(model_class):
        for x_d_type_classifier, y_d_type_classifier in product(
            [numpy.float32, numpy.float64], [numpy.int32, numpy.int64]
        ):
            model.fit(x_train.astype(x_d_type_classifier), y_train.astype(y_d_type_classifier))

    # Train normally
    model.fit(x_train, y_train)

    if is_classifier_or_partial_classifier(model_class):
        y_pred_clear = model.predict(x_train, fhe="disable")
        # Check that the predicted classes are all contained in the model class list
        assert set(numpy.unique(y_pred_clear)).issubset(set(model.classes_))

    # Compile the model
    model.compile(
        x_train,
        configuration=default_configuration,
    )

    # Execute in FHE, but don't check the value.
    # Since FHE execution introduces some stochastic errors,
    # accuracy of FHE compiled classifiers and regressors is measured in the benchmarks
    model.predict(x_test[0, :], fhe="execute")


@pytest.mark.parametrize(
    "model_classes, bad_types, expected_error",
    [
        pytest.param(
            _get_sklearn_neural_net_models(regressor=False, classifier=True),
            ("float32", "uint8"),
            "Neural Network classifier target dtype .* int64 .*",
            id="NeuralNetClassifier-target-uint8",
        ),
        pytest.param(
            _get_sklearn_neural_net_models(regressor=True, classifier=False),
            ("float64", "complex64"),
            "Neural Network regressor target dtype .* float32 .*",
            id="NeuralNetRegressor-target-complex64",
        ),
        pytest.param(
            _get_sklearn_neural_net_models(),
            ("complex64", "int64"),
            "Neural Network .* input dtype .* float32 .*",
            id="NeuralNets-input-complex64",
        ),
        pytest.param(
            _get_sklearn_neural_net_models(),
            ("int64", "int64"),
            "Neural Network .* input dtype .* float32 .*",
            id="NeuralNets-input-int64",
        ),
        pytest.param(
            _get_sklearn_neural_net_models(),
            ("str", "int64"),
            "Neural Network .* input dtype .* float32 .*",
            id="NeuralNets-input-str",
        ),
        pytest.param(
            _get_sklearn_neural_net_models(),
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
        x, y = load_data(model_class)

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
@pytest.mark.parametrize("model_class", _get_sklearn_neural_net_models())
@pytest.mark.parametrize("accum_bits", [5, 8])
def test_structured_pruning(
    activation_function, model_class, accum_bits, load_data, default_configuration
):
    """Test whether the sklearn quantized NN wrappers compile to FHE and execute well on encrypted
    inputs"""
    n_features = 10

    # Get the data-set. The data generation is seeded in load_data.
    if is_classifier_or_partial_classifier(model_class):
        x, y = load_data(
            model_class,
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

    # Get the data-set. The data generation is seeded in load_data.
    elif is_regressor_or_partial_regressor(model_class):
        x, y, _ = load_data(
            model_class,
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
        raise ValueError(f"Data generator not implemented for {str(model_class)}")

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
        "module__n_accum_bits": accum_bits,
        "module__activation_function": activation_function,
        "max_epochs": 2,
        "verbose": 0,
    }

    if is_classifier_or_partial_classifier(model_class):
        params["criterion__weight"] = class_weights

    model = model_class(**params)

    with pytest.raises(AttributeError, match=".* model is not fitted.*"):
        model.prune(x_train, y_train, 0.5)

    model.fit(x_train, y_train)

    with pytest.raises(
        ValueError, match="Valid values for `n_prune_neurons_percentage` are in the.*"
    ):
        model.prune(x_train, y_train, -0.1)

    with pytest.raises(
        ValueError, match="Valid values for `n_prune_neurons_percentage` are in the.*"
    ):
        model.prune(x_train, y_train, 1.0)

    def _get_number_of_neurons(module: SparseQuantNeuralNetwork):
        neurons = {}
        idx = 0
        for layer in module.features:
            if not isinstance(layer, nn.Linear) and not isinstance(layer, qnn.QuantLinear):
                continue
            neurons[idx] = layer.weight.shape[0]
            idx += 1
        return neurons

    neurons_orig = _get_number_of_neurons(model.base_module)

    # Compile the model
    model.compile(
        x_train,
        configuration=default_configuration,
    )

    pruned_model = model.prune(x_train, y_train, 0.5)
    neurons_pruned = _get_number_of_neurons(pruned_model.base_module)

    # Compile the pruned model, this will also perform ONNX export and calibration
    pruned_model.compile(
        x_train,
        configuration=default_configuration,
    )

    with pytest.raises(
        ValueError, match="Cannot apply structured pruning optimization to an already pruned model"
    ):
        pruned_model.prune(x_train, y_train, 0.5)

    # Test prediction with QuantizedModule
    pruned_model.predict(x_test)

    assert neurons_pruned[0] < neurons_orig[0]


@pytest.mark.parametrize("model_class", _get_sklearn_neural_net_models())
@pytest.mark.parametrize(
    "unsupported_parameter, expected_error, expected_message",
    [
        pytest.param(
            {"train_split": lambda x: x + 1},
            NotImplementedError,
            (
                "Serializing a custom Callable object is not secure and is therefore disabled. "
                "Please set `train_split` to either None or a ValidSplit instance."
            ),
            id="train_split_callable",
        ),
        pytest.param(
            {"callbacks": lambda x: x + 1},
            NotImplementedError,
            "Serializing a custom Callable object is not secure and is therefore disabled. ",
            id="callbacks_callable",
        ),
        pytest.param(
            {"callbacks": None},
            NotImplementedError,
            (
                "Serializing a custom Callable object is not secure and is therefore disabled. "
                "Additionally, the serialization of skorch's different callback classes is not "
                "supported. Please set `callbacks` to 'disable'. Got .*"
            ),
            id="callbacks_none",
        ),
        pytest.param(
            {"predict_nonlinearity": lambda x: x + 1},
            NotImplementedError,
            (
                "Serializing a custom Callable object is not secure and is therefore disabled. "
                "Please set`predict_nonlinearity` to either None or 'auto'."
            ),
            id="predict_non_linearity_callable",
        ),
    ],
)
def test_serialization_unsupported_parameters(
    model_class,
    unsupported_parameter,
    expected_error,
    expected_message,
    verbose=True,
):
    """Test serialization errors for unsupported parameters."""

    parameters = {
        "module__n_layers": 1,
        "callbacks": "disable",
    }

    parameters.update(unsupported_parameter)

    model = model_class(**parameters)

    if verbose:
        print("Run check_serialization on unsupported parameters")

    with pytest.raises(expected_error, match=expected_message):
        model.dumps()


@pytest.mark.parametrize(
    "activation_function",
    [
        pytest.param(nn.ReLU),
        pytest.param(nn.Sigmoid),
    ],
)
@pytest.mark.parametrize("num_layers", [2, 4])
@pytest.mark.parametrize(
    "model_class", _get_sklearn_neural_net_models(regressor=False, classifier=True)
)
@pytest.mark.parametrize("use_power_of_two_scaling", [True, False])
def test_power_of_two_scaling(
    activation_function,
    model_class,
    num_layers,
    load_data,
    use_power_of_two_scaling,
    default_configuration,
):
    """Check that built-in neural networks can use roundPBS optimization."""

    n_features = 10

    # Get the data-set. The data generation is seeded in load_data.
    x, y = load_data(
        model_class,
        n_samples=1000,
        n_features=n_features,
        n_redundant=0,
        n_repeated=0,
        n_informative=n_features,
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

    # Configure a minimal neural network and train it quickly
    params = {
        "module__n_layers": num_layers,
        "module__n_w_bits": 4,
        "module__n_a_bits": 4,
        "module__n_accum_bits": 32,
        "module__activation_function": activation_function,
        "module__power_of_two_scaling": use_power_of_two_scaling,
        "max_epochs": 2,
        "verbose": 0,
    }

    model = model_class(**params)

    utils.QUANT_ROUND_LIKE_ROUND_PBS = True

    # Train normally. This also converts the torch NN to a QuantizedModule
    # and thus applies the PowerOfTwoScalingRoundPBSAdapter that
    # detects and applies round PBS optimization
    model.fit(x_train, y_train)

    # Count the number of patterns that were optimized with roundPBS
    num_round_pbs_layers = 0
    for _, node_op in model.quantized_module_.quant_layers_dict.values():
        if isinstance(node_op, QuantizedMixingOp):
            num_round_pbs_layers += 1 if node_op.rounding_threshold_bits is not None else 0
            lsbs_to_remove = (
                node_op.lsbs_to_remove["matmul"]
                if (node_op.lsbs_to_remove is not None) and ("matmul" in node_op.lsbs_to_remove)
                else None
            )
            assert node_op.rounding_threshold_bits == lsbs_to_remove

    # Apply the PowerOfTwoScalingRoundPBSAdapter again. The second time
    # the adapter will ignore already optimized patterns but report them
    # as ignored.
    adapter = PowerOfTwoScalingRoundPBSAdapter(model.quantized_module_)
    round_pbs_patterns = adapter.process()

    # The power-of-two optimization will only work
    # when Relu activations are used and scaling factors are forced to be 2**s
    if activation_function is nn.ReLU and use_power_of_two_scaling:
        assert (
            len(round_pbs_patterns) == 0
        ), "Expected number of round PBS optimized patterns was not matched"
        assert (
            adapter.num_ignored_valid_patterns == num_layers - 1
        ), "Expected number of ignored round PBS optimizable patterns was not matched"

        y_pred_clear_round = model.predict(x_test, fhe="disable")

        # Compile the model to ensure rounding is taken into account
        # in compilation
        model.compile(
            x_train,
            configuration=default_configuration,
        )

        # Compute the results with simulation, which uses the actual
        # lookup tables.
        y_pred_sim_round = model.predict(x_test, fhe="simulate")

        # Ensure rounding was compiled in the circuit
        # the number of rounding nodes should be equal
        num_rounding_mlir = model.fhe_circuit.mlir.count(".round")

        assert (
            num_rounding_mlir == num_layers - 1
        ), "Power-of-to adapter: Rounding nodes not found in MLIR"

        # Remove rounding in the network to perform inference without the optimization.
        # We expect a network that was optimized with the power-of-two adapter
        # to be exactly correct to the non-optimized one
        for _, node_op in model.quantized_module_.quant_layers_dict.values():
            if isinstance(node_op, QuantizedMixingOp):
                node_op.rounding_threshold_bits = None
                node_op.lsbs_to_remove = None

        # Predict with the unoptimized network
        y_pred_clear_no_round = model.predict(x_test, fhe="disable")

        # Compare the result with the optimized network with and without
        # rounding. Tolerate at most 1 error
        assert numpy.sum(y_pred_clear_round != y_pred_clear_no_round) <= 1
        assert numpy.sum(y_pred_sim_round != y_pred_clear_no_round) <= 1
    else:
        # If the optimization is not expected to work, check that no patterns were
        # detected
        assert (
            adapter.num_ignored_valid_patterns == 0
        ), "Optimization performed but not expected for round PBS optimizable patterns"
