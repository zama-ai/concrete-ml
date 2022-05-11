"""Tests for the sklearn decision trees."""
import types
import warnings
from functools import partial

import numpy
import onnx
import pytest
from concrete.numpy import MAXIMUM_BIT_WIDTH
from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import ConvergenceWarning
from torch import nn

# Remark that the dump tests for torch module is directly done in test_compile_torch.py
from concrete.ml.sklearn import (
    DecisionTreeClassifier,
    GammaRegressor,
    LinearRegression,
    LinearSVC,
    LinearSVR,
    LogisticRegression,
    NeuralNetClassifier,
    NeuralNetRegressor,
    PoissonRegressor,
    RandomForestClassifier,
    TweedieRegressor,
    XGBClassifier,
)


def check_onnx_file_dump(algo, algo_name, x, y, str_expected, default_configuration):
    """Fit the model and dump the corresponding ONNX."""

    # Set the model
    model = algo(
        n_bits=6,
    )

    model_params = model.get_params()
    if "random_state" in model_params:
        model_params["random_state"] = numpy.random.randint(0, 2**15)

    model.set_params(**model_params)

    with warnings.catch_warnings():
        # Sometimes, we miss convergence, which is not a problem for our test
        warnings.simplefilter("ignore", category=ConvergenceWarning)

        # Sometimes, we hit "RuntimeWarning: overflow encountered in exp", which is not a problem
        # for our test
        if algo_name == "NeuralNetRegressor":
            warnings.simplefilter("ignore", category=RuntimeWarning)

        model.fit(x, y)

    with warnings.catch_warnings():
        # Sometimes, we hit "RuntimeWarning: overflow encountered in exp", which is not a problem
        # for our test
        if algo_name == "NeuralNetRegressor":
            warnings.simplefilter("ignore", category=RuntimeWarning)

        # Use virtual lib to not have issues with precision
        model.compile(x, default_configuration, use_virtual_lib=True)

    # Get ONNX model
    onnx_model = model.onnx_model

    # Save locally on disk, if one wants to have a look
    onnx.save(onnx_model, "/tmp/" + algo_name + ".onnx")

    # Remove initializers, since they change from one seed to the other
    if algo_name in ["DecisionTreeClassifier", "RandomForestClassifier", "XGBClassifier"]:
        while len(onnx_model.graph.initializer) > 0:
            del onnx_model.graph.initializer[0]

    str_model = onnx.helper.printable_graph(onnx_model.graph)
    print(f"{algo_name}:")
    print(str_model)

    # Test equality when it does not depend on seeds
    if algo not in [RandomForestClassifier, RandomForestClassifier]:
        assert str_model == str_expected


@pytest.mark.parametrize(
    "algo, algo_name, str_expected",
    [
        (
            partial(
                DecisionTreeClassifier,
                max_depth=4,
            ),
            "DecisionTreeClassifier",
            """graph torch-jit-export (
  %input_0[DOUBLE, symx20]
) {
  %onnx::Greater_7 = Gemm[alpha = 1, beta = 0, transB = 1](%_operators.0.weight_1, %input_0)
  %onnx::Not_8 = Greater(%onnx::Greater_7, %_operators.0.bias_1)
  %onnx::Reshape_9 = Not(%onnx::Not_8)
  %onnx::Reshape_10 = Constant[value = <Tensor>]()
  %onnx::Cast_11 = Reshape(%onnx::Reshape_9, %onnx::Reshape_10)
  %onnx::Reshape_13 = MatMul(%_operators.0.weight_2, %onnx::Cast_11)
  %onnx::Reshape_14 = Constant[value = <Tensor>]()
  %onnx::Equal_15 = Reshape(%onnx::Reshape_13, %onnx::Reshape_14)
  %onnx::Reshape_16 = Equal(%_operators.0.bias_2, %onnx::Equal_15)
  %onnx::Reshape_17 = Constant[value = <Tensor>]()
  %onnx::Cast_18 = Reshape(%onnx::Reshape_16, %onnx::Reshape_17)
  %onnx::Reshape_20 = MatMul(%_operators.0.weight_3, %onnx::Cast_18)
  %onnx::Reshape_21 = Constant[value = <Tensor>]()
  %x = Reshape(%onnx::Reshape_20, %onnx::Reshape_21)
  return %x
}""",
        ),
        (
            partial(
                RandomForestClassifier,
                max_depth=4,
            ),
            "RandomForestClassifier",
            """graph torch-jit-export (
  %input_0[DOUBLE, symx20]
) {
  %onnx::Greater_7 = Gemm[alpha = 1, beta = 0, transB = 1](%_operators.0.weight_1, %input_0)
  %onnx::Not_8 = Greater(%onnx::Greater_7, %_operators.0.bias_1)
  %onnx::Reshape_9 = Not(%onnx::Not_8)
  %onnx::Reshape_10 = Constant[value = <Tensor>]()
  %onnx::Cast_11 = Reshape(%onnx::Reshape_9, %onnx::Reshape_10)
  %onnx::Reshape_13 = MatMul(%_operators.0.weight_2, %onnx::Cast_11)
  %onnx::Reshape_14 = Constant[value = <Tensor>]()
  %onnx::Equal_15 = Reshape(%onnx::Reshape_13, %onnx::Reshape_14)
  %onnx::Reshape_16 = Equal(%_operators.0.bias_2, %onnx::Equal_15)
  %onnx::Reshape_17 = Constant[value = <Tensor>]()
  %onnx::Cast_18 = Reshape(%onnx::Reshape_16, %onnx::Reshape_17)
  %onnx::Reshape_20 = MatMul(%_operators.0.weight_3, %onnx::Cast_18)
  %onnx::Reshape_21 = Constant[value = <Tensor>]()
  %x = Reshape(%onnx::Reshape_20, %onnx::Reshape_21)
  return %x
}""",
        ),
        (
            partial(
                XGBClassifier,
                max_depth=4,
            ),
            "XGBClassifier",
            """graph torch-jit-export (
  %input_0[DOUBLE, symx20]
) {
  %onnx::Less_7 = Gemm[alpha = 1, beta = 0, transB = 1](%_operators.0.weight_1, %input_0)
  %onnx::Reshape_8 = Less(%onnx::Less_7, %_operators.0.bias_1)
  %onnx::Reshape_9 = Constant[value = <Tensor>]()
  %onnx::Cast_10 = Reshape(%onnx::Reshape_8, %onnx::Reshape_9)
  %onnx::Reshape_12 = MatMul(%_operators.0.weight_2, %onnx::Cast_10)
  %onnx::Reshape_13 = Constant[value = <Tensor>]()
  %onnx::Equal_14 = Reshape(%onnx::Reshape_12, %onnx::Reshape_13)
  %onnx::Reshape_15 = Equal(%_operators.0.bias_2, %onnx::Equal_14)
  %onnx::Reshape_16 = Constant[value = <Tensor>]()
  %onnx::Cast_17 = Reshape(%onnx::Reshape_15, %onnx::Reshape_16)
  %onnx::Reshape_19 = MatMul(%_operators.0.weight_3, %onnx::Cast_17)
  %onnx::Reshape_20 = Constant[value = <Tensor>]()
  %x = Reshape(%onnx::Reshape_19, %onnx::Reshape_20)
  return %x
}""",
        ),
    ],
)
@pytest.mark.parametrize(
    "load_data",
    [
        pytest.param(
            lambda: make_classification(
                n_samples=100,
                n_features=20,
                n_classes=4,
                n_informative=10,
                random_state=numpy.random.randint(0, 2**15),
            ),
            id="make_classification",
        ),
    ],
)
def test_tree_classifier(
    algo,
    algo_name,
    str_expected,
    load_data,
    default_configuration,
):
    """Tests the tree models."""
    x, y = load_data()

    check_onnx_file_dump(algo, algo_name, x, y, str_expected, default_configuration)


@pytest.mark.parametrize(
    "algo, str_expected",
    [
        (
            PoissonRegressor,
            """graph torch-jit-export (
  %onnx::MatMul_0[DOUBLE, 20]
) initializers (
  %linear.bias[DOUBLE, scalar]
  %onnx::MatMul_7[DOUBLE, 20x1]
) {
  %onnx::Add_4 = MatMul(%onnx::MatMul_0, %onnx::MatMul_7)
  %onnx::Exp_5 = Add(%linear.bias, %onnx::Add_4)
  %6 = Exp(%onnx::Exp_5)
  return %6
}""",
        ),
        (
            GammaRegressor,
            """graph torch-jit-export (
  %onnx::MatMul_0[DOUBLE, 20]
) initializers (
  %linear.bias[DOUBLE, scalar]
  %onnx::MatMul_7[DOUBLE, 20x1]
) {
  %onnx::Add_4 = MatMul(%onnx::MatMul_0, %onnx::MatMul_7)
  %onnx::Exp_5 = Add(%linear.bias, %onnx::Add_4)
  %6 = Exp(%onnx::Exp_5)
  return %6
}""",
        ),
        (
            TweedieRegressor,
            """graph torch-jit-export (
  %onnx::MatMul_0[DOUBLE, 20]
) initializers (
  %linear.bias[DOUBLE, scalar]
  %onnx::MatMul_6[DOUBLE, 20x1]
) {
  %onnx::Add_4 = MatMul(%onnx::MatMul_0, %onnx::MatMul_6)
  %5 = Add(%linear.bias, %onnx::Add_4)
  return %5
}""",
        ),
        (
            LinearSVC,
            """graph torch-jit-export (
  %input_0[DOUBLE, symx20]
) initializers (
  %_operators.0.coefficients[FLOAT, 20x4]
  %_operators.0.intercepts[FLOAT, 4]
) {
  %onnx::Sigmoid_6 = Gemm[alpha = 1, beta = 1](%input_0, %_operators.0.coefficients, """
            + """%_operators.0.intercepts)
  %onnx::ReduceSum_7 = Sigmoid(%onnx::Sigmoid_6)
  return %onnx::ReduceSum_7
}""",
        ),
        (
            LogisticRegression,
            """graph torch-jit-export (
  %input_0[DOUBLE, symx20]
) initializers (
  %_operators.0.coefficients[FLOAT, 20x1]
  %_operators.0.intercepts[FLOAT, 1]
) {
  %onnx::Sigmoid_6 = Gemm[alpha = 1, beta = 1](%input_0, %_operators.0.coefficients, """
            + """%_operators.0.intercepts)
  %onnx::Sub_7 = Sigmoid(%onnx::Sigmoid_6)
  return %onnx::Sub_7
}""",
        ),
        (
            LinearRegression,
            """graph torch-jit-export (
  %input_0[DOUBLE, symx20]
) initializers (
  %_operators.0.coefficients[FLOAT, 20x1]
  %_operators.0.intercepts[FLOAT, 1]
) {
  %variable = Gemm[alpha = 1, beta = 1](%input_0, %_operators.0.coefficients, """
            + """%_operators.0.intercepts)
  return %variable
}""",
        ),
        (
            LinearSVR,
            """graph torch-jit-export (
  %input_0[DOUBLE, symx20]
) initializers (
  %_operators.0.coefficients[FLOAT, 20x1]
  %_operators.0.intercepts[FLOAT, 1]
) {
  %variable = Gemm[alpha = 1, beta = 1](%input_0, %_operators.0.coefficients, """
            + """%_operators.0.intercepts)
  return %variable
}""",
        ),
        (
            lambda n_bits: NeuralNetClassifier(
                n_bits=n_bits,
                **{
                    "module__n_layers": 3,
                    "module__n_w_bits": 2,
                    "module__n_a_bits": 2,
                    "module__n_accum_bits": MAXIMUM_BIT_WIDTH,
                    "module__n_outputs": 2,
                    "module__input_dim": 20,
                    "module__activation_function": nn.SELU,
                    "max_epochs": 10,
                    "verbose": 0,
                },
            ),
            """graph torch-jit-export (
  %onnx::MatMul_0[FLOAT, 20]
) initializers (
  %features.fc0.bias[FLOAT, 80]
  %features.fc1.bias[FLOAT, 80]
  %features.fc2.bias[FLOAT, 2]
  %onnx::MatMul_19[FLOAT, 20x80]
  %onnx::MatMul_20[FLOAT, 80x80]
  %onnx::MatMul_21[FLOAT, 80x2]
) {
  %onnx::Add_8 = MatMul(%onnx::MatMul_0, %onnx::MatMul_19)
  %input = Add(%features.fc0.bias, %onnx::Add_8)
  %onnx::MatMul_10 = Selu(%input)
  %onnx::Add_12 = MatMul(%onnx::MatMul_10, %onnx::MatMul_20)
  %input.3 = Add(%features.fc1.bias, %onnx::Add_12)
  %onnx::MatMul_14 = Selu(%input.3)
  %onnx::Add_16 = MatMul(%onnx::MatMul_14, %onnx::MatMul_21)
  %input.7 = Add(%features.fc2.bias, %onnx::Add_16)
  %18 = Selu(%input.7)
  return %18
}""",
        ),
        (
            lambda n_bits: NeuralNetRegressor(
                n_bits=n_bits,
                **{
                    "module__n_layers": 3,
                    "module__n_w_bits": 2,
                    "module__n_a_bits": 2,
                    "module__n_accum_bits": MAXIMUM_BIT_WIDTH,
                    "module__n_outputs": 2,
                    "module__input_dim": 10,
                    "module__activation_function": nn.SELU,
                    "max_epochs": 10,
                    "verbose": 0,
                },
            ),
            """graph torch-jit-export (
  %onnx::MatMul_0[FLOAT, 10]
) initializers (
  %features.fc0.bias[FLOAT, 40]
  %features.fc1.bias[FLOAT, 40]
  %features.fc2.bias[FLOAT, 2]
  %onnx::MatMul_19[FLOAT, 10x40]
  %onnx::MatMul_20[FLOAT, 40x40]
  %onnx::MatMul_21[FLOAT, 40x2]
) {
  %onnx::Add_8 = MatMul(%onnx::MatMul_0, %onnx::MatMul_19)
  %input = Add(%features.fc0.bias, %onnx::Add_8)
  %onnx::MatMul_10 = Selu(%input)
  %onnx::Add_12 = MatMul(%onnx::MatMul_10, %onnx::MatMul_20)
  %input.3 = Add(%features.fc1.bias, %onnx::Add_12)
  %onnx::MatMul_14 = Selu(%input.3)
  %onnx::Add_16 = MatMul(%onnx::MatMul_14, %onnx::MatMul_21)
  %input.7 = Add(%features.fc2.bias, %onnx::Add_16)
  %18 = Selu(%input.7)
  return %18
}""",
        ),
    ],
)
def test_other_models(
    algo,
    str_expected,
    default_configuration,
):
    """Tests the linear models."""
    if isinstance(algo, (types.FunctionType, types.LambdaType)):
        if algo(0).__class__ is NeuralNetClassifier:
            algo_name = "NeuralNetClassifier"
        if algo(0).__class__ is NeuralNetRegressor:
            algo_name = "NeuralNetRegressor"
    else:
        algo_name = algo.__name__

    assert isinstance(algo_name, str), f"problem {algo}"

    n_classes = 4

    if algo_name in ["LogisticRegression", "NeuralNetClassifier"]:
        n_classes = 2

    if algo_name in ["NeuralNetRegressor"]:
        x, y, _ = make_regression(
            1000,
            n_features=10,
            n_informative=10,
            n_targets=2,
            noise=2,
            random_state=numpy.random.randint(0, 2**15),
            coef=True,
        )

        if y.ndim == 1:
            y = numpy.expand_dims(y, 1)

        x = x.astype(numpy.float32)
        y = y.astype(numpy.float32)

    else:
        x, y = make_classification(
            n_samples=100,
            n_features=20,
            n_classes=n_classes,
            n_informative=10,
            random_state=numpy.random.randint(0, 2**15),
        )

        # Get the dataset
        if algo_name == "GammaRegressor":
            y = numpy.abs(y) + 1

        if algo_name in ["NeuralNetClassifier"]:
            x = x.astype(numpy.float32)

    check_onnx_file_dump(algo, algo_name, x, y, str_expected, default_configuration)
