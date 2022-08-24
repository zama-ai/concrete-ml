"""Tests for the sklearn decision trees."""
import warnings
from functools import partial

import numpy
import onnx
import pytest
from shared import classifiers, regressors
from sklearn.exceptions import ConvergenceWarning

# Remark that the dump tests for torch module is directly done in test_compile_torch.py
from concrete.ml.sklearn import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    ElasticNet,
    GammaRegressor,
    Lasso,
    LinearRegression,
    LinearSVC,
    LinearSVR,
    LogisticRegression,
    NeuralNetClassifier,
    NeuralNetRegressor,
    PoissonRegressor,
    RandomForestClassifier,
    Ridge,
    TweedieRegressor,
    XGBClassifier,
)


def check_onnx_file_dump(model, parameters, load_data, str_expected, default_configuration):
    """Fit the model and dump the corresponding ONNX."""

    if isinstance(model, partial):
        model_name = model.func.__name__
    else:
        model_name = model.__name__

    # Get the data
    if model_name == "NeuralNetClassifier":
        x, y = load_data(
            dataset="classification",
            n_samples=100,
            n_features=10,
            n_classes=2,
            n_informative=10,
            n_redundant=0,
            random_state=numpy.random.randint(0, 2**15),
        )
        x = x.astype(numpy.float32)
    elif model_name == "NeuralNetRegressor":
        x, y = load_data(
            dataset="regression",
            strictly_positive=False,
            n_samples=1000,
            n_features=10,
            n_informative=10,
            n_targets=1,
            noise=2.0,
            random_state=numpy.random.randint(0, 2**15),
        )
        x = x.astype(numpy.float32)
        y = y.reshape(-1, 1).astype(numpy.float32)
    else:
        x, y = load_data(**parameters)

    # Set the model
    model = model(n_bits=6)

    model_params = model.get_params()
    if "random_state" in model_params:
        model_params["random_state"] = numpy.random.randint(0, 2**15)

    model.set_params(**model_params)

    with warnings.catch_warnings():
        # Sometimes, we miss convergence, which is not a problem for our test
        warnings.simplefilter("ignore", category=ConvergenceWarning)

        # Sometimes, we hit "RuntimeWarning: overflow encountered in exp", which is not a problem
        # for our test
        if model_name == "NeuralNetRegressor":
            warnings.simplefilter("ignore", category=RuntimeWarning)

        model.fit(x, y)

    with warnings.catch_warnings():
        # Use virtual lib to not have issues with precision
        model.compile(x, default_configuration, use_virtual_lib=True)

    # Get ONNX model
    onnx_model = model.onnx_model

    # Save locally on disk, if one wants to have a look
    onnx.save(onnx_model, "/tmp/" + model_name + ".onnx")

    # Remove initializers, since they change from one seed to the other
    if model_name in [
        "DecisionTreeRegressor",
        "DecisionTreeClassifier",
        "RandomForestClassifier",
        "XGBClassifier",
    ]:
        while len(onnx_model.graph.initializer) > 0:
            del onnx_model.graph.initializer[0]

    str_model = onnx.helper.printable_graph(onnx_model.graph)
    print(f"{model_name}:")
    print(str_model)

    # Test equality when it does not depend on seeds
    if model_name not in {"RandomForestClassifier"}:
        assert str_model == str_expected


@pytest.mark.parametrize("model, parameters", classifiers + regressors)
def test_dump(
    model,
    parameters,
    load_data,
    default_configuration,
):
    """Tests dump."""
    expected_strings = {
        XGBClassifier: """graph torch_jit (
  %input_0[DOUBLE, symx10]
) {
  %onnx::Less_7 = Gemm[alpha = 1, beta = 0, transB = 1](%_operators.0.weight_1, %input_0)
  %onnx::Reshape_8 = Less(%onnx::Less_7, %_operators.0.bias_1)
  %onnx::Reshape_9 = Constant[value = <Tensor>]()
  %onnx::Cast_10 = Reshape[allowzero = 0](%onnx::Reshape_8, %onnx::Reshape_9)
  %onnx::Reshape_12 = MatMul(%_operators.0.weight_2, %onnx::Cast_10)
  %onnx::Reshape_13 = Constant[value = <Tensor>]()
  %onnx::Equal_14 = Reshape[allowzero = 0](%onnx::Reshape_12, %onnx::Reshape_13)
  %onnx::Reshape_15 = Equal(%_operators.0.bias_2, %onnx::Equal_14)
  %onnx::Reshape_16 = Constant[value = <Tensor>]()
  %onnx::Cast_17 = Reshape[allowzero = 0](%onnx::Reshape_15, %onnx::Reshape_16)
  %onnx::Reshape_19 = MatMul(%_operators.0.weight_3, %onnx::Cast_17)
  %onnx::Reshape_20 = Constant[value = <Tensor>]()
  %x = Reshape[allowzero = 0](%onnx::Reshape_19, %onnx::Reshape_20)
  return %x
}""",
        GammaRegressor: """graph torch_jit (
  %onnx::MatMul_0[DOUBLE, 10]
) initializers (
  %linear.bias[DOUBLE, scalar]
  %onnx::MatMul_7[DOUBLE, 10x1]
) {
  %onnx::Add_4 = MatMul(%onnx::MatMul_0, %onnx::MatMul_7)
  %onnx::Exp_5 = Add(%linear.bias, %onnx::Add_4)
  %6 = Exp(%onnx::Exp_5)
  return %6
}""",
        LinearRegression: """graph torch_jit (
  %input_0[DOUBLE, symx10]
) initializers (
  %_operators.0.coefficients[FLOAT, 10x2]
  %_operators.0.intercepts[FLOAT, 2]
) {
  %variable = Gemm[alpha = 1, beta = 1](%input_0, """
        """%_operators.0.coefficients, %_operators.0.intercepts)
  return %variable
}""",
        TweedieRegressor: """graph torch_jit (
  %onnx::MatMul_0[DOUBLE, 10]
) initializers (
  %linear.bias[DOUBLE, scalar]
  %onnx::MatMul_6[DOUBLE, 10x1]
) {
  %onnx::Add_4 = MatMul(%onnx::MatMul_0, %onnx::MatMul_6)
  %5 = Add(%linear.bias, %onnx::Add_4)
  return %5
}""",
        PoissonRegressor: """graph torch_jit (
  %onnx::MatMul_0[DOUBLE, 10]
) initializers (
  %linear.bias[DOUBLE, scalar]
  %onnx::MatMul_7[DOUBLE, 10x1]
) {
  %onnx::Add_4 = MatMul(%onnx::MatMul_0, %onnx::MatMul_7)
  %onnx::Exp_5 = Add(%linear.bias, %onnx::Add_4)
  %6 = Exp(%onnx::Exp_5)
  return %6
}""",
        DecisionTreeClassifier: """graph torch_jit (
  %input_0[DOUBLE, symx10]
) {
  %onnx::LessOrEqual_7 = Gemm[alpha = 1, beta = 0, transB = 1](%_operators.0.weight_1, %input_0)
  %onnx::Reshape_8 = LessOrEqual(%onnx::LessOrEqual_7, %_operators.0.bias_1)
  %onnx::Reshape_9 = Constant[value = <Tensor>]()
  %onnx::Cast_10 = Reshape[allowzero = 0](%onnx::Reshape_8, %onnx::Reshape_9)
  %onnx::Reshape_12 = MatMul(%_operators.0.weight_2, %onnx::Cast_10)
  %onnx::Reshape_13 = Constant[value = <Tensor>]()
  %onnx::Equal_14 = Reshape[allowzero = 0](%onnx::Reshape_12, %onnx::Reshape_13)
  %onnx::Reshape_15 = Equal(%_operators.0.bias_2, %onnx::Equal_14)
  %onnx::Reshape_16 = Constant[value = <Tensor>]()
  %onnx::Cast_17 = Reshape[allowzero = 0](%onnx::Reshape_15, %onnx::Reshape_16)
  %onnx::Reshape_19 = MatMul(%_operators.0.weight_3, %onnx::Cast_17)
  %onnx::Reshape_20 = Constant[value = <Tensor>]()
  %x = Reshape[allowzero = 0](%onnx::Reshape_19, %onnx::Reshape_20)
  return %x
}""",
        LinearSVR: """graph torch_jit (
  %input_0[DOUBLE, symx10]
) initializers (
  %_operators.0.coefficients[FLOAT, 10x1]
  %_operators.0.intercepts[FLOAT, 1]
) {
  %variable = Gemm[alpha = 1, beta = 1](%input_0, """
        """%_operators.0.coefficients, %_operators.0.intercepts)
  return %variable
}""",
        LogisticRegression: """graph torch_jit (
  %input_0[DOUBLE, symx10]
) initializers (
  %_operators.0.coefficients[FLOAT, 10x1]
  %_operators.0.intercepts[FLOAT, 1]
) {
  %onnx::Sigmoid_6 = Gemm[alpha = 1, beta = 1](%input_0, """
        """%_operators.0.coefficients, %_operators.0.intercepts)
  %onnx::Sub_7 = Sigmoid(%onnx::Sigmoid_6)
  return %onnx::Sub_7
}""",
        LinearSVC: """graph torch_jit (
  %input_0[DOUBLE, symx10]
) initializers (
  %_operators.0.coefficients[FLOAT, 10x1]
  %_operators.0.intercepts[FLOAT, 1]
) {
  %onnx::Sigmoid_6 = Gemm[alpha = 1, beta = 1](%input_0, """
        """%_operators.0.coefficients, %_operators.0.intercepts)
  %onnx::Sub_7 = Sigmoid(%onnx::Sigmoid_6)
  return %onnx::Sub_7
}""",
        RandomForestClassifier: "Not tested",
        NeuralNetClassifier: """graph torch_jit (
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
  %onnx::MatMul_10 = Relu(%input)
  %onnx::Add_12 = MatMul(%onnx::MatMul_10, %onnx::MatMul_20)
  %input.3 = Add(%features.fc1.bias, %onnx::Add_12)
  %onnx::MatMul_14 = Relu(%input.3)
  %onnx::Add_16 = MatMul(%onnx::MatMul_14, %onnx::MatMul_21)
  %input.7 = Add(%features.fc2.bias, %onnx::Add_16)
  %18 = Relu(%input.7)
  return %18
}""",
        NeuralNetRegressor: """graph torch_jit (
  %onnx::MatMul_0[FLOAT, 10]
) initializers (
  %features.fc0.bias[FLOAT, 10]
  %features.fc1.bias[FLOAT, 10]
  %features.fc2.bias[FLOAT, 1]
  %onnx::MatMul_19[FLOAT, 10x10]
  %onnx::MatMul_20[FLOAT, 10x10]
  %onnx::MatMul_21[FLOAT, 10x1]
) {
  %onnx::Add_8 = MatMul(%onnx::MatMul_0, %onnx::MatMul_19)
  %input = Add(%features.fc0.bias, %onnx::Add_8)
  %onnx::MatMul_10 = Relu(%input)
  %onnx::Add_12 = MatMul(%onnx::MatMul_10, %onnx::MatMul_20)
  %input.3 = Add(%features.fc1.bias, %onnx::Add_12)
  %onnx::MatMul_14 = Relu(%input.3)
  %onnx::Add_16 = MatMul(%onnx::MatMul_14, %onnx::MatMul_21)
  %input.7 = Add(%features.fc2.bias, %onnx::Add_16)
  %18 = Relu(%input.7)
  return %18
}""",
        Ridge: """graph torch_jit (
  %input_0[DOUBLE, symx10]
) initializers (
  %_operators.0.coefficients[FLOAT, 10x1]
  %_operators.0.intercepts[FLOAT, 1]
) {
  %variable = Gemm[alpha = 1, beta = 1]"""
        """(%input_0, %_operators.0.coefficients, %_operators.0.intercepts)
  return %variable
}""",
        Lasso: """graph torch_jit (
  %input_0[DOUBLE, symx10]
) initializers (
  %_operators.0.coefficients[FLOAT, 10x1]
  %_operators.0.intercepts[FLOAT, 1]
) {
  %variable = Gemm[alpha = 1, beta = 1]"""
        """(%input_0, %_operators.0.coefficients, %_operators.0.intercepts)
  return %variable
}""",
        ElasticNet: """graph torch_jit (
  %input_0[DOUBLE, symx10]
) initializers (
  %_operators.0.coefficients[FLOAT, 10x1]
  %_operators.0.intercepts[FLOAT, 1]
) {
  %variable = Gemm[alpha = 1, beta = 1]"""
        """(%input_0, %_operators.0.coefficients, %_operators.0.intercepts)
  return %variable
}""",
        DecisionTreeRegressor: """graph torch_jit (
  %input_0[DOUBLE, symx10]
) {
  %onnx::LessOrEqual_7 = Gemm[alpha = 1, beta = 0, transB = 1](%_operators.0.weight_1, %input_0)
  %onnx::Reshape_8 = LessOrEqual(%onnx::LessOrEqual_7, %_operators.0.bias_1)
  %onnx::Reshape_9 = Constant[value = <Tensor>]()
  %onnx::Cast_10 = Reshape[allowzero = 0](%onnx::Reshape_8, %onnx::Reshape_9)
  %onnx::Reshape_12 = MatMul(%_operators.0.weight_2, %onnx::Cast_10)
  %onnx::Reshape_13 = Constant[value = <Tensor>]()
  %onnx::Equal_14 = Reshape[allowzero = 0](%onnx::Reshape_12, %onnx::Reshape_13)
  %onnx::Reshape_15 = Equal(%_operators.0.bias_2, %onnx::Equal_14)
  %onnx::Reshape_16 = Constant[value = <Tensor>]()
  %onnx::Cast_17 = Reshape[allowzero = 0](%onnx::Reshape_15, %onnx::Reshape_16)
  %onnx::Reshape_19 = MatMul(%_operators.0.weight_3, %onnx::Cast_17)
  %onnx::Reshape_20 = Constant[value = <Tensor>]()
  %x = Reshape[allowzero = 0](%onnx::Reshape_19, %onnx::Reshape_20)
  return %x
}""",
    }

    if isinstance(model, partial):
        model_class = model.func
    else:
        model_class = model

    str_expected = expected_strings[model_class] if model_class in expected_strings else ""

    check_onnx_file_dump(model, parameters, load_data, str_expected, default_configuration)
