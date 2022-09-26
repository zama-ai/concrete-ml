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
    RandomForestRegressor,
    Ridge,
    TweedieRegressor,
    XGBClassifier,
    XGBRegressor,
)


def check_onnx_file_dump(model, parameters, load_data, str_expected, default_configuration):
    """Fit the model and dump the corresponding ONNX."""

    if isinstance(model, partial):
        model_name = model.func.__name__
    else:
        model_name = model.__name__

    # Get the dataset. The data generation is seeded in load_data.
    if model_name == "NeuralNetClassifier":
        x, y = load_data(
            dataset="classification",
            n_samples=100,
            n_features=10,
            n_classes=2,
            n_informative=10,
            n_redundant=0,
        )
        x = x.astype(numpy.float32)

    # Get the dataset. The data generation is seeded in load_data.
    elif model_name == "NeuralNetRegressor":
        x, y = load_data(
            dataset="regression",
            strictly_positive=False,
            n_samples=1000,
            n_features=10,
            n_informative=10,
            n_targets=1,
            noise=2.0,
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
        "RandomForestRegressor",
        "XGBClassifier",
    ]:
        while len(onnx_model.graph.initializer) > 0:
            del onnx_model.graph.initializer[0]

    str_model = onnx.helper.printable_graph(onnx_model.graph)
    print(f"{model_name}:")
    print(str_model)

    # Test equality when it does not depend on seeds
    if model_name not in {"RandomForestClassifier", "RandomForestRegressor"}:
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
        XGBRegressor: """graph torch_jit (
  %input_0[DOUBLE, symx10]
) initializers (
  %_operators.0.base_prediction[INT64, 1]
  %_operators.0.weight_1[INT64, 140x10]
  %_operators.0.bias_1[INT64, 140x1]
  %_operators.0.weight_2[INT64, 20x8x7]
  %_operators.0.bias_2[INT64, 160x1]
  %_operators.0.weight_3[INT64, 20x1x8]
) {
  %onnx::Less_8 = Gemm[alpha = 1, beta = 0, transB = 1](%_operators.0.weight_1, %input_0)
  %onnx::Reshape_9 = Less(%onnx::Less_8, %_operators.0.bias_1)
  %onnx::Reshape_10 = Constant[value = <Tensor>]()
  %onnx::Cast_11 = Reshape[allowzero = 0](%onnx::Reshape_9, %onnx::Reshape_10)
  %onnx::Reshape_13 = MatMul(%_operators.0.weight_2, %onnx::Cast_11)
  %onnx::Reshape_14 = Constant[value = <Tensor>]()
  %onnx::Equal_15 = Reshape[allowzero = 0](%onnx::Reshape_13, %onnx::Reshape_14)
  %onnx::Reshape_16 = Equal(%_operators.0.bias_2, %onnx::Equal_15)
  %onnx::Reshape_17 = Constant[value = <Tensor>]()
  %onnx::Cast_18 = Reshape[allowzero = 0](%onnx::Reshape_16, %onnx::Reshape_17)
  %onnx::Reshape_20 = MatMul(%_operators.0.weight_3, %onnx::Cast_18)
  %onnx::Reshape_21 = Constant[value = <Tensor>]()
  %x = Reshape[allowzero = 0](%onnx::Reshape_20, %onnx::Reshape_21)
  return %x
}""",
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
  %onnx::MatMul_6[DOUBLE, 10x1]
) {
  %onnx::Add_4 = MatMul(%onnx::MatMul_0, %onnx::MatMul_6)
  %5 = Add(%linear.bias, %onnx::Add_4)
  return %5
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
  %onnx::MatMul_6[DOUBLE, 10x1]
) {
  %onnx::Add_4 = MatMul(%onnx::MatMul_0, %onnx::MatMul_6)
  %5 = Add(%linear.bias, %onnx::Add_4)
  return %5
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
  return %onnx::Sigmoid_6
}""",
        LinearSVC: """graph torch_jit (
  %input_0[DOUBLE, symx10]
) initializers (
  %_operators.0.coefficients[FLOAT, 10x1]
  %_operators.0.intercepts[FLOAT, 1]
) {
  %onnx::Sigmoid_6 = Gemm[alpha = 1, beta = 1](%input_0, """
        """%_operators.0.coefficients, %_operators.0.intercepts)
  return %onnx::Sigmoid_6
}""",
        RandomForestClassifier: "Not tested",
        RandomForestRegressor: "Not tested",
        NeuralNetClassifier: """graph torch_jit (
  %inp.1[FLOAT, 1x10]
) initializers (
  %features.fc0.bias[FLOAT, 40]
  %features.fc1.bias[FLOAT, 40]
  %features.fc2.bias[FLOAT, 2]
  %bit_width[FLOAT, scalar]
  %scale[FLOAT, scalar]
  %zero_point[FLOAT, scalar]
  %bit_width.3[FLOAT, scalar]
  %x[FLOAT, 40x10]
  %scale.3[FLOAT, scalar]
  %zero_point.3[FLOAT, scalar]
  %bit_width.7[FLOAT, scalar]
  %scale.7[FLOAT, scalar]
  %zero_point.7[FLOAT, scalar]
  %bit_width.11[FLOAT, scalar]
  %x.3[FLOAT, 40x40]
  %scale.11[FLOAT, scalar]
  %zero_point.11[FLOAT, scalar]
  %bit_width.15[FLOAT, scalar]
  %scale.15[FLOAT, scalar]
  %zero_point.15[FLOAT, scalar]
  %bit_width.19[FLOAT, scalar]
  %x.7[FLOAT, 2x40]
  %scale.19[FLOAT, scalar]
  %zero_point.19[FLOAT, scalar]
) {
  %onnx::Gemm_13 = Quant[narrow = 0, rounding_mode = 'ROUND', signed = 1](%inp.1, %scale, %zero_point, %bit_width)
  %onnx::Gemm_18 = Quant[narrow = 1, rounding_mode = 'ROUND', signed = 1](%x, %scale.3, %zero_point.3, %bit_width.3)
  %input = Gemm[alpha = 1, beta = 1, transB = 1](%onnx::Gemm_13, %onnx::Gemm_18, %features.fc0.bias)
  %inp = Relu(%input)
  %onnx::Gemm_24 = Quant[narrow = 0, rounding_mode = 'ROUND', signed = 1](%inp, %scale.7, %zero_point.7, %bit_width.7)
  %onnx::Gemm_29 = Quant[narrow = 1, rounding_mode = 'ROUND', signed = 1](%x.3, %scale.11, %zero_point.11, %bit_width.11)
  %input.3 = Gemm[alpha = 1, beta = 1, transB = 1](%onnx::Gemm_24, %onnx::Gemm_29, %features.fc1.bias)
  %inp.4 = Relu(%input.3)
  %onnx::Gemm_35 = Quant[narrow = 0, rounding_mode = 'ROUND', signed = 1](%inp.4, %scale.15, %zero_point.15, %bit_width.15)
  %onnx::Gemm_40 = Quant[narrow = 1, rounding_mode = 'ROUND', signed = 1](%x.7, %scale.19, %zero_point.19, %bit_width.19)
  %41 = Gemm[alpha = 1, beta = 1, transB = 1](%onnx::Gemm_35, %onnx::Gemm_40, %features.fc2.bias)
  return %41
}""",
        NeuralNetRegressor: """graph torch_jit (
  %inp.1[FLOAT, 1x10]
) initializers (
  %features.fc0.bias[FLOAT, 10]
  %features.fc1.bias[FLOAT, 10]
  %features.fc2.bias[FLOAT, 1]
  %bit_width[FLOAT, scalar]
  %scale[FLOAT, scalar]
  %zero_point[FLOAT, scalar]
  %bit_width.3[FLOAT, scalar]
  %x[FLOAT, 10x10]
  %scale.3[FLOAT, scalar]
  %zero_point.3[FLOAT, scalar]
  %bit_width.7[FLOAT, scalar]
  %scale.7[FLOAT, scalar]
  %zero_point.7[FLOAT, scalar]
  %bit_width.11[FLOAT, scalar]
  %x.3[FLOAT, 10x10]
  %scale.11[FLOAT, scalar]
  %zero_point.11[FLOAT, scalar]
  %bit_width.15[FLOAT, scalar]
  %scale.15[FLOAT, scalar]
  %zero_point.15[FLOAT, scalar]
  %bit_width.19[FLOAT, scalar]
  %x.7[FLOAT, 1x10]
  %scale.19[FLOAT, scalar]
  %zero_point.19[FLOAT, scalar]
) {
  %onnx::Gemm_13 = Quant[narrow = 0, rounding_mode = 'ROUND', signed = 1](%inp.1, %scale, %zero_point, %bit_width)
  %onnx::Gemm_18 = Quant[narrow = 1, rounding_mode = 'ROUND', signed = 1](%x, %scale.3, %zero_point.3, %bit_width.3)
  %input = Gemm[alpha = 1, beta = 1, transB = 1](%onnx::Gemm_13, %onnx::Gemm_18, %features.fc0.bias)
  %inp = Relu(%input)
  %onnx::Gemm_24 = Quant[narrow = 0, rounding_mode = 'ROUND', signed = 1](%inp, %scale.7, %zero_point.7, %bit_width.7)
  %onnx::Gemm_29 = Quant[narrow = 1, rounding_mode = 'ROUND', signed = 1](%x.3, %scale.11, %zero_point.11, %bit_width.11)
  %input.3 = Gemm[alpha = 1, beta = 1, transB = 1](%onnx::Gemm_24, %onnx::Gemm_29, %features.fc1.bias)
  %inp.4 = Relu(%input.3)
  %onnx::Gemm_35 = Quant[narrow = 0, rounding_mode = 'ROUND', signed = 1](%inp.4, %scale.15, %zero_point.15, %bit_width.15)
  %onnx::Gemm_40 = Quant[narrow = 1, rounding_mode = 'ROUND', signed = 1](%x.7, %scale.19, %zero_point.19, %bit_width.19)
  %41 = Gemm[alpha = 1, beta = 1, transB = 1](%onnx::Gemm_35, %onnx::Gemm_40, %features.fc2.bias)
  return %41
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
