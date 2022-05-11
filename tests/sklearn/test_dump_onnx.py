"""Tests for the sklearn decision trees."""
import warnings
from functools import partial

import numpy
import onnx
import pytest
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning

from concrete.ml.sklearn import (
    DecisionTreeClassifier,
    GammaRegressor,
    LinearRegression,
    LinearSVC,
    LinearSVR,
    LogisticRegression,
    PoissonRegressor,
    RandomForestClassifier,
    TweedieRegressor,
    XGBClassifier,
)


def check_onnx_file_dump(algo, algo_name, load_data, str_expected, default_configuration):
    """Fit the model and dump the corresponding ONNX."""

    # Get the dataset
    x, y = load_data()

    if algo is GammaRegressor:
        y = numpy.abs(y) + 1

    # Set the model
    model = algo(
        n_bits=6,
    )

    model_params = model.get_params()
    if "random_state" in model_params:
        model_params["random_state"] = numpy.random.randint(0, 2**15)

    model.set_params(**model_params)

    # Sometimes, we miss convergence, which is not a problem for our test
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(x, y)

    # Use virtual lib to not have issues with precision
    model.compile(x, default_configuration, use_virtual_lib=True)

    # Get ONNX model
    onnx_model = model.get_onnx()

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

    check_onnx_file_dump(algo, algo_name, load_data, str_expected, default_configuration)


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
    ],
)
@pytest.mark.parametrize(
    "load_data",
    [
        pytest.param(
            lambda n_classes: make_classification(
                n_samples=100,
                n_features=20,
                n_classes=n_classes,
                n_informative=10,
                random_state=numpy.random.randint(0, 2**15),
            ),
            id="make_classification",
        )
    ],
)
def test_linear_models(
    algo,
    load_data,
    str_expected,
    default_configuration,
):
    """Tests the linear models."""
    algo_name = algo.__name__

    n_classes = 4 if algo is not LogisticRegression else 2

    def load_data_with_n_classes():
        """Fix n_classes."""
        return load_data(n_classes)

    check_onnx_file_dump(
        algo, algo_name, load_data_with_n_classes, str_expected, default_configuration
    )
