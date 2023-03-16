"""Tests for the sklearn decision trees."""


import warnings

import numpy
import onnx
import pytest
from sklearn.exceptions import ConvergenceWarning

from concrete.ml.common.utils import is_model_class_in_a_list
from concrete.ml.pytest.utils import get_model_name, sklearn_models_and_datasets
from concrete.ml.sklearn import get_sklearn_tree_models

# Remark that the dump tests for torch module is directly done in test_compile_torch.py

# We are dumping long raw strings in this test, we therefore need to disable pylint from checking it
# pylint: disable=line-too-long


def check_onnx_file_dump(model_class, parameters, load_data, str_expected, default_configuration):
    """Fit the model and dump the corresponding ONNX."""

    # Get the dataset. The data generation is seeded in load_data.
    x, y = load_data(model_class, **parameters)

    # Set the model
    model = model_class(n_bits=6)

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
        model.compile(x, default_configuration)

    # Get ONNX model
    onnx_model = model.onnx_model

    # Remove initializers, since they change from one seed to the other
    model_name = get_model_name(model_class)
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
    if not is_model_class_in_a_list(
        model_class, get_sklearn_tree_models(str_in_class_name="RandomForest")
    ):
        # The expected graph is usually a string and we therefore directly test if it is equal to
        # the retrieved graph's string. However, in some cases such as for TweedieRegressor models,
        # this graph can slightly changed depending on some input's values. We then expected the
        # string to match as least one of them expected strings (as a list)
        if isinstance(str_expected, str):
            assert str_model == str_expected
        else:
            assert str_model in str_expected


@pytest.mark.parametrize("model_class, parameters", sklearn_models_and_datasets)
def test_dump(
    model_class,
    parameters,
    load_data,
    default_configuration,
):
    """Tests dump."""

    model_name = get_model_name(model_class)

    # Some models have been done with different n_classes which create different ONNX
    if parameters.get("n_classes", 2) != 2 and model_name in ["LinearSVC", "LogisticRegression"]:
        return

    n_classes = parameters.get("n_classes", 2)

    # Ignore long lines here
    # ruff: noqa: E501
    expected_strings = {
        "LinearSVC": """graph torch_jit (
  %input_0[DOUBLE, symx10]
) initializers (
  %_operators.0.coefficients[FLOAT, 10x1]
  %_operators.0.intercepts[FLOAT, 1]
) {
  %/_operators.0/Gemm_output_0 = Gemm[alpha = 1, beta = 1](%input_0, %_operators.0.coefficients, %_operators.0.intercepts)
  return %/_operators.0/Gemm_output_0
}""",
        "NeuralNetClassifier": """graph torch_jit (
  %inp.1[FLOAT, 1x10]
) initializers (
  %features.fc0.bias[FLOAT, 40]
  %features.fc1.bias[FLOAT, 40]
  """
        + f"%features.fc2.bias[FLOAT, {n_classes}]"
        + """
  %/features/quant0/act_quant/export_handler/Constant_output_0[FLOAT, scalar]
  %/features/quant0/act_quant/export_handler/Constant_1_output_0[FLOAT, scalar]
  %/features/quant0/act_quant/export_handler/Constant_2_output_0[FLOAT, scalar]
  %/features/fc0/weight_quant/export_handler/Constant_output_0[FLOAT, 40x10]
  %/features/fc0/weight_quant/export_handler/Constant_1_output_0[FLOAT, scalar]
  %/features/quant1/act_quant/export_handler/Constant_output_0[FLOAT, scalar]
  %/features/fc1/weight_quant/export_handler/Constant_output_0[FLOAT, 40x40]
  %/features/fc1/weight_quant/export_handler/Constant_1_output_0[FLOAT, scalar]
  %/features/quant2/act_quant/export_handler/Constant_output_0[FLOAT, scalar]
  """
        + f"%/features/fc2/weight_quant/export_handler/Constant_output_0[FLOAT, {n_classes}x40]"
        + """
  %/features/fc2/weight_quant/export_handler/Constant_1_output_0[FLOAT, scalar]
) {
  %/features/quant0/act_quant/export_handler/Quant_output_0 = Quant[narrow = 0, rounding_mode = 'ROUND', signed = 1](%inp.1, %/features/quant0/act_quant/export_handler/Constant_1_output_0, %/features/quant0/act_quant/export_handler/Constant_2_output_0, %/features/quant0/act_quant/export_handler/Constant_output_0)
  %/features/fc0/weight_quant/export_handler/Quant_output_0 = Quant[narrow = 0, rounding_mode = 'ROUND', signed = 1](%/features/fc0/weight_quant/export_handler/Constant_output_0, %/features/fc0/weight_quant/export_handler/Constant_1_output_0, %/features/quant0/act_quant/export_handler/Constant_2_output_0, %/features/quant0/act_quant/export_handler/Constant_output_0)
  %/features/fc0/Gemm_output_0 = Gemm[alpha = 1, beta = 1, transB = 1](%/features/quant0/act_quant/export_handler/Quant_output_0, %/features/fc0/weight_quant/export_handler/Quant_output_0, %features.fc0.bias)
  %/features/act0/Relu_output_0 = Relu(%/features/fc0/Gemm_output_0)
  %/features/quant1/act_quant/export_handler/Quant_output_0 = Quant[narrow = 0, rounding_mode = 'ROUND', signed = 1](%/features/act0/Relu_output_0, %/features/quant1/act_quant/export_handler/Constant_output_0, %/features/quant0/act_quant/export_handler/Constant_2_output_0, %/features/quant0/act_quant/export_handler/Constant_output_0)
  %/features/fc1/weight_quant/export_handler/Quant_output_0 = Quant[narrow = 0, rounding_mode = 'ROUND', signed = 1](%/features/fc1/weight_quant/export_handler/Constant_output_0, %/features/fc1/weight_quant/export_handler/Constant_1_output_0, %/features/quant0/act_quant/export_handler/Constant_2_output_0, %/features/quant0/act_quant/export_handler/Constant_output_0)
  %/features/fc1/Gemm_output_0 = Gemm[alpha = 1, beta = 1, transB = 1](%/features/quant1/act_quant/export_handler/Quant_output_0, %/features/fc1/weight_quant/export_handler/Quant_output_0, %features.fc1.bias)
  %/features/act1/Relu_output_0 = Relu(%/features/fc1/Gemm_output_0)
  %/features/quant2/act_quant/export_handler/Quant_output_0 = Quant[narrow = 0, rounding_mode = 'ROUND', signed = 1](%/features/act1/Relu_output_0, %/features/quant2/act_quant/export_handler/Constant_output_0, %/features/quant0/act_quant/export_handler/Constant_2_output_0, %/features/quant0/act_quant/export_handler/Constant_output_0)
  %/features/fc2/weight_quant/export_handler/Quant_output_0 = Quant[narrow = 0, rounding_mode = 'ROUND', signed = 1](%/features/fc2/weight_quant/export_handler/Constant_output_0, %/features/fc2/weight_quant/export_handler/Constant_1_output_0, %/features/quant0/act_quant/export_handler/Constant_2_output_0, %/features/quant0/act_quant/export_handler/Constant_output_0)
  %31 = Gemm[alpha = 1, beta = 1, transB = 1](%/features/quant2/act_quant/export_handler/Quant_output_0, %/features/fc2/weight_quant/export_handler/Quant_output_0, %features.fc2.bias)
  return %31
}""",
        "NeuralNetRegressor": """graph torch_jit (
  %inp.1[FLOAT, 1x10]
) initializers (
  %features.fc0.bias[FLOAT, 10]
  %features.fc1.bias[FLOAT, 10]
  %features.fc2.bias[FLOAT, 1]
  %/features/quant0/act_quant/export_handler/Constant_output_0[FLOAT, scalar]
  %/features/quant0/act_quant/export_handler/Constant_1_output_0[FLOAT, scalar]
  %/features/quant0/act_quant/export_handler/Constant_2_output_0[FLOAT, scalar]
  %/features/fc0/weight_quant/export_handler/Constant_output_0[FLOAT, 10x10]
  %/features/fc0/weight_quant/export_handler/Constant_1_output_0[FLOAT, scalar]
  %/features/quant1/act_quant/export_handler/Constant_output_0[FLOAT, scalar]
  %/features/fc1/weight_quant/export_handler/Constant_output_0[FLOAT, 10x10]
  %/features/fc1/weight_quant/export_handler/Constant_1_output_0[FLOAT, scalar]
  %/features/quant2/act_quant/export_handler/Constant_output_0[FLOAT, scalar]
  %/features/fc2/weight_quant/export_handler/Constant_output_0[FLOAT, 1x10]
  %/features/fc2/weight_quant/export_handler/Constant_1_output_0[FLOAT, scalar]
) {
  %/features/quant0/act_quant/export_handler/Quant_output_0 = Quant[narrow = 0, rounding_mode = 'ROUND', signed = 1](%inp.1, %/features/quant0/act_quant/export_handler/Constant_1_output_0, %/features/quant0/act_quant/export_handler/Constant_2_output_0, %/features/quant0/act_quant/export_handler/Constant_output_0)
  %/features/fc0/weight_quant/export_handler/Quant_output_0 = Quant[narrow = 0, rounding_mode = 'ROUND', signed = 1](%/features/fc0/weight_quant/export_handler/Constant_output_0, %/features/fc0/weight_quant/export_handler/Constant_1_output_0, %/features/quant0/act_quant/export_handler/Constant_2_output_0, %/features/quant0/act_quant/export_handler/Constant_output_0)
  %/features/fc0/Gemm_output_0 = Gemm[alpha = 1, beta = 1, transB = 1](%/features/quant0/act_quant/export_handler/Quant_output_0, %/features/fc0/weight_quant/export_handler/Quant_output_0, %features.fc0.bias)
  %/features/act0/Relu_output_0 = Relu(%/features/fc0/Gemm_output_0)
  %/features/quant1/act_quant/export_handler/Quant_output_0 = Quant[narrow = 0, rounding_mode = 'ROUND', signed = 1](%/features/act0/Relu_output_0, %/features/quant1/act_quant/export_handler/Constant_output_0, %/features/quant0/act_quant/export_handler/Constant_2_output_0, %/features/quant0/act_quant/export_handler/Constant_output_0)
  %/features/fc1/weight_quant/export_handler/Quant_output_0 = Quant[narrow = 0, rounding_mode = 'ROUND', signed = 1](%/features/fc1/weight_quant/export_handler/Constant_output_0, %/features/fc1/weight_quant/export_handler/Constant_1_output_0, %/features/quant0/act_quant/export_handler/Constant_2_output_0, %/features/quant0/act_quant/export_handler/Constant_output_0)
  %/features/fc1/Gemm_output_0 = Gemm[alpha = 1, beta = 1, transB = 1](%/features/quant1/act_quant/export_handler/Quant_output_0, %/features/fc1/weight_quant/export_handler/Quant_output_0, %features.fc1.bias)
  %/features/act1/Relu_output_0 = Relu(%/features/fc1/Gemm_output_0)
  %/features/quant2/act_quant/export_handler/Quant_output_0 = Quant[narrow = 0, rounding_mode = 'ROUND', signed = 1](%/features/act1/Relu_output_0, %/features/quant2/act_quant/export_handler/Constant_output_0, %/features/quant0/act_quant/export_handler/Constant_2_output_0, %/features/quant0/act_quant/export_handler/Constant_output_0)
  %/features/fc2/weight_quant/export_handler/Quant_output_0 = Quant[narrow = 0, rounding_mode = 'ROUND', signed = 1](%/features/fc2/weight_quant/export_handler/Constant_output_0, %/features/fc2/weight_quant/export_handler/Constant_1_output_0, %/features/quant0/act_quant/export_handler/Constant_2_output_0, %/features/quant0/act_quant/export_handler/Constant_output_0)
  %31 = Gemm[alpha = 1, beta = 1, transB = 1](%/features/quant2/act_quant/export_handler/Quant_output_0, %/features/fc2/weight_quant/export_handler/Quant_output_0, %features.fc2.bias)
  return %31
}""",
        "LogisticRegression": """graph torch_jit (
  %input_0[DOUBLE, symx10]
) initializers (
  %_operators.0.coefficients[FLOAT, 10x1]
  %_operators.0.intercepts[FLOAT, 1]
) {
  %/_operators.0/Gemm_output_0 = Gemm[alpha = 1, beta = 1](%input_0, %_operators.0.coefficients, %_operators.0.intercepts)
  return %/_operators.0/Gemm_output_0
}""",
        "DecisionTreeRegressor": """graph torch_jit (
  %input_0[DOUBLE, symx10]
) {
  %/_operators.0/Gemm_output_0 = Gemm[alpha = 1, beta = 0, transB = 1](%_operators.0.weight_1, %input_0)
  %/_operators.0/LessOrEqual_output_0 = LessOrEqual(%/_operators.0/Gemm_output_0, %_operators.0.bias_1)
  %/_operators.0/Constant_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Reshape_output_0 = Reshape[allowzero = 0](%/_operators.0/LessOrEqual_output_0, %/_operators.0/Constant_output_0)
  %/_operators.0/MatMul_output_0 = MatMul(%_operators.0.weight_2, %/_operators.0/Reshape_output_0)
  %/_operators.0/Constant_1_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Reshape_1_output_0 = Reshape[allowzero = 0](%/_operators.0/MatMul_output_0, %/_operators.0/Constant_1_output_0)
  %/_operators.0/Equal_output_0 = Equal(%_operators.0.bias_2, %/_operators.0/Reshape_1_output_0)
  %/_operators.0/Constant_2_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Reshape_2_output_0 = Reshape[allowzero = 0](%/_operators.0/Equal_output_0, %/_operators.0/Constant_2_output_0)
  %/_operators.0/MatMul_1_output_0 = MatMul(%_operators.0.weight_3, %/_operators.0/Reshape_2_output_0)
  %/_operators.0/Constant_3_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Reshape_3_output_0 = Reshape[allowzero = 0](%/_operators.0/MatMul_1_output_0, %/_operators.0/Constant_3_output_0)
  %transposed_output = Transpose[perm = [2, 1, 0]](%/_operators.0/Reshape_3_output_0)
  return %transposed_output
}""",
        "RandomForestClassifier": """graph torch_jit (
  %input_0[DOUBLE, symx10]
) {
  %/_operators.0/Gemm_output_0 = Gemm[alpha = 1, beta = 0, transB = 1](%_operators.0.weight_1, %input_0)
  %/_operators.0/LessOrEqual_output_0 = LessOrEqual(%/_operators.0/Gemm_output_0, %_operators.0.bias_1)
  %/_operators.0/Constant_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Reshape_output_0 = Reshape[allowzero = 0](%/_operators.0/LessOrEqual_output_0, %/_operators.0/Constant_output_0)
  %/_operators.0/MatMul_output_0 = MatMul(%_operators.0.weight_2, %/_operators.0/Reshape_output_0)
  %/_operators.0/Constant_1_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Reshape_1_output_0 = Reshape[allowzero = 0](%/_operators.0/MatMul_output_0, %/_operators.0/Constant_1_output_0)
  %/_operators.0/Equal_output_0 = Equal(%_operators.0.bias_2, %/_operators.0/Reshape_1_output_0)
  %/_operators.0/Constant_2_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Reshape_2_output_0 = Reshape[allowzero = 0](%/_operators.0/Equal_output_0, %/_operators.0/Constant_2_output_0)
  %/_operators.0/MatMul_1_output_0 = MatMul(%_operators.0.weight_3, %/_operators.0/Reshape_2_output_0)
  %/_operators.0/Constant_3_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Reshape_3_output_0 = Reshape[allowzero = 0](%/_operators.0/MatMul_1_output_0, %/_operators.0/Constant_3_output_0)
  %transposed_output = Transpose[perm = [2, 1, 0]](%/_operators.0/Reshape_3_output_0)
  return %transposed_output
}""",
        "PoissonRegressor": """graph torch_jit (
  %input_0[DOUBLE, symx10]
) initializers (
  %_operators.0.coefficients[FLOAT, 10x1]
  %_operators.0.intercepts[FLOAT, 1]
) {
  %/_operators.0/Gemm_output_0 = Gemm[alpha = 1, beta = 1](%input_0, %_operators.0.coefficients, %_operators.0.intercepts)
  return %/_operators.0/Gemm_output_0
}""",
        "TweedieRegressor": [
            """graph torch_jit (
  %input_0[DOUBLE, symx10]
) initializers (
  %_operators.0.coefficients[FLOAT, 10x1]
  %_operators.0.intercepts[FLOAT, 1]
) {
  %/_operators.0/Gemm_output_0 = Gemm[alpha = 1, beta = 1](%input_0, %_operators.0.coefficients, %_operators.0.intercepts)
  return %/_operators.0/Gemm_output_0
}""",
            """graph torch_jit (
  %input_0[DOUBLE, symx10]
) initializers (
  %_operators.0.coefficients[FLOAT, 10x1]
  %_operators.0.intercepts[FLOAT, 1]
) {
  %variable = Gemm[alpha = 1, beta = 1](%input_0, %_operators.0.coefficients, %_operators.0.intercepts)
  return %variable
}""",
        ],
        "Ridge": """graph torch_jit (
  %input_0[DOUBLE, symx10]
) initializers (
  %_operators.0.coefficients[FLOAT, 10x1]
  %_operators.0.intercepts[FLOAT, 1]
) {
  %variable = Gemm[alpha = 1, beta = 1](%input_0, %_operators.0.coefficients, %_operators.0.intercepts)
  return %variable
}""",
        "DecisionTreeClassifier": """graph torch_jit (
  %input_0[DOUBLE, symx10]
) {
  %/_operators.0/Gemm_output_0 = Gemm[alpha = 1, beta = 0, transB = 1](%_operators.0.weight_1, %input_0)
  %/_operators.0/LessOrEqual_output_0 = LessOrEqual(%/_operators.0/Gemm_output_0, %_operators.0.bias_1)
  %/_operators.0/Constant_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Reshape_output_0 = Reshape[allowzero = 0](%/_operators.0/LessOrEqual_output_0, %/_operators.0/Constant_output_0)
  %/_operators.0/MatMul_output_0 = MatMul(%_operators.0.weight_2, %/_operators.0/Reshape_output_0)
  %/_operators.0/Constant_1_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Reshape_1_output_0 = Reshape[allowzero = 0](%/_operators.0/MatMul_output_0, %/_operators.0/Constant_1_output_0)
  %/_operators.0/Equal_output_0 = Equal(%_operators.0.bias_2, %/_operators.0/Reshape_1_output_0)
  %/_operators.0/Constant_2_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Reshape_2_output_0 = Reshape[allowzero = 0](%/_operators.0/Equal_output_0, %/_operators.0/Constant_2_output_0)
  %/_operators.0/MatMul_1_output_0 = MatMul(%_operators.0.weight_3, %/_operators.0/Reshape_2_output_0)
  %/_operators.0/Constant_3_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Reshape_3_output_0 = Reshape[allowzero = 0](%/_operators.0/MatMul_1_output_0, %/_operators.0/Constant_3_output_0)
  %transposed_output = Transpose[perm = [2, 1, 0]](%/_operators.0/Reshape_3_output_0)
  return %transposed_output
}""",
        "GammaRegressor": """graph torch_jit (
  %input_0[DOUBLE, symx10]
) initializers (
  %_operators.0.coefficients[FLOAT, 10x1]
  %_operators.0.intercepts[FLOAT, 1]
) {
  %/_operators.0/Gemm_output_0 = Gemm[alpha = 1, beta = 1](%input_0, %_operators.0.coefficients, %_operators.0.intercepts)
  return %/_operators.0/Gemm_output_0
}""",
        "ElasticNet": """graph torch_jit (
  %input_0[DOUBLE, symx10]
) initializers (
  %_operators.0.coefficients[FLOAT, 10x1]
  %_operators.0.intercepts[FLOAT, 1]
) {
  %variable = Gemm[alpha = 1, beta = 1](%input_0, %_operators.0.coefficients, %_operators.0.intercepts)
  return %variable
}""",
        "LinearSVR": """graph torch_jit (
  %input_0[DOUBLE, symx10]
) initializers (
  %_operators.0.coefficients[FLOAT, 10x1]
  %_operators.0.intercepts[FLOAT, 1]
) {
  %variable = Gemm[alpha = 1, beta = 1](%input_0, %_operators.0.coefficients, %_operators.0.intercepts)
  return %variable
}""",
        "XGBClassifier": """graph torch_jit (
  %input_0[DOUBLE, symx10]
) {
  %/_operators.0/Gemm_output_0 = Gemm[alpha = 1, beta = 0, transB = 1](%_operators.0.weight_1, %input_0)
  %/_operators.0/Less_output_0 = Less(%/_operators.0/Gemm_output_0, %_operators.0.bias_1)
  %/_operators.0/Constant_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Reshape_output_0 = Reshape[allowzero = 0](%/_operators.0/Less_output_0, %/_operators.0/Constant_output_0)
  %/_operators.0/MatMul_output_0 = MatMul(%_operators.0.weight_2, %/_operators.0/Reshape_output_0)
  %/_operators.0/Constant_1_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Reshape_1_output_0 = Reshape[allowzero = 0](%/_operators.0/MatMul_output_0, %/_operators.0/Constant_1_output_0)
  %/_operators.0/Equal_output_0 = Equal(%_operators.0.bias_2, %/_operators.0/Reshape_1_output_0)
  %/_operators.0/Constant_2_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Reshape_2_output_0 = Reshape[allowzero = 0](%/_operators.0/Equal_output_0, %/_operators.0/Constant_2_output_0)
  %/_operators.0/MatMul_1_output_0 = MatMul(%_operators.0.weight_3, %/_operators.0/Reshape_2_output_0)
  %/_operators.0/Constant_3_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Reshape_3_output_0 = Reshape[allowzero = 0](%/_operators.0/MatMul_1_output_0, %/_operators.0/Constant_3_output_0)
  %/_operators.0/Squeeze_output_0 = Squeeze(%/_operators.0/Reshape_3_output_0, %axes_squeeze)
  %/_operators.0/Transpose_output_0 = Transpose[perm = [1, 0]](%/_operators.0/Squeeze_output_0)
  %/_operators.0/Constant_4_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Reshape_4_output_0 = Reshape[allowzero = 0](%/_operators.0/Transpose_output_0, %/_operators.0/Constant_4_output_0)
  return %/_operators.0/Reshape_4_output_0
}""",
        "RandomForestRegressor": """graph torch_jit (
  %input_0[DOUBLE, symx10]
) {
  %/_operators.0/Gemm_output_0 = Gemm[alpha = 1, beta = 0, transB = 1](%_operators.0.weight_1, %input_0)
  %/_operators.0/LessOrEqual_output_0 = LessOrEqual(%/_operators.0/Gemm_output_0, %_operators.0.bias_1)
  %/_operators.0/Constant_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Reshape_output_0 = Reshape[allowzero = 0](%/_operators.0/LessOrEqual_output_0, %/_operators.0/Constant_output_0)
  %/_operators.0/MatMul_output_0 = MatMul(%_operators.0.weight_2, %/_operators.0/Reshape_output_0)
  %/_operators.0/Constant_1_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Reshape_1_output_0 = Reshape[allowzero = 0](%/_operators.0/MatMul_output_0, %/_operators.0/Constant_1_output_0)
  %/_operators.0/Equal_output_0 = Equal(%_operators.0.bias_2, %/_operators.0/Reshape_1_output_0)
  %/_operators.0/Constant_2_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Reshape_2_output_0 = Reshape[allowzero = 0](%/_operators.0/Equal_output_0, %/_operators.0/Constant_2_output_0)
  %/_operators.0/MatMul_1_output_0 = MatMul(%_operators.0.weight_3, %/_operators.0/Reshape_2_output_0)
  %/_operators.0/Constant_3_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Reshape_3_output_0 = Reshape[allowzero = 0](%/_operators.0/MatMul_1_output_0, %/_operators.0/Constant_3_output_0)
  %transposed_output = Transpose[perm = [2, 1, 0]](%/_operators.0/Reshape_3_output_0)
  return %transposed_output
}""",
        "XGBRegressor": """graph torch_jit (
  %input_0[DOUBLE, symx10]
) initializers (
  %_operators.0.base_prediction[INT64, 1]
  %_operators.0.weight_1[INT64, 140x10]
  %_operators.0.bias_1[INT64, 140x1]
  %_operators.0.weight_2[INT64, 20x8x7]
  %_operators.0.bias_2[INT64, 160x1]
  %_operators.0.weight_3[INT64, 20x1x8]
  %axes_squeeze[INT64, 1]
) {
  %/_operators.0/Gemm_output_0 = Gemm[alpha = 1, beta = 0, transB = 1](%_operators.0.weight_1, %input_0)
  %/_operators.0/Less_output_0 = Less(%/_operators.0/Gemm_output_0, %_operators.0.bias_1)
  %/_operators.0/Constant_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Reshape_output_0 = Reshape[allowzero = 0](%/_operators.0/Less_output_0, %/_operators.0/Constant_output_0)
  %/_operators.0/MatMul_output_0 = MatMul(%_operators.0.weight_2, %/_operators.0/Reshape_output_0)
  %/_operators.0/Constant_1_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Reshape_1_output_0 = Reshape[allowzero = 0](%/_operators.0/MatMul_output_0, %/_operators.0/Constant_1_output_0)
  %/_operators.0/Equal_output_0 = Equal(%_operators.0.bias_2, %/_operators.0/Reshape_1_output_0)
  %/_operators.0/Constant_2_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Reshape_2_output_0 = Reshape[allowzero = 0](%/_operators.0/Equal_output_0, %/_operators.0/Constant_2_output_0)
  %/_operators.0/MatMul_1_output_0 = MatMul(%_operators.0.weight_3, %/_operators.0/Reshape_2_output_0)
  %/_operators.0/Constant_3_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Reshape_3_output_0 = Reshape[allowzero = 0](%/_operators.0/MatMul_1_output_0, %/_operators.0/Constant_3_output_0)
  %/_operators.0/Squeeze_output_0 = Squeeze(%/_operators.0/Reshape_3_output_0, %axes_squeeze)
  %/_operators.0/Transpose_output_0 = Transpose[perm = [1, 0]](%/_operators.0/Squeeze_output_0)
  %/_operators.0/Constant_4_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Reshape_4_output_0 = Reshape[allowzero = 0](%/_operators.0/Transpose_output_0, %/_operators.0/Constant_4_output_0)
  return %/_operators.0/Reshape_4_output_0
}""",
        "LinearRegression": """graph torch_jit (
  %input_0[DOUBLE, symx10]
) initializers (
  %_operators.0.coefficients[FLOAT, 10x2]
  %_operators.0.intercepts[FLOAT, 2]
) {
  %variable = Gemm[alpha = 1, beta = 1](%input_0, %_operators.0.coefficients, %_operators.0.intercepts)
  return %variable
}""",
        "Lasso": """graph torch_jit (
  %input_0[DOUBLE, symx10]
) initializers (
  %_operators.0.coefficients[FLOAT, 10x1]
  %_operators.0.intercepts[FLOAT, 1]
) {
  %variable = Gemm[alpha = 1, beta = 1](%input_0, %_operators.0.coefficients, %_operators.0.intercepts)
  return %variable
}""",
    }

    str_expected = expected_strings[model_name]
    check_onnx_file_dump(model_class, parameters, load_data, str_expected, default_configuration)
