"""Tests for the sklearn decision trees."""

import warnings
from functools import partial

import numpy
import onnx
import pytest
from sklearn.exceptions import ConvergenceWarning

from concrete.ml.common.utils import is_model_class_in_a_list
from concrete.ml.pytest.utils import UNIQUE_MODELS_AND_DATASETS, get_model_name
from concrete.ml.sklearn import _get_sklearn_tree_models
from concrete.ml.sklearn.qnn import NeuralNetClassifier, NeuralNetRegressor

# Remark that the dump tests for torch module is directly done in test_compile_torch.py

# We are dumping long raw strings in this test, we therefore need to disable pylint from checking it
# pylint: disable=line-too-long


def check_onnx_file_dump(
    model_class, parameters, load_data, default_configuration, use_fhe_sum=False
):
    """Fit the model and dump the corresponding ONNX."""

    model_name = get_model_name(model_class)
    n_classes = parameters.get("n_classes", 2)

    # Set the model
    model = model_class()

    # Set `_fhe_ensembling` for tree based models only
    if model_class in _get_sklearn_tree_models():

        # pylint: disable=protected-access
        model._fhe_ensembling = use_fhe_sum

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
  %/_operators.0/Reshape_output_0 = Reshape[allowzero = 0](%/_operators.0/LessOrEqual_output_0, %/_operators.0/Constant_output_0)
  %/_operators.0/MatMul_output_0 = MatMul(%_operators.0.weight_2, %/_operators.0/Reshape_output_0)
  %/_operators.0/Reshape_1_output_0 = Reshape[allowzero = 0](%/_operators.0/MatMul_output_0, %/_operators.0/Constant_1_output_0)
  %/_operators.0/Equal_output_0 = Equal(%_operators.0.bias_2, %/_operators.0/Reshape_1_output_0)
  %/_operators.0/Reshape_2_output_0 = Reshape[allowzero = 0](%/_operators.0/Equal_output_0, %/_operators.0/Constant_2_output_0)
  %/_operators.0/MatMul_1_output_0 = MatMul(%_operators.0.weight_3, %/_operators.0/Reshape_2_output_0)
  %/_operators.0/Reshape_3_output_0 = Reshape[allowzero = 0](%/_operators.0/MatMul_1_output_0, %/_operators.0/Constant_3_output_0)
  """
        + (
            """%/_operators.0/ReduceSum_output_0 = ReduceSum[keepdims = 0](%/_operators.0/Reshape_3_output_0, %onnx::ReduceSum_22)
  %transposed_output = Transpose[perm = [1, 0]](%/_operators.0/ReduceSum_output_0)
  """
            if use_fhe_sum
            else "%transposed_output = Transpose[perm = [2, 1, 0]](%/_operators.0/Reshape_3_output_0)\n  "
        )
        + """return %transposed_output
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
  %/_operators.0/Reshape_output_0 = Reshape[allowzero = 0](%/_operators.0/LessOrEqual_output_0, %/_operators.0/Constant_output_0)
  %/_operators.0/MatMul_output_0 = MatMul(%_operators.0.weight_2, %/_operators.0/Reshape_output_0)
  %/_operators.0/Reshape_1_output_0 = Reshape[allowzero = 0](%/_operators.0/MatMul_output_0, %/_operators.0/Constant_1_output_0)
  %/_operators.0/Equal_output_0 = Equal(%_operators.0.bias_2, %/_operators.0/Reshape_1_output_0)
  %/_operators.0/Reshape_2_output_0 = Reshape[allowzero = 0](%/_operators.0/Equal_output_0, %/_operators.0/Constant_2_output_0)
  %/_operators.0/MatMul_1_output_0 = MatMul(%_operators.0.weight_3, %/_operators.0/Reshape_2_output_0)
  %/_operators.0/Reshape_3_output_0 = Reshape[allowzero = 0](%/_operators.0/MatMul_1_output_0, %/_operators.0/Constant_3_output_0)
  """
        + (
            """%/_operators.0/ReduceSum_output_0 = ReduceSum[keepdims = 0](%/_operators.0/Reshape_3_output_0, %onnx::ReduceSum_22)
  %transposed_output = Transpose[perm = [1, 0]](%/_operators.0/ReduceSum_output_0)
  """
            if use_fhe_sum is True
            else "%transposed_output = Transpose[perm = [2, 1, 0]](%/_operators.0/Reshape_3_output_0)\n  "
        )
        + """return %transposed_output
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
  %/_operators.0/Reshape_output_0 = Reshape[allowzero = 0](%/_operators.0/Less_output_0, %/_operators.0/Constant_output_0)
  %/_operators.0/MatMul_output_0 = MatMul(%_operators.0.weight_2, %/_operators.0/Reshape_output_0)
  %/_operators.0/Reshape_1_output_0 = Reshape[allowzero = 0](%/_operators.0/MatMul_output_0, %/_operators.0/Constant_1_output_0)
  %/_operators.0/Equal_output_0 = Equal(%_operators.0.bias_2, %/_operators.0/Reshape_1_output_0)
  %/_operators.0/Reshape_2_output_0 = Reshape[allowzero = 0](%/_operators.0/Equal_output_0, %/_operators.0/Constant_2_output_0)
  %/_operators.0/MatMul_1_output_0 = MatMul(%_operators.0.weight_3, %/_operators.0/Reshape_2_output_0)
  %/_operators.0/Reshape_3_output_0 = Reshape[allowzero = 0](%/_operators.0/MatMul_1_output_0, %/_operators.0/Constant_3_output_0)
  %/_operators.0/Squeeze_output_0 = Squeeze(%/_operators.0/Reshape_3_output_0, %axes_squeeze)
  %/_operators.0/Transpose_output_0 = Transpose[perm = [1, 0]](%/_operators.0/Squeeze_output_0)
  %/_operators.0/Reshape_4_output_0 = Reshape[allowzero = 0](%/_operators.0/Transpose_output_0, %/_operators.0/Constant_4_output_0)
  """
        + (
            """%/_operators.0/ReduceSum_output_0 = ReduceSum[keepdims = 0](%/_operators.0/Reshape_4_output_0, %onnx::ReduceSum_26)
  return %/_operators.0/ReduceSum_output_0
}"""
            if use_fhe_sum is True
            else "return %/_operators.0/Reshape_4_output_0\n}"
        ),
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
  """
        + (
            """%/_operators.0/ReduceSum_output_0 = ReduceSum[keepdims = 0](%/_operators.0/Reshape_3_output_0, %onnx::ReduceSum_22)
  %transposed_output = Transpose[perm = [1, 0]](%/_operators.0/ReduceSum_output_0)
  """
            if use_fhe_sum is True
            else "%transposed_output = Transpose[perm = [2, 1, 0]](%/_operators.0/Reshape_3_output_0)"
        )
        + """return %transposed_output
}""",
        "XGBRegressor": """graph torch_jit (
  %input_0[DOUBLE, symx10]
) initializers (
  %_operators.0.weight_1[INT64, 140x10]
  %_operators.0.bias_1[INT64, 140x1]
  %_operators.0.weight_2[INT64, 20x8x7]
  %_operators.0.bias_2[INT64, 160x1]
  %_operators.0.weight_3[INT64, 20x1x8]
  %axes_squeeze[INT64, 1]
  %/_operators.0/Constant_output_0[INT64, 3]
  %/_operators.0/Constant_1_output_0[INT64, 2]
  %/_operators.0/Constant_2_output_0[INT64, 3]
  %/_operators.0/Constant_3_output_0[INT64, 3]
  %/_operators.0/Constant_4_output_0[INT64, 3]"""
        + ("\n  %onnx::ReduceSum_27[INT64, 1]" if use_fhe_sum is True else "")
        + """
) {
  %/_operators.0/Gemm_output_0 = Gemm[alpha = 1, beta = 0, transB = 1](%_operators.0.weight_1, %input_0)
  %/_operators.0/Less_output_0 = Less(%/_operators.0/Gemm_output_0, %_operators.0.bias_1)
  %/_operators.0/Reshape_output_0 = Reshape[allowzero = 0](%/_operators.0/Less_output_0, %/_operators.0/Constant_output_0)
  %/_operators.0/MatMul_output_0 = MatMul(%_operators.0.weight_2, %/_operators.0/Reshape_output_0)
  %/_operators.0/Reshape_1_output_0 = Reshape[allowzero = 0](%/_operators.0/MatMul_output_0, %/_operators.0/Constant_1_output_0)
  %/_operators.0/Equal_output_0 = Equal(%_operators.0.bias_2, %/_operators.0/Reshape_1_output_0)
  %/_operators.0/Reshape_2_output_0 = Reshape[allowzero = 0](%/_operators.0/Equal_output_0, %/_operators.0/Constant_2_output_0)
  %/_operators.0/MatMul_1_output_0 = MatMul(%_operators.0.weight_3, %/_operators.0/Reshape_2_output_0)
  %/_operators.0/Reshape_3_output_0 = Reshape[allowzero = 0](%/_operators.0/MatMul_1_output_0, %/_operators.0/Constant_3_output_0)
  %/_operators.0/Squeeze_output_0 = Squeeze(%/_operators.0/Reshape_3_output_0, %axes_squeeze)
  %/_operators.0/Transpose_output_0 = Transpose[perm = [1, 0]](%/_operators.0/Squeeze_output_0)
  %/_operators.0/Reshape_4_output_0 = Reshape[allowzero = 0](%/_operators.0/Transpose_output_0, %/_operators.0/Constant_4_output_0)
  """
        + (
            """%/_operators.0/ReduceSum_output_0 = ReduceSum[keepdims = 0](%/_operators.0/Reshape_4_output_0, %onnx::ReduceSum_27)
  return %/_operators.0/ReduceSum_output_0
}"""
            if use_fhe_sum is True
            else """return %/_operators.0/Reshape_4_output_0\n}"""
        ),
        "LinearRegression": """graph torch_jit (
  %input_0[DOUBLE, symx10]
) initializers (
  %_operators.0.coefficients[FLOAT, 10x2]
  %_operators.0.intercepts[FLOAT, 2]
) {
  %variable = Gemm[alpha = 1, beta = 1](%input_0, %_operators.0.coefficients, %_operators.0.intercepts)
  return %variable
}""",
        "SGDRegressor": """graph torch_jit (
  %input_0[DOUBLE, symx10]
) initializers (
  %_operators.0.coefficients[FLOAT, 10x1]
  %_operators.0.intercepts[FLOAT, 1]
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
        "KNeighborsClassifier": """graph torch_jit (
  %input_0[DOUBLE, symx2]
) {
  %/_operators.0/Constant_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Unsqueeze_output_0 = Unsqueeze(%input_0, %/_operators.0/Constant_output_0)
  %/_operators.0/Constant_1_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Sub_output_0 = Sub(%/_operators.0/Unsqueeze_output_0, %onnx::Sub_46)
  %/_operators.0/Constant_2_output_0 = Constant[value = <Scalar Tensor []>]()
  %/_operators.0/Pow_output_0 = Pow(%/_operators.0/Sub_output_0, %/_operators.0/Constant_2_output_0)
  %/_operators.0/Constant_3_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/ReduceSum_output_0 = ReduceSum[keepdims = 0, noop_with_empty_axes = 0](%/_operators.0/Pow_output_0, %/_operators.0/Constant_3_output_0)
  %/_operators.0/Pow_1_output_0 = Pow(%/_operators.0/ReduceSum_output_0, %/_operators.0/Constant_1_output_0)
  %/_operators.0/Constant_4_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/TopK_output_0, %/_operators.0/TopK_output_1 = TopK[axis = 1, largest = 0, sorted = 1](%/_operators.0/Pow_1_output_0, %/_operators.0/Constant_4_output_0)
  %/_operators.0/Constant_5_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Reshape_output_0 = Reshape[allowzero = 0](%/_operators.0/TopK_output_1, %/_operators.0/Constant_5_output_0)
  %/_operators.0/Gather_output_0 = Gather[axis = 0](%_operators.0.train_labels, %/_operators.0/Reshape_output_0)
  %/_operators.0/Shape_output_0 = Shape(%/_operators.0/TopK_output_1)
  %/_operators.0/ConstantOfShape_output_0 = ConstantOfShape[value = <Tensor>](%/_operators.0/Shape_output_0)
  %/_operators.0/Constant_6_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Reshape_1_output_0 = Reshape[allowzero = 0](%/_operators.0/Gather_output_0, %/_operators.0/Constant_6_output_0)
  %/_operators.0/Constant_7_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/ScatterElements_output_0 = ScatterElements[axis = 1](%/_operators.0/Constant_7_output_0, %/_operators.0/Reshape_1_output_0, %/_operators.0/ConstantOfShape_output_0)
  %/_operators.0/Constant_8_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Add_output_0 = Add(%/_operators.0/Constant_8_output_0, %/_operators.0/ScatterElements_output_0)
  %onnx::ReduceSum_36 = Constant[value = <Tensor>]()
  %/_operators.0/ReduceSum_1_output_0 = ReduceSum[keepdims = 1](%/_operators.0/Add_output_0, %onnx::ReduceSum_36)
  %/_operators.0/Constant_9_output_0 = Constant[value = <Scalar Tensor []>]()
  %/_operators.0/Equal_output_0 = Equal(%/_operators.0/ReduceSum_1_output_0, %/_operators.0/Constant_9_output_0)
  %/_operators.0/Constant_10_output_0 = Constant[value = <Tensor>]()
  %/_operators.0/Where_output_0 = Where(%/_operators.0/Equal_output_0, %/_operators.0/Constant_10_output_0, %/_operators.0/ReduceSum_1_output_0)
  %/_operators.0/Constant_11_output_0 = Constant[value = <Scalar Tensor []>]()
  %/_operators.0/Pow_2_output_0 = Pow(%/_operators.0/Where_output_0, %/_operators.0/Constant_11_output_0)
  %onnx::ArgMax_44 = Mul(%/_operators.0/Pow_2_output_0, %/_operators.0/Add_output_0)
  %variable = ArgMax[axis = 1, keepdims = 0, select_last_index = 0](%onnx::ArgMax_44)
  return %variable, %onnx::ArgMax_44
}""",
        "SGDClassifier": "graph torch_jit (\n  %input_0[DOUBLE, symx10]\n) initializers (\n  %_operators.0.coefficients[FLOAT, 10x1]\n  %_operators.0.intercepts[FLOAT, 1]\n) {\n  %/_operators.0/Gemm_output_0 = Gemm[alpha = 1, beta = 1](%input_0, %_operators.0.coefficients, %_operators.0.intercepts)\n  return %/_operators.0/Gemm_output_0\n}",
    }

    str_expected = expected_strings.get(model_name, "")

    # Get the data-set. The data generation is seeded in load_data.
    x, y = load_data(model_class, **parameters)

    model_params = model.get_params()
    if "random_state" in model_params:
        model_params["random_state"] = numpy.random.randint(0, 2**15)

        model.set_params(**model_params)

    if model_name == "KNeighborsClassifier":
        # KNN can only be compiled with small quantization bit numbers for now
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3979
        model.n_bits = 2

    with warnings.catch_warnings():
        # Sometimes, we miss convergence, which is not a problem for our test
        warnings.simplefilter("ignore", category=ConvergenceWarning)

        model.fit(x, y)

    with warnings.catch_warnings():
        # Use FHE simulation to not have issues with precision
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
        "KNeighborsClassifier",
    ]:
        while len(onnx_model.graph.initializer) > 0:
            del onnx_model.graph.initializer[0]

    str_model = onnx.helper.printable_graph(onnx_model.graph)
    print(f"\nCurrent {model_name=}:\n{str_model}")
    print(f"\nExpected {model_name=}:\n{str_expected}")

    # Test equality when it does not depend on seeds
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3266
    if not is_model_class_in_a_list(model_class, _get_sklearn_tree_models(select="RandomForest")):
        # The expected graph is usually a string and we therefore directly test if it is equal to
        # the retrieved graph's string. However, in some cases such as for TweedieRegressor models,
        # this graph can slightly changed depending on some input's values. We then expected the
        # string to match as least one of them expected strings (as a list)
        if isinstance(str_expected, str):
            assert str_model == str_expected
        else:
            assert str_model in str_expected


@pytest.mark.parametrize("model_class, parameters", UNIQUE_MODELS_AND_DATASETS)
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

    if model_name == "NeuralNetClassifier":
        model_class = partial(
            NeuralNetClassifier,
            module__n_layers=3,
            module__power_of_two_scaling=False,
            max_epochs=1,
            verbose=0,
            callbacks="disable",
        )
    elif model_name == "NeuralNetRegressor":
        model_class = partial(
            NeuralNetRegressor,
            module__n_layers=3,
            module__n_w_bits=2,
            module__n_a_bits=2,
            module__n_accum_bits=7,  # Stay with 7 bits for test exec time
            module__n_hidden_neurons_multiplier=1,
            module__power_of_two_scaling=False,
            max_epochs=1,
            verbose=0,
            callbacks="disable",
        )

    check_onnx_file_dump(model_class, parameters, load_data, default_configuration)

    # Additional tests exclusively dedicated for tree ensemble models.
    if model_class in _get_sklearn_tree_models():
        check_onnx_file_dump(
            model_class, parameters, load_data, default_configuration, use_fhe_sum=True
        )
