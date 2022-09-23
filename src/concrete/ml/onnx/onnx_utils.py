"""Utils to interpret an ONNX model with numpy."""
# Utils to interpret an ONNX model with numpy.


# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications copyright (C) 2022 Zama:
# - variable renaming
# - streamlining of some functions
# - mypy typing hints
# - existing and new ops implementation in separate file

# Original file:
# https://github.com/google/jax/blob/f6d329b2d9b5f83c6a59e5739aa1ca8d4d1ffa1c/examples/onnx2xla.py


from typing import Any, Callable, Dict, Tuple

import numpy
import onnx
from onnx import numpy_helper

from .ops_impl import (
    numpy_abs,
    numpy_acos,
    numpy_acosh,
    numpy_add,
    numpy_asin,
    numpy_asinh,
    numpy_atan,
    numpy_atanh,
    numpy_batchnorm,
    numpy_brevitas_quant,
    numpy_cast,
    numpy_celu,
    numpy_clip,
    numpy_constant,
    numpy_cos,
    numpy_cosh,
    numpy_div,
    numpy_elu,
    numpy_equal,
    numpy_erf,
    numpy_exp,
    numpy_flatten,
    numpy_gemm,
    numpy_greater,
    numpy_greater_float,
    numpy_greater_or_equal,
    numpy_greater_or_equal_float,
    numpy_hardsigmoid,
    numpy_hardswish,
    numpy_identity,
    numpy_leakyrelu,
    numpy_less,
    numpy_less_float,
    numpy_less_or_equal,
    numpy_less_or_equal_float,
    numpy_log,
    numpy_matmul,
    numpy_mul,
    numpy_not,
    numpy_not_float,
    numpy_or,
    numpy_or_float,
    numpy_pad,
    numpy_pow,
    numpy_prelu,
    numpy_reduce_sum,
    numpy_relu,
    numpy_reshape,
    numpy_round,
    numpy_selu,
    numpy_sigmoid,
    numpy_sin,
    numpy_sinh,
    numpy_softplus,
    numpy_sub,
    numpy_tan,
    numpy_tanh,
    numpy_thresholdedrelu,
    numpy_transpose,
    numpy_where,
    torch_avgpool,
    torch_conv,
)

ATTR_TYPES = dict(onnx.AttributeProto.AttributeType.items())
ATTR_GETTERS = {
    ATTR_TYPES["FLOAT"]: lambda attr: attr.f,
    ATTR_TYPES["INT"]: lambda attr: attr.i,
    ATTR_TYPES["STRING"]: lambda attr: attr.s.decode("utf-8"),
    ATTR_TYPES["TENSOR"]: lambda attr: numpy_helper.to_array(attr.t),
    ATTR_TYPES["FLOATS"]: lambda attr: attr.floats,
    ATTR_TYPES["INTS"]: lambda attr: tuple(attr.ints),
    ATTR_TYPES["STRINGS"]: lambda attr: attr.strings,
    ATTR_TYPES["TENSORS"]: lambda attr: tuple(numpy_helper.to_array(val) for val in attr.tensors),
}

# pylint: enable=invalid-name

# We are using OPSET_VERSION_FOR_ONNX_EXPORT for ONNX export, implement the relevant revisions of
# the operators
ONNX_OPS_TO_NUMPY_IMPL: Dict[str, Callable[..., Tuple[numpy.ndarray, ...]]] = {
    "Add": numpy_add,
    "Clip": numpy_clip,
    "Constant": numpy_constant,
    "Cos": numpy_cos,
    "Cosh": numpy_cosh,
    "Acos": numpy_acos,
    "Acosh": numpy_acosh,
    "MatMul": numpy_matmul,
    "Gemm": numpy_gemm,
    "Relu": numpy_relu,
    "Selu": numpy_selu,
    "Elu": numpy_elu,
    "Erf": numpy_erf,
    "ThresholdedRelu": numpy_thresholdedrelu,
    "LeakyRelu": numpy_leakyrelu,
    "Celu": numpy_celu,
    "Sin": numpy_sin,
    "Sinh": numpy_sinh,
    "Asin": numpy_asin,
    "Asinh": numpy_asinh,
    "Sigmoid": numpy_sigmoid,
    "HardSigmoid": numpy_hardsigmoid,
    "Tan": numpy_tan,
    "Tanh": numpy_tanh,
    "Atan": numpy_atan,
    "Atanh": numpy_atanh,
    "Softplus": numpy_softplus,
    "Abs": numpy_abs,
    "Div": numpy_div,
    "Mul": numpy_mul,
    "Sub": numpy_sub,
    "Log": numpy_log,
    "Exp": numpy_exp,
    "Equal": numpy_equal,
    "Identity": numpy_identity,
    "Reshape": numpy_reshape,
    "Transpose": numpy_transpose,
    "Conv": torch_conv,
    "PRelu": numpy_prelu,
    "HardSwish": numpy_hardswish,
    "AveragePool": torch_avgpool,
    "Pad": numpy_pad,
    "Where": numpy_where,
    "Cast": numpy_cast,
    "BatchNormalization": numpy_batchnorm,
    "Flatten": numpy_flatten,
    "Round": numpy_round,
    "Pow": numpy_pow,
    "ReduceSum": numpy_reduce_sum,
    "onnx.brevitas.Quant": numpy_brevitas_quant,
}

# Creating the following dictionaries was introduced following the performance regression issues
# observed in https://github.com/zama-ai/concrete-ml-internal/issues/1357.
# Comparison operators from numpy return boolean values, which is a specific subtype of numpy
# integer types. However, the problem lies in the fact that while this is the expected behavior for
# tree-based models, QuantizedOps only handle float values in order to properly quantize. The
# current solution therefore dissociates the numpy operators used in both cases.
# FIXME: to remove once https://github.com/zama-ai/concrete-ml-internal/issues/1117 is done.

# Comparison operators needed for QuantizedOps as they cast the boolean outputs into floats.
ONNX_COMPARISON_OPS_TO_NUMPY_IMPL_FLOAT: Dict[str, Callable[..., Tuple[numpy.ndarray, ...]]] = {
    "Or": numpy_or_float,
    "Not": numpy_not_float,
    "Greater": numpy_greater_float,
    "GreaterOrEqual": numpy_greater_or_equal_float,
    "Less": numpy_less_float,
    "LessOrEqual": numpy_less_or_equal_float,
}

# Comparison operators used in tree-based models as they keep the outputs' boolean dtype.
ONNX_COMPARISON_OPS_TO_NUMPY_IMPL_BOOL: Dict[str, Callable[..., Tuple[numpy.ndarray, ...]]] = {
    "Or": numpy_or,
    "Not": numpy_not,
    "Greater": numpy_greater,
    "GreaterOrEqual": numpy_greater_or_equal,
    "Less": numpy_less,
    "LessOrEqual": numpy_less_or_equal,
}

# All numpy operators used in QuantizedOps
ONNX_OPS_TO_NUMPY_IMPL.update(ONNX_COMPARISON_OPS_TO_NUMPY_IMPL_FLOAT)

# All numpy operators used for tree-based models
ONNX_OPS_TO_NUMPY_IMPL_BOOL = {**ONNX_OPS_TO_NUMPY_IMPL, **ONNX_COMPARISON_OPS_TO_NUMPY_IMPL_BOOL}


IMPLEMENTED_ONNX_OPS = set(ONNX_OPS_TO_NUMPY_IMPL.keys())


def get_attribute(attribute: onnx.AttributeProto) -> Any:
    """Get the attribute from an ONNX AttributeProto.

    Args:
        attribute (onnx.AttributeProto): The attribute to retrieve the value from.

    Returns:
        Any: The stored attribute value.
    """
    return ATTR_GETTERS[attribute.type](attribute)


def get_op_name(node):
    """Construct the qualified name of the ONNX operator.

    Args:
        node (Any): ONNX graph node

    Returns:
        result (str): qualified name
    """
    return node.domain + ("" if node.domain == "" else ".") + node.op_type


def execute_onnx_with_numpy(
    graph: onnx.GraphProto,
    *inputs: numpy.ndarray,
) -> Tuple[numpy.ndarray, ...]:
    """Execute the provided ONNX graph on the given inputs.

    Args:
        graph (onnx.GraphProto): The ONNX graph to execute.
        *inputs: The inputs of the graph.

    Returns:
        Tuple[numpy.ndarray]: The result of the graph's execution.
    """
    node_results: Dict[str, numpy.ndarray] = dict(
        {graph_input.name: input_value for graph_input, input_value in zip(graph.input, inputs)},
        **{
            initializer.name: numpy_helper.to_array(initializer)
            for initializer in graph.initializer
        },
    )
    for node in graph.node:
        curr_inputs = (node_results[input_name] for input_name in node.input)
        attributes = {attribute.name: get_attribute(attribute) for attribute in node.attribute}
        outputs = ONNX_OPS_TO_NUMPY_IMPL_BOOL[node.op_type](*curr_inputs, **attributes)

        node_results.update(zip(node.output, outputs))
    return tuple(node_results[output.name] for output in graph.output)
