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

"""Utils to interpret an ONNX model with numpy."""
from typing import Any, Callable, Dict, Tuple

import numpy
import onnx
from onnx import numpy_helper

from .ops_impl import (
    numpy_add,
    numpy_clip,
    numpy_constant,
    numpy_gemm,
    numpy_matmul,
    numpy_relu,
    numpy_sigmoid,
)

ATTR_TYPES = dict(onnx.AttributeProto.AttributeType.items())
ATTR_GETTERS = {
    ATTR_TYPES["FLOAT"]: lambda attr: attr.f,
    ATTR_TYPES["INT"]: lambda attr: attr.i,
    ATTR_TYPES["STRING"]: lambda attr: attr.s,
    ATTR_TYPES["TENSOR"]: lambda attr: numpy_helper.to_array(attr.t),
    ATTR_TYPES["FLOATS"]: lambda attr: attr.floats,
    ATTR_TYPES["INTS"]: lambda attr: attr.ints,
    ATTR_TYPES["STRINGS"]: lambda attr: attr.strings,
    ATTR_TYPES["TENSORS"]: lambda attr: tuple(numpy_helper.to_array(val) for val in attr.tensors),
}

# pylint: enable=invalid-name

# We are using opset 14 for ONNX export, implement the relevant revisions of the operators
ONNX_OPS_TO_NUMPY_IMPL: Dict[str, Callable[..., Tuple[numpy.ndarray, ...]]] = {
    "Add": numpy_add,
    "Clip": numpy_clip,
    "Constant": numpy_constant,
    "MatMul": numpy_matmul,
    "Gemm": numpy_gemm,
    "Relu": numpy_relu,
    "Sigmoid": numpy_sigmoid,
}

IMPLEMENTED_ONNX_OPS = set(ONNX_OPS_TO_NUMPY_IMPL.keys())


def get_attribute(attribute: onnx.AttributeProto) -> Any:
    """Get the attribute from an ONNX AttributeProto.

    Args:
        attribute (onnx.AttributeProto): The attribute to retrieve the value from.

    Returns:
        Any: The stored attribute value.
    """
    return ATTR_GETTERS[attribute.type](attribute)


def execute_onnx_with_numpy(
    graph: onnx.GraphProto, *inputs: numpy.ndarray
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
        outputs = ONNX_OPS_TO_NUMPY_IMPL[node.op_type](*curr_inputs, **attributes)
        node_results.update(zip(node.output, outputs))
    return tuple(node_results[output.name] for output in graph.output)
