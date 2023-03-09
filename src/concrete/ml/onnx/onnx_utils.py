"""Utils to interpret an ONNX model with numpy."""
# Utils to interpret an ONNX model with numpy.


#
#                                  Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/
#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
#
#    1. Definitions.
#
#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.
#
#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.
#
#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.

#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.

#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.

#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.

#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).

#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.

#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."

#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.

#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.

#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.

#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:

#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and

#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and

#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and

#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.

#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.

#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.

#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.

#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.

#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.

#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.

#    END OF TERMS AND CONDITIONS

#    APPENDIX: How to apply the Apache License to your work.

#       To apply the Apache License to your work, attach the following
#       boilerplate notice, with the fields enclosed by brackets "[]"
#       replaced with your own identifying information. (Don't include
#       the brackets!)  The text should be enclosed in the appropriate
#       comment syntax for the file format. We also recommend that a
#       file or class name and description of purpose be included on the
#       same "printed page" as the copyright notice for easier
#       identification within third-party archives.

#    Copyright 2018 Google LLC

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

#
# Modifications copyright (C) 2022-2023 Zama:
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
    numpy_avgpool,
    numpy_batchnorm,
    numpy_brevitas_quant,
    numpy_cast,
    numpy_celu,
    numpy_clip,
    numpy_concatenate,
    numpy_constant,
    numpy_constant_of_shape,
    numpy_conv,
    numpy_cos,
    numpy_cosh,
    numpy_div,
    numpy_elu,
    numpy_equal,
    numpy_erf,
    numpy_exp,
    numpy_flatten,
    numpy_floor,
    numpy_gather,
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
    numpy_max,
    numpy_maxpool,
    numpy_min,
    numpy_mul,
    numpy_neg,
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
    numpy_shape,
    numpy_sigmoid,
    numpy_sign,
    numpy_sin,
    numpy_sinh,
    numpy_slice,
    numpy_softplus,
    numpy_squeeze,
    numpy_sub,
    numpy_tan,
    numpy_tanh,
    numpy_thresholdedrelu,
    numpy_transpose,
    numpy_unsqueeze,
    numpy_where,
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
    "Conv": numpy_conv,
    "PRelu": numpy_prelu,
    "HardSwish": numpy_hardswish,
    "AveragePool": numpy_avgpool,
    "MaxPool": numpy_maxpool,
    "Pad": numpy_pad,
    "Where": numpy_where,
    "Cast": numpy_cast,
    "BatchNormalization": numpy_batchnorm,
    "Flatten": numpy_flatten,
    "Round": numpy_round,
    "Pow": numpy_pow,
    "ReduceSum": numpy_reduce_sum,
    "onnx.brevitas.Quant": numpy_brevitas_quant,
    "Floor": numpy_floor,
    "Max": numpy_max,
    "Min": numpy_min,
    "Neg": numpy_neg,
    "Sign": numpy_sign,
    "Concat": numpy_concatenate,
    "Unsqueeze": numpy_unsqueeze,
    "Squeeze": numpy_squeeze,
    "Slice": numpy_slice,
    "Gather": numpy_gather,
    "Shape": numpy_shape,
    "ConstantOfShape": numpy_constant_of_shape,
}


# Comparison operators from numpy return boolean values, which is a specific subtype of numpy
# integer types. However, while this is the expected behavior for tree-based models, QuantizedOps
# can only handle float values in order to properly quantize. The current solution therefore
# dissociates the numpy operators used in both cases.

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


def get_op_type(node):
    """Construct the qualified type name of the ONNX operator.

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


# https://github.com/microsoft/onnxruntime/blob/fdce4fa6af437b0b822958ab47b3b8f77f9e14ae/tools/python/remove_initializer_from_input.py
# https://github.com/microsoft/onnxruntime/issues/4033
def remove_initializer_from_input(model: onnx.ModelProto):  # pragma: no cover
    """Remove initializers from model inputs.

    In some cases, ONNX initializers may appear, erroneously, as graph inputs.
    This function searches all model inputs and removes those that are initializers.

    Args:
        model (onnx.ModelProto): the model to clean

    Returns:
        onnx.ModelProto: the cleaned model
    """

    inputs = model.graph.input
    name_to_input: Dict[str, Any] = {}
    for model_input in inputs:
        name_to_input[model_input.name] = model_input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    return model
