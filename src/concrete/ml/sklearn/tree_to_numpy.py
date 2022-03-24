"""Implements the conversion of a tree model to a numpy function."""
import warnings
from typing import Callable, Optional, Tuple

import numpy
import onnx
from onnx import numpy_helper

from ..common.debugging.custom_assert import assert_true
from ..onnx.convert import get_equivalent_numpy_forward
from ..onnx.onnx_model_manipulations import (
    cut_onnx_graph_after_node_name,
    keep_following_outputs_discard_others,
    remove_transpose_in_first_gemm_node,
    replace_uncessary_nodes_by_identity,
    simplify_onnx_model,
)
from ..quantization import QuantizedArray

# pylint: disable=wrong-import-position,wrong-import-order

# Silence hummingbird warnings
warnings.filterwarnings("ignore")
from hummingbird.ml import convert as hb_convert  # noqa: E402

# pylint: enable=wrong-import-position,wrong-import-order

# pylint: disable=too-many-branches


def tree_to_numpy(
    model: onnx.ModelProto,
    x: numpy.ndarray,
    framework: str,
    output_n_bits: Optional[int] = 7,
    use_workaround_for_transpose: bool = False,
) -> Tuple[Callable, QuantizedArray]:
    """Convert the tree inference to a numpy functions using Hummingbird.

    Args:
        model (onnx.ModelProto): The model to convert.
        x (numpy.ndarray): The input data.
        framework (str): The framework from which the onnx_model is generated.
            (options: 'xgboost', 'sklearn')
        output_n_bits (int): The number of bits of the output.
        use_workaround_for_transpose (bool): Whether to use the workaround for transpose.

    Returns:
        Union[Callable, QuantizedArray]: A tuple with a function that takes a numpy array and
            returns a numpy array, QuantizedArray object to quantize and dequantize
            the output of the tree.
    """
    # mypy
    assert output_n_bits is not None

    assert_true(
        framework in ["xgboost", "sklearn"],
        f"framework={framework} is not supported. It must be either 'xgboost' or 'sklearn'",
    )

    # Silence hummingbird warnings
    warnings.filterwarnings("ignore")

    # Convert model to onnx using hummingbird
    if framework == "sklearn":
        onnx_model = hb_convert(
            model, backend="onnx", test_input=x, extra_config={"tree_implementation": "gemm"}
        ).model
    else:
        onnx_model = hb_convert(
            model,
            backend="onnx",
            test_input=x,
            extra_config={"tree_implementation": "gemm", "n_features": x.shape[1]},
        ).model

    # The tree returned by hummingbird has two outputs which is not supported currently by the
    # compiler (as it only returns one output). Here we explicitely only keep the output named
    # "variable", which after inspecting the hummingbird code is considered the canonical
    # output. This was causing issues as the virtual lib (correctly) returned both outputs which
    # the predict method did not expect as it was only getting one output from the compiler
    # engine.
    # The output we keep is the output giving the actual classes out, the other one is the
    # one-hot encoded vectors to indicate which class is predicted.
    # This is fine for now as we remove the argmax operator.

    # Check we do have two outputs first
    assert_true(len(onnx_model.graph.output) == 2)

    output_to_follow = "variable"
    if framework == "xgboost":
        # Find Reshape node after last MatMul node
        # (last MatMul node takes an input with "weight_3" in the name)
        node_cut_id = ""
        for node in onnx_model.graph.node:
            if node_cut_id != "" and node.op_type == "Reshape" and node_cut_id in node.input:
                cut_node_name = node.name
                break

            if len(node.input) > 0 and "weight_3" in node.input[0] and node.op_type == "MatMul":
                node_cut_id = node.output[0]

        output_to_follow = cut_onnx_graph_after_node_name(onnx_model, cut_node_name)
    keep_following_outputs_discard_others(onnx_model, (output_to_follow,))

    # TODO remove Transpose from the list when #292 is done
    op_type_to_remove = ["Transpose", "ArgMax", "ReduceSum", "Cast"]
    replace_uncessary_nodes_by_identity(onnx_model, op_type_to_remove)

    # Modify onnx graph to fit in FHE
    for i, initializer in enumerate(onnx_model.graph.initializer):
        # All constants in our tree should be integers.
        # Tree thresholds can be rounded up or down (depending on the tree implementation)
        # while the final probabilities/regression values must be quantized.
        init_tensor = numpy_helper.to_array(initializer)
        if "weight_3" in initializer.name:
            # This is the prediction tensor.
            # Quantize probabilities and store QuantizedArray
            # IMPORTANT: we must use symmetric signed quantization such that
            # 0 in clear == 0 in quantized.
            output_is_signed = False
            if init_tensor.min() < 0:
                output_is_signed = True
            q_y = QuantizedArray(
                n_bits=output_n_bits,
                values=init_tensor,
                is_signed=output_is_signed,
                is_symmetric=True,
            )
            # Make sure the zero_point is 0
            assert_true(
                q_y.zero_point == 0,
                "Zero point is not 0. Symmetric signed quantization must work.",
            )
            init_tensor = q_y.quant()
        else:
            if framework == "xgboost":
                # xgboost uses "<" operator thus we must round up.
                init_tensor = numpy.ceil(init_tensor)
            elif framework == "sklearn":
                # sklearn trees use ">" operator thus we must round down.
                init_tensor = numpy.floor(init_tensor)
        new_initializer = numpy_helper.from_array(init_tensor.astype(int), initializer.name)
        onnx_model.graph.initializer[i].CopyFrom(new_initializer)

    if use_workaround_for_transpose:
        # Since the transpose is currently not implemented in concrete numpy
        # the input is transposed in clear. We need to update the Gemm
        # where the input is transposed.

        # Gemm has transA and transB parameter. B is the input.
        # If we transpose the input before, we don't have to do it afterward.
        # In FHE we currently only send 1 example so the input has shape (1, n_features)
        # We simply need to transpose it to (n_features, 1)

        # FIXME remove this workaround once #292 is fixed
        remove_transpose_in_first_gemm_node(onnx_model)

    simplify_onnx_model(onnx_model)
    _tensor_tree_predict = get_equivalent_numpy_forward(onnx_model)

    return (_tensor_tree_predict, q_y)
