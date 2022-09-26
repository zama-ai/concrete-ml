"""Implements the conversion of a tree model to a numpy function."""
import warnings
from enum import Enum
from typing import Callable, List, Optional, Tuple

import numpy
import onnx
from concrete.numpy import MAXIMUM_TLU_BIT_WIDTH
from onnx import numpy_helper

from ..common.debugging.custom_assert import assert_true
from ..common.utils import get_onnx_opset_version
from ..onnx.convert import OPSET_VERSION_FOR_ONNX_EXPORT, get_equivalent_numpy_forward
from ..onnx.onnx_model_manipulations import (
    clean_graph_after_node,
    keep_following_outputs_discard_others,
    remove_node_types,
)
from ..quantization import QuantizedArray
from ..quantization.quantizers import UniformQuantizer

# pylint: disable=wrong-import-position,wrong-import-order

# Silence hummingbird warnings
warnings.filterwarnings("ignore")
from hummingbird.ml import convert as hb_convert  # noqa: E402

# pylint: enable=wrong-import-position,wrong-import-order

# pylint: disable=too-many-branches


class Task(Enum):
    """Task enumerate."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


EXPECTED_NUMBER_OF_OUTPUTS_PER_TASK = {Task.CLASSIFICATION: 2, Task.REGRESSION: 1}


def tree_to_numpy(
    model: onnx.ModelProto,
    x: numpy.ndarray,
    framework: str,
    task: Task,
    output_n_bits: Optional[int] = MAXIMUM_TLU_BIT_WIDTH,
) -> Tuple[Callable, List[UniformQuantizer], onnx.ModelProto]:
    """Convert the tree inference to a numpy functions using Hummingbird.

    Args:
        model (onnx.ModelProto): The model to convert.
        x (numpy.ndarray): The input data.
        framework (str): The framework from which the onnx_model is generated.
            (options: 'xgboost', 'sklearn')
        task (Task): The task the model is solving
        output_n_bits (int): The number of bits of the output.

    Returns:
        Tuple[Callable, List[QuantizedArray], onnx.ModelProto]: A tuple with a function that takes a
            numpy array and returns a numpy array, QuantizedArray object to quantize and dequantize
            the output of the tree, and the ONNX model.
    """
    # mypy
    assert output_n_bits is not None

    assert_true(
        framework in ["xgboost", "sklearn"],
        f"framework={framework} is not supported. It must be either 'xgboost' or 'sklearn'",
    )

    # Silence hummingbird warnings
    warnings.filterwarnings("ignore")

    extra_config = {
        "tree_implementation": "gemm",
        "onnx_target_opset": OPSET_VERSION_FOR_ONNX_EXPORT,
    }
    if framework != "sklearn":
        extra_config["n_features"] = x.shape[1]

    onnx_model = hb_convert(
        model,
        backend="onnx",
        test_input=x,
        extra_config=extra_config,
    ).model

    # Make sure the onnx version returned by hummingbird is OPSET_VERSION_FOR_ONNX_EXPORT
    onnx_version = get_onnx_opset_version(onnx_model)
    assert_true(
        onnx_version == OPSET_VERSION_FOR_ONNX_EXPORT,
        f"The onnx version returned by Hummingbird is {onnx_version} "
        f"instead of {OPSET_VERSION_FOR_ONNX_EXPORT}",
    )

    # The tree returned by hummingbird has two outputs which is not supported currently by the
    # compiler (as it only returns one output). Here we explicitly only keep the output named
    # "variable", which after inspecting the hummingbird code is considered the canonical
    # output. This was causing issues as the virtual lib (correctly) returned both outputs which
    # the predict method did not expect as it was only getting one output from the compiler
    # engine.
    # The output we keep is the output giving the actual classes out, the other one is the
    # one-hot encoded vectors to indicate which class is predicted.
    # This is fine for now as we remove the argmax operator.

    # Check we do have the correct number of output for the given task
    expected_number_of_outputs = EXPECTED_NUMBER_OF_OUTPUTS_PER_TASK[task]
    assert_true(
        len(onnx_model.graph.output) == expected_number_of_outputs,
        on_error_msg=f"{len(onnx_model.graph.output)} != 2",
    )

    # If the framework used is XGBoost, remove all the ONNX graph's nodes that follow the last
    # MatMul operator
    if framework == "xgboost":
        # Find Reshape node after last MatMul node
        # (last MatMul node takes an input with "weight_3" in the name)
        node_cut_id = ""
        for node in onnx_model.graph.node:
            if node_cut_id != "" and node.op_type == "Reshape" and node_cut_id in node.input:
                node_name_to_cut = node.name
                break

            if len(node.input) > 0 and "weight_3" in node.input[0] and node.op_type == "MatMul":
                node_cut_id = node.output[0]

        clean_graph_after_node(onnx_model=onnx_model, node_name=node_name_to_cut)

    # Else, keep track of the last output
    else:
        keep_following_outputs_discard_others(onnx_model, ("variable",))

    # TODO remove Transpose from the list when #931 is done
    # TODO remove Gather from the list when #345 is done
    op_type_to_remove = ["Transpose", "ArgMax", "ReduceSum", "Cast", "Gather"]
    remove_node_types(onnx_model, op_type_to_remove)

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

            quant_args = {}
            if numpy.min(init_tensor) < 0:
                # If we have negative values, use a symmetric quantization
                # in order to have a zero zero-point

                is_signed = is_symmetric = True
            else:
                # To ensure the zero-point is 0 we force the
                # range of the quantizer to [0..max(init_tensor)]

                is_signed = is_symmetric = False
                quant_args["rmax"] = numpy.max(init_tensor)
                quant_args["rmin"] = 0
                quant_args["uvalues"] = []

            q_y = QuantizedArray(
                n_bits=output_n_bits,
                values=init_tensor,
                is_signed=is_signed,
                is_symmetric=is_symmetric,
                **quant_args,
            )
            # Make sure the zero_point is 0.
            # This is important for the final summation to work.
            # The problem comes from the GEMM approach in Hummingbird where the default value of
            # rows in the matrix that selects the correct output in each tree is set to True when
            # the node does not exist in the tree. So the same value as a selected node.
            # For Hummingbird, this is not a problem as the values for nodes that do not exist
            # are set to 0.
            # Here, if we use asymmetric quantization, there is a risk that the zero_point is
            # not 0. The matmul will then select values != 0 and sum them.
            # The resulting value will be different from the expected terminal node value.
            assert_true(
                q_y.quantizer.zero_point == 0,
                "Zero point is not 0. Symmetric signed quantization must work.",
            )
            init_tensor = q_y.qvalues
        else:
            if framework == "xgboost":
                # xgboost uses "<" operator thus we must round up.
                init_tensor = numpy.ceil(init_tensor)
            elif framework == "sklearn":
                # sklearn trees use ">" operator thus we must round down.
                init_tensor = numpy.floor(init_tensor)
        new_initializer = numpy_helper.from_array(init_tensor.astype(int), initializer.name)
        onnx_model.graph.initializer[i].CopyFrom(new_initializer)

    _tensor_tree_predict = get_equivalent_numpy_forward(onnx_model)

    return (_tensor_tree_predict, [q_y.quantizer], onnx_model)
