"""Implements the conversion of a tree model to a numpy function."""
import warnings
from typing import Callable, List, Tuple

import numpy
import onnx
from onnx import numpy_helper

from ..common.debugging.custom_assert import assert_true
from ..common.utils import (
    MAX_BITWIDTH_BACKWARD_COMPATIBLE,
    get_onnx_opset_version,
    is_regressor_or_partial_regressor,
)
from ..onnx.convert import OPSET_VERSION_FOR_ONNX_EXPORT, get_equivalent_numpy_forward
from ..onnx.onnx_model_manipulations import clean_graph_at_node_op_type, remove_node_types
from ..quantization import QuantizedArray
from ..quantization.quantizers import UniformQuantizer

# pylint: disable=wrong-import-position,wrong-import-order

# Silence Hummingbird warnings
warnings.filterwarnings("ignore")
from hummingbird.ml import convert as hb_convert  # noqa: E402

# pylint: enable=wrong-import-position,wrong-import-order

# pylint: disable=too-many-branches


def get_onnx_model(model: Callable, x: numpy.ndarray, framework: str) -> onnx.ModelProto:
    """Create ONNX model with Hummingbird convert method.

    Args:
        model (Callable): The tree model to convert.
        x (numpy.ndarray): Dataset used to trace the tree inference and convert the model to ONNX.
        framework (str): The framework from which the ONNX model is generated.
            (options: 'xgboost', 'sklearn')

    Returns:
        onnx.ModelProto: The ONNX model.
    """

    # Silence Hummingbird warnings
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
    return onnx_model


def workaround_squeeze_node_xgboost(onnx_model: onnx.ModelProto):
    """Workaround to fix torch issue that does not export the proper axis in the ONNX squeeze node.

    FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2778
    The squeeze ops does not have the proper dimensions.
    remove the following workaround when the issue is fixed
    Add the axis attribute to the Squeeze node

    Args:
        onnx_model (onnx.ModelProto): The ONNX model.
    """
    target_node_id_list = [
        i for i, node in enumerate(onnx_model.graph.node) if node.op_type == "Squeeze"
    ]
    assert_true(
        len(target_node_id_list) == 1,
        "Multiple Squeeze node found which is unexpected in tree-based models",
    )
    axes_input_name = "axes_squeeze"
    axes_input = onnx.helper.make_tensor(axes_input_name, onnx.TensorProto.INT64, [1], (1,))

    onnx_model.graph.initializer.append(axes_input)
    onnx_model.graph.node[target_node_id_list[0]].input.insert(1, axes_input_name)


def add_transpose_after_last_node(onnx_model: onnx.ModelProto):
    """Add transpose after last node.

    Args:
        onnx_model (onnx.ModelProto): The ONNX model.
    """
    # Get the output node
    output_node = onnx_model.graph.output[0]

    # Create the node with perm attribute equal to (2, 1, 0)
    transpose_node = onnx.helper.make_node(
        "Transpose",
        inputs=[output_node.name],
        outputs=["transposed_output"],
        perm=[2, 1, 0],
    )

    onnx_model.graph.node.append(transpose_node)
    onnx_model.graph.output[0].name = "transposed_output"


def preprocess_tree_predictions(
    init_tensor: numpy.ndarray,
    output_n_bits: int,
) -> QuantizedArray:
    """Apply post-processing from the graph.

    Args:
        init_tensor (numpy.ndarray): Model parameters to be pre-processed.
        output_n_bits (int): The number of bits of the output.

    Returns:
        QuantizedArray: Quantizer for the tree predictions.
    """

    # Quantize probabilities and store QuantizedArray
    # IMPORTANT: we must use symmetric signed quantization such that
    # 0 in clear == 0 in quantized.

    quant_args = {}

    # If we have negative values, use a symmetric quantization
    # in order to have a zero zero-point
    if numpy.min(init_tensor) < 0:
        is_signed = is_symmetric = True

    # To ensure the zero-point is 0 we force the
    # range of the quantizer to [0..max(init_tensor)]
    else:
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
    # Make sure the zero_point is 0 to prevent errors in Hummingbird's GEMM approach.
    # Asymmetric quantization may cause the zero_point to be non-zero
    # which leads to incorrect results.
    assert_true(
        q_y.quantizer.zero_point == 0,
        "Zero point is not 0. Symmetric signed quantization must work.",
    )
    return q_y


def tree_onnx_graph_preprocessing(
    onnx_model: onnx.ModelProto, framework: str, expected_number_of_outputs: int
):
    """Apply pre-processing onto the ONNX graph.

    Args:
        onnx_model (onnx.ModelProto): The ONNX model.
        framework (str): The framework from which the ONNX model is generated.
            (options: 'xgboost', 'sklearn')
        expected_number_of_outputs (int): The expected number of outputs in the ONNX model.
    """
    # Make sure the ONNX version returned by Hummingbird is OPSET_VERSION_FOR_ONNX_EXPORT
    onnx_version = get_onnx_opset_version(onnx_model)
    assert_true(
        onnx_version == OPSET_VERSION_FOR_ONNX_EXPORT,
        f"The ONNX version returned by Hummingbird is {onnx_version} "
        f"instead of {OPSET_VERSION_FOR_ONNX_EXPORT}",
    )

    # Check we do have the correct number of ONNX output.
    # Hummingbird returns two outputs for classification (predict and predict_proba)
    # while a single output for regression (predict)
    assert_true(
        len(onnx_model.graph.output) == expected_number_of_outputs,
        on_error_msg=f"{len(onnx_model.graph.output)} != 2",
    )

    # Cut the graph at the ReduceSum node as large sum are not yet supported.
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/451
    clean_graph_at_node_op_type(onnx_model, "ReduceSum")

    if framework == "xgboost":
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2778
        # The squeeze ops does not have the proper dimensions.
        # remove the following workaround when the issue is fixed
        # Add the axis attribute to the Squeeze node
        workaround_squeeze_node_xgboost(onnx_model)
    else:
        # Add a transpose node after the last node.
        # sklearn models apply the reduce sum before the transpose.
        # To have equivalent output between xgboost in sklearn,
        # apply the transpose before returning the output.
        add_transpose_after_last_node(onnx_model)

    # Cast nodes are not necessary so remove them.
    op_type_to_remove = ["Cast"]
    remove_node_types(onnx_model, op_type_to_remove)


def tree_values_preprocessing(
    onnx_model: onnx.ModelProto,
    framework: str,
    output_n_bits: int,
) -> QuantizedArray:
    """Pre-process tree values.

    Args:
        onnx_model (onnx.ModelProto): The ONNX model.
        framework (str): The framework from which the ONNX model is generated.
            (options: 'xgboost', 'sklearn')
        output_n_bits (int): The number of bits of the output.

    Returns:
        QuantizedArray: Quantizer for the tree predictions.
    """

    # Modify ONNX graph to fit in FHE
    for i, initializer in enumerate(onnx_model.graph.initializer):
        # All constants in our tree should be integers.
        # Tree thresholds can be rounded up or down (depending on the tree implementation)
        # while the final probabilities/regression values must be quantized.
        # We extract the value stored in each initializer node into the init_tensor.
        init_tensor = numpy_helper.to_array(initializer)
        if "weight_3" in initializer.name:
            # weight_3 is the prediction tensor, apply the required pre-processing
            q_y = preprocess_tree_predictions(init_tensor, output_n_bits)

            # Get the preprocessed tree predictions to replace the current (non-quantized)
            # values in the onnx_model.
            init_tensor = q_y.qvalues
        else:
            if framework == "xgboost":
                # xgboost uses "<" operator thus we must round up.
                init_tensor = numpy.ceil(init_tensor)
            elif framework == "sklearn":
                # sklearn trees use ">" operator thus we must round down.
                init_tensor = numpy.floor(init_tensor)
        new_initializer = numpy_helper.from_array(init_tensor.astype(numpy.int64), initializer.name)
        onnx_model.graph.initializer[i].CopyFrom(new_initializer)
    return q_y


# pylint: disable=too-many-locals
def tree_to_numpy(
    model: Callable,
    x: numpy.ndarray,
    framework: str,
    output_n_bits: int = MAX_BITWIDTH_BACKWARD_COMPATIBLE,
) -> Tuple[Callable, List[UniformQuantizer], onnx.ModelProto]:
    """Convert the tree inference to a numpy functions using Hummingbird.

    Args:
        model (Callable): The tree model to convert.
        x (numpy.ndarray): The input data.
        framework (str): The framework from which the ONNX model is generated.
            (options: 'xgboost', 'sklearn')
        output_n_bits (int): The number of bits of the output. Default to 8.

    Returns:
        Tuple[Callable, List[QuantizedArray], onnx.ModelProto]: A tuple with a function that takes a
            numpy array and returns a numpy array, QuantizedArray object to quantize and de-quantize
            the output of the tree, and the ONNX model.
    """
    # mypy
    assert output_n_bits is not None

    assert_true(
        framework in ["xgboost", "sklearn"],
        f"framework={framework} is not supported. It must be either 'xgboost' or 'sklearn'",
    )

    onnx_model = get_onnx_model(model, x, framework)

    # Get the expected number of ONNX outputs in the sklearn model.
    expected_number_of_outputs = 1 if is_regressor_or_partial_regressor(model) else 2

    # ONNX graph pre-processing to make the model FHE friendly
    # i.e., delete irrelevant nodes and cut the graph before the final ensemble sum)
    tree_onnx_graph_preprocessing(onnx_model, framework, expected_number_of_outputs)

    # Tree values pre-processing
    # i.e., mainly predictions quantization
    # but also rounding the threshold such that they are now integers
    q_y = tree_values_preprocessing(onnx_model, framework, output_n_bits)

    _tree_inference = get_equivalent_numpy_forward(onnx_model)

    return (_tree_inference, [q_y.quantizer], onnx_model)
