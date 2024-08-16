"""Implements the conversion of a tree model to a numpy function."""

import math
import os
import warnings
from typing import Callable, List, Optional, Tuple

import numpy
import onnx
import sklearn
from onnx import numpy_helper

from ..common.debugging.custom_assert import assert_true
from ..common.utils import (
    MAX_BITWIDTH_BACKWARD_COMPATIBLE,
    get_onnx_opset_version,
    is_regressor_or_partial_regressor,
)
from ..onnx.convert import (
    OPSET_VERSION_FOR_ONNX_EXPORT,
    get_equivalent_numpy_forward_from_onnx_tree,
)
from ..onnx.onnx_model_manipulations import (
    clean_graph_after_node_op_type,
    clean_graph_at_node_op_type,
    remove_node_types,
)
from ..onnx.onnx_utils import get_op_type
from ..quantization import QuantizedArray
from ..quantization.quantizers import UniformQuantizer

# pylint: disable=wrong-import-position,wrong-import-order

# Silence Hummingbird warnings
warnings.filterwarnings("ignore")

from hummingbird.ml import convert as hb_convert  # noqa: E402

# pylint: disable=too-many-branches
# pylint: enable=wrong-import-position,wrong-import-order

# Most significant bits to retain when applying rounding to the tree
MSB_TO_KEEP_FOR_TREES = 1

# Minimum bitwidth to apply rounding
MIN_CIRCUIT_THRESHOLD_FOR_TREES = 4


def get_onnx_model(model, x: numpy.ndarray, framework: str) -> onnx.ModelProto:
    """Create ONNX model with Hummingbird convert method.

    Args:
        model: The tree model to convert.
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


def assert_add_node_and_constant_in_xgboost_regressor_graph(onnx_model: onnx.ModelProto):
    """Assert if an Add node with a specific constant exists in the ONNX graph.

    Args:
        onnx_model (onnx.ModelProto): The ONNX model.
    """

    constant_add_name = "_operators.0.base_prediction"
    is_expected_add_node_present = False
    initializer_value_correct = False

    # Find the initializer with the specified name
    initializer = next(
        (init for init in onnx_model.graph.initializer if init.name == constant_add_name), None
    )

    # Check if the initializer exists and its value is 0.5
    if initializer:
        values = onnx.numpy_helper.to_array(initializer)
        if values.size == 1 and values[0] == 0.5:
            initializer_value_correct = True

    # Iterate over all nodes in the model's graph
    for node in onnx_model.graph.node:
        # Check if the node is an "Add" node and has the
        # specified initializer as one of its inputs
        if node.op_type == "Add" and constant_add_name in node.input:
            is_expected_add_node_present = True
            break

    assert_true(
        is_expected_add_node_present and initializer_value_correct,
        "XGBoostRegressor is not supported.",
    )


def add_transpose_after_last_node(onnx_model: onnx.ModelProto, fhe_ensembling: bool = False):
    """Add transpose after last node.

    Args:
        onnx_model (onnx.ModelProto): The ONNX model.
        fhe_ensembling (bool): Determines whether the sum of the trees' outputs is computed in FHE.
            Default to False.
    """
    # Get the output node
    output_node = onnx_model.graph.output[0]

    # The state of the 'fhe_ensembling' variable affects the structure of the model's ONNX graph.
    # When the option is enabled, the graph is cut after the ReduceSum node.
    # When it is disabled, the graph is cut at the ReduceSum node, which alters the output shape.
    # Therefore, it is necessary to adjust this shape with the correct permutation.

    # When using FHE sum for tree ensembles, create the node with perm attribute equal to (1, 0)
    # Otherwise, create the node with perm attribute equal to (2, 1, 0)
    perm = [1, 0] if fhe_ensembling else [2, 1, 0]

    transpose_node = onnx.helper.make_node(
        "Transpose",
        inputs=[output_node.name],
        outputs=["transposed_output"],
        perm=perm,
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
    onnx_model: onnx.ModelProto,
    framework: str,
    expected_number_of_outputs: int,
    fhe_ensembling: bool = False,
):
    """Apply pre-processing onto the ONNX graph.

    Args:
        onnx_model (onnx.ModelProto): The ONNX model.
        framework (str): The framework from which the ONNX model is generated.
            (options: 'xgboost', 'sklearn')
        expected_number_of_outputs (int): The expected number of outputs in the ONNX model.
        fhe_ensembling (bool): Determines whether the sum of the trees' outputs is computed in FHE.
            Default to False.
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

    # Check that a XGBoostRegressor onnx graph has the + 0.5 add node.
    if framework == "xgboost":
        # Make sure it is a regression model
        # (by checking it has a single output, as mentioned above)
        if len(onnx_model.graph.output) == 1:
            assert_add_node_and_constant_in_xgboost_regressor_graph(onnx_model)

    # Cut the graph after the ReduceSum node to remove
    # argmax, sigmoid, softmax from the graph.
    if fhe_ensembling:
        clean_graph_after_node_op_type(onnx_model, "ReduceSum")
    else:
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
        add_transpose_after_last_node(onnx_model, fhe_ensembling)

    # Cast nodes are not necessary so remove them.
    remove_node_types(onnx_model, op_types_to_remove=["Cast"])


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
    q_y = QuantizedArray(
        n_bits=1, values=numpy.zeros(shape=(2,), dtype=numpy.float64), value_is_float=True
    )

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

        elif "bias_1" in initializer.name:
            if framework == "xgboost":
                # xgboost uses "<" (Less) operator thus we must round up.
                init_tensor = numpy.ceil(init_tensor)
            elif framework == "sklearn":
                # sklearn trees use "<=" (LessOrEqual) operator thus we must round down.
                init_tensor = numpy.floor(init_tensor)
        new_initializer = numpy_helper.from_array(init_tensor.astype(numpy.int64), initializer.name)
        onnx_model.graph.initializer[i].CopyFrom(new_initializer)

    return q_y


# pylint: disable=too-many-locals
def tree_to_numpy(
    model: sklearn.base.BaseEstimator,
    x: numpy.ndarray,
    framework: str,
    use_rounding: bool = True,
    fhe_ensembling: bool = False,
    output_n_bits: int = MAX_BITWIDTH_BACKWARD_COMPATIBLE,
) -> Tuple[Callable, List[UniformQuantizer], onnx.ModelProto]:
    """Convert the tree inference to a numpy functions using Hummingbird.

    Args:
        model (Callable): The tree model to convert.
        x (numpy.ndarray): The input data.
        use_rounding (bool): Determines whether the rounding feature is enabled or disabled.
            Default to True.
        fhe_ensembling (bool): Determines whether the sum of the trees' outputs is computed in FHE.
            Default to False.
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

    lsbs_to_remove_for_trees: Optional[Tuple[int, int]] = None

    assert_true(
        framework in ["xgboost", "sklearn"],
        f"framework={framework} is not supported. It must be either 'xgboost' or 'sklearn'",
    )

    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4545
    # Execute with 2 example for efficiency in large data scenarios to prevent slowdown
    # but also to work around the HB export issue
    onnx_model = get_onnx_model(model, x[:2] if x.shape[0] > 1 else x, framework)

    # Compute for tree-based models the LSB to remove in stage 1 and stage 2
    if use_rounding:
        # First LSB refers to Less or LessOrEqual comparisons
        # Second LSB refers to Equal comparison
        lsbs_to_remove_for_trees = _compute_lsb_to_remove_for_trees(onnx_model, x)

        # mypy
        assert len(lsbs_to_remove_for_trees) == 2

    # Get the expected number of ONNX outputs in the sklearn model.
    expected_number_of_outputs = 1 if is_regressor_or_partial_regressor(model) else 2

    # ONNX graph pre-processing to make the model FHE friendly
    # i.e., delete irrelevant nodes and cut the graph before the final ensemble sum)
    tree_onnx_graph_preprocessing(onnx_model, framework, expected_number_of_outputs, fhe_ensembling)

    # Tree values pre-processing
    # i.e., mainly predictions quantization
    # but also rounding the threshold such that they are now integers
    q_y = tree_values_preprocessing(onnx_model, framework, output_n_bits)

    _tree_inference, onnx_model = get_equivalent_numpy_forward_from_onnx_tree(
        onnx_model, lsbs_to_remove_for_trees=lsbs_to_remove_for_trees
    )

    return (_tree_inference, [q_y.quantizer], onnx_model)


# Remove this function once the truncate feature is released
# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4143
def _compute_lsb_to_remove_for_trees(
    onnx_model: onnx.ModelProto, q_x: numpy.ndarray
) -> Tuple[int, int]:
    """Compute the LSB to remove for the comparison operators in the trees.

    Referring to this paper: https://arxiv.org/pdf/2010.04804.pdf, there are
    2 levels of comparison for trees, one at the level of X.A < B and a second at
    the level of I.C == D.

    Args:
        onnx_model (onnx.ModelProto): The model to clean
        q_x (numpy.ndarray): The quantized inputs

    Returns:
        Tuple[int, int]: the number of LSB to remove for level 1 and level 2

    Raises:
        ValueError: if comparison function ('Less' or 'LessOrEqual') cannot be determined.
    """

    def get_bitwidth(array: numpy.ndarray) -> int:
        """Compute the bitwidth required to represent the largest value in `array`.

        Args:
            array (umpy.ndarray): The array for which the bitwidth needs to be checked.

        Returns:
            int: The required bits to represent the array.
        """

        max_val = numpy.max(numpy.abs(array))

        # + 1 is added to include the sign bit
        bitwidth = math.ceil(math.log2(max_val + 1)) + 1
        return bitwidth

    def get_lsbs_to_remove_for_trees(array: numpy.ndarray) -> int:
        """Update the number of LSBs to remove based on overflow detection.

        this function works only for MSB = 1

        Args:
            array (numpy.ndarray): The array for which the bitwidth needs to be checked.

        Returns:
            int: The updated LSB to remove.
        """

        lsbs_to_remove_for_trees: int = 0

        prev_bitwidth = get_bitwidth(array)

        if prev_bitwidth > MIN_CIRCUIT_THRESHOLD_FOR_TREES:

            if prev_bitwidth - MSB_TO_KEEP_FOR_TREES > 0:

                msb = MSB_TO_KEEP_FOR_TREES if MSB_TO_KEEP_FOR_TREES > 1 else 0
                lsbs_to_remove_for_trees = prev_bitwidth - msb

        return lsbs_to_remove_for_trees

    quant_params = {
        onnx_init.name: numpy_helper.to_array(onnx_init)
        for onnx_init in onnx_model.graph.initializer
        if "weight" in onnx_init.name or "bias" in onnx_init.name
    }

    key_mat_1 = [key for key in quant_params.keys() if "_1" in key and "weight" in key][0]
    key_bias_1 = [key for key in quant_params.keys() if "_1" in key and "bias" in key][0]

    key_mat_2 = [key for key in quant_params.keys() if "_2" in key and "weight" in key][0]
    key_bias_2 = [key for key in quant_params.keys() if "_2" in key and "bias" in key][0]

    # shape: (nodes, features) or (trees * nodes, features)
    mat_1 = quant_params[key_mat_1]

    # shape: (nodes, 1) or (trees * nodes, 1)
    bias_1 = quant_params[key_bias_1]

    # shape: (trees, leaves, nodes)
    mat_2 = quant_params[key_mat_2]

    # shape: (leaves, 1) or (trees * leaves, 1)
    bias_2 = quant_params[key_bias_2]

    n_features = mat_1.shape[1]
    n_nodes = mat_2.shape[2]
    n_leaves = mat_2.shape[1]

    mat_1 = mat_1.reshape(-1, n_nodes, n_features)
    bias_1 = bias_1.reshape(-1, 1, n_nodes)
    bias_2 = bias_2.reshape(-1, 1, n_leaves)

    required_onnx_operators = set(get_op_type(node) for node in onnx_model.graph.node)

    # If operator is `<`, np.less(x, y) is equivalent to:
    # round_bit_pattern((x - y) - half, lsbs_to_remove_for_trees=r) < 0.
    # Therefore, stage_1 = (q_x @ mat_1.transpose(0, 2, 1)) - bias_1
    if "Less" in required_onnx_operators:
        stage_1 = (q_x @ mat_1.transpose(0, 2, 1)) - bias_1
        matrix_q = stage_1 < 0

    # Else, if operator is `<=`, np.less_equal(x, y) is equivalent to:
    # round_bit_pattern((y - x) - half, lsbs_to_remove_for_trees=r) >= 0.
    # Therefore, stage_1 = bias_1 - (q_x @ mat_1.transpose(0, 2, 1))
    elif "LessOrEqual" in required_onnx_operators:
        stage_1 = bias_1 - (q_x @ mat_1.transpose(0, 2, 1))
        matrix_q = stage_1 >= 0

    else:  # pragma: no cover
        raise ValueError("Couldn't see if the comparison is 'Less' or 'LessOrEqual'")

    lsbs_to_remove_for_trees_stage_1 = get_lsbs_to_remove_for_trees(stage_1)

    # If operator is `==`, np.equal(x, y) is equivalent to:
    # round_bit_pattern((x - y) - half, lsbs_to_remove_for_trees=r) >= 0.
    # Therefore, stage_2 = bias_1 - (q_x @ mat_2.transpose(0, 2, 1))
    stage_2 = ((bias_2 - matrix_q @ mat_2.transpose(0, 2, 1))).max(axis=0)

    lsbs_to_remove_for_trees_stage_2 = get_lsbs_to_remove_for_trees(stage_2)

    return (lsbs_to_remove_for_trees_stage_1, lsbs_to_remove_for_trees_stage_2)


# pylint: disable=too-many-locals,too-many-statements,too-many-branches
def onnx_fp32_model_to_quantized_model(
    onnx_model: onnx.ModelProto,
    n_bits: int,
    framework: str,
    expected_number_of_outputs: int,
    n_features: int,
    model_inputs: Optional[numpy.ndarray] = None,
):
    """Build a FHE-compliant onnx-model using a fitted scikit-learn model.

    Args:
        onnx_model (onnx.ModelProto): The fitted scikit-learn as a Hummingbird onnx model to convert
        n_bits (int): Number of bits to quantize the model. If an int is passed
            for n_bits, the value will be used for quantizing inputs and weights. If a dict is
            passed, then it should contain "op_inputs" and "op_weights" as keys with
            corresponding number of quantization bits so that:
            - op_inputs : number of bits to quantize the input values
            - op_weights: number of bits to quantize the learned parameters
        framework (str): either sklearn or xgboost
        expected_number_of_outputs (int): expected number of outputs
        n_features (int): number of features as inputs of the model
        model_inputs (Optional[numpy.ndarray]): optional dataset to use for quantization

    Returns:
        onnx.ModelProto: The converted onnx model
        Optional[Tuple[int, int]]: Least significant bits to remove
        list[UniformQuantizer]: inputs quantizers
        list[UniformQuantizer]: outputs quantizers
    """
    # Get feature -> thresholds mappings and threshold values
    weight_1 = numpy.empty((0,))
    bias_1 = numpy.empty((0,))
    bias_1_index = -1
    bias_1_name = ""

    for initializer_index, initializer in enumerate(onnx_model.graph.initializer):
        init_tensor = numpy_helper.to_array(initializer)
        if "weight_1" in initializer.name:
            # weight_1 is the feature node selector
            weight_1 = init_tensor.copy()
        elif "bias_1" in initializer.name:
            # bias _1 is the threshold tensor
            bias_1 = init_tensor.copy()
            bias_1_index = initializer_index
            bias_1_name = initializer.name

    assert bias_1_name
    assert bias_1_index >= 0
    assert weight_1.size != 0
    assert bias_1.size != 0

    # Compute input/threshold quantizers
    input_quantizers: List[UniformQuantizer] = []

    # Quantization of each feature in X
    for feature_index in range(n_features):

        # Get all thresholds for a given feature
        threshold_for_feature: numpy.ndarray = bias_1[weight_1[:, feature_index] == 1][:, 0]

        # Sorting threshold values makes things easier afterwards
        threshold_for_feature.sort()

        # All unique threshold values
        unique_threshold_for_feature_sorted = numpy.unique(threshold_for_feature)
        unique_threshold_for_feature_sorted.sort()
        num_unique_thresholds = len(unique_threshold_for_feature_sorted)

        if num_unique_thresholds >= 1:
            max_threshold_value = unique_threshold_for_feature_sorted.max()
            min_threshold_value = unique_threshold_for_feature_sorted.min()
        else:
            max_threshold_value = 1.0
            min_threshold_value = 0.0

        # We compute a epsilon such that we have one quantized value on each side of the range
        # This offset will either be a right or left offset according to the framework
        number_of_need_offset_values = 2
        if num_unique_thresholds == 0:
            epsilon = 1.0
        elif num_unique_thresholds == 1:
            epsilon = 1.0
        else:
            epsilon = (max_threshold_value - min_threshold_value) / (
                (2**n_bits) - number_of_need_offset_values
            )

        # Input quantizers based on thresholds
        if model_inputs is None:
            if num_unique_thresholds:
                min_quantization_value = min_threshold_value
                max_quantization_value = max_threshold_value
            else:
                min_quantization_value = 0
                max_quantization_value = 1.0

            if num_unique_thresholds == 1:
                # If there is only one threshold for this feature
                # We want the threshold to be in the middle of a quantization bin
                min_quantization_value -= epsilon
                max_quantization_value += epsilon
            elif framework == "xgboost":
                # XGBoost uses a < op so we must add a left offset
                min_quantization_value -= epsilon
            else:
                # scikit-learn uses =< op so we must add a right offset
                max_quantization_value += epsilon

        # Quantizer based on data
        else:
            min_quantization_value = model_inputs[:, feature_index].min()
            max_quantization_value = model_inputs[:, feature_index].max()

        min_quantization_value = float(min_quantization_value)
        max_quantization_value = float(max_quantization_value)
        input_quantizer = QuantizedArray(
            n_bits=n_bits,
            values=numpy.array([min_quantization_value, max_quantization_value]),
        ).quantizer
        input_quantizers.append(input_quantizer)

    # Convert thresholds to their quantized equivalent
    quantized_thresholds_array = numpy.empty(bias_1.shape, dtype=numpy.int64)
    dequantized_thresholds_array = numpy.empty(bias_1.shape, dtype=numpy.float64)

    for threshold_index, threshold_value in enumerate(bias_1[:, 0]):
        feature_index = int(weight_1[threshold_index, :].argmax())
        quantized_threshold_value = (
            input_quantizers[feature_index].quant(threshold_value).astype(numpy.int64)
        )
        dequantized_threshold_value = input_quantizers[feature_index].dequant(
            quantized_threshold_value
        )
        quantized_thresholds_array[threshold_index, 0] = quantized_threshold_value
        dequantized_thresholds_array[threshold_index, 0] = dequantized_threshold_value

    onnx_model.graph.initializer[bias_1_index].CopyFrom(
        numpy_helper.from_array(
            quantized_thresholds_array,
            bias_1_name,
        )
    )

    # Modify the graph inplace to keep only the parts that are of interest to us
    tree_onnx_graph_preprocessing(onnx_model, framework, expected_number_of_outputs)

    # Get the preprocessed tree predictions to replace the current
    # (non-quantized) values in the onnx_model.
    q_y = None
    for initializer_index, initializer in enumerate(onnx_model.graph.initializer):
        init_tensor = numpy_helper.to_array(initializer)
        if "weight_3" in initializer.name:
            # weight_3 is the prediction tensor
            # Here we quantize it
            q_y = preprocess_tree_predictions(init_tensor, n_bits)
            init_tensor_as_int = q_y.qvalues.astype(numpy.int64)
        else:
            init_tensor_as_int = init_tensor.astype(numpy.int64)
        assert (
            isinstance(init_tensor_as_int, numpy.ndarray)
            and init_tensor_as_int.dtype == numpy.int64
        )
        new_initializer = numpy_helper.from_array(init_tensor_as_int, initializer.name)
        onnx_model.graph.initializer[initializer_index].CopyFrom(new_initializer)

    # Convert the tree inference with Numpy operators
    enable_rounding = bool(int(os.environ.get("TREES_USE_ROUNDING", 1)))

    if not enable_rounding:
        warnings.simplefilter("always")
        warnings.warn(
            "Using Concrete tree-based models without the `rounding feature` is deprecated. "
            "Consider setting 'use_rounding' to `True` for making the FHE inference faster "
            "and key generation.",
            category=DeprecationWarning,
            stacklevel=2,
        )

    lsbs_to_remove_for_trees: Optional[Tuple[int, int]] = None

    assert q_y is not None
    output_quantizers = [q_y.quantizer]

    if enable_rounding:
        # Quantize data to compute lsbs to remove
        if model_inputs is None:
            # If we have no data we can just randomly generate a dataset
            assert isinstance(n_features, int)
            calibration_set_size = 1_000
            quantized_model_inputs = numpy.empty(
                (calibration_set_size, n_features), dtype=numpy.int64
            )
            for feature_index in range(n_features):
                min_value = input_quantizers[feature_index].rmin
                assert min_value is not None
                max_value = input_quantizers[feature_index].rmax
                assert max_value is not None
                quantized_model_inputs[:, feature_index] = (
                    input_quantizers[feature_index]
                    .quant(numpy.linspace(min_value, max_value, calibration_set_size))
                    .astype(numpy.int64)
                )
            quantized_model_inputs = numpy.random.permutation(quantized_model_inputs)
        else:
            # Quantize using the learned quantization parameters for each feature
            quantized_model_inputs = numpy.zeros_like(model_inputs, dtype=numpy.int64)
            for i, input_quantizer in enumerate(input_quantizers):
                quantized_model_inputs[:, i] = input_quantizer.quant(model_inputs[:, i])

        # Compute for tree-based models the LSB to remove in stage 1 and stage 2
        # First LSB refers to Less or LessOrEqual comparisons
        # Second LSB refers to Equal comparison
        assert quantized_model_inputs.dtype == numpy.int64
        lsbs_to_remove_for_trees = _compute_lsb_to_remove_for_trees(
            onnx_model, quantized_model_inputs
        )

        # mypy
        assert len(lsbs_to_remove_for_trees) == 2

    return onnx_model, lsbs_to_remove_for_trees, input_quantizers, output_quantizers
