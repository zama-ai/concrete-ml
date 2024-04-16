"""Optimization passes for QuantizedModules."""

from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple

import numpy

from ..common.debugging import assert_true
from .base_quantized_op import QuantizedMixingOp, QuantizedOp
from .quantized_module import QuantizedModule
from .quantized_ops import (
    QuantizedBrevitasQuant,
    QuantizedConv,
    QuantizedGemm,
    QuantizedMatMul,
    QuantizedRelu,
)

# A dictionary that contains for a quantized op a list of predecessor ops
# Each predecessor op is stored along with its output tensor name
PredecessorsType = DefaultDict[Optional[QuantizedOp], List[Tuple[Optional[QuantizedOp], str]]]

# A list of optimizable patterns. For a "Mixing" op that supports rounding accumulators
# we store a list of ops which contain information that allows us to
# compute the integer scaling factor for the Mixing op.
# The quantizer op of the input to the the Mixing op is stored in the second member of the tuple
PatternDict = Dict[QuantizedMixingOp, Tuple[List[Optional[QuantizedOp]], Optional[QuantizedOp]]]


class PowerOfTwoScalingRoundPBSAdapter:
    """Detect neural network patterns that can be optimized with round PBS."""

    SUPPORTED_ROUND_PBS_OPS = (QuantizedGemm, QuantizedMatMul, QuantizedConv)
    SUPPORTED_ROUND_PBS_OP_PREDECESSOR = {
        QuantizedBrevitasQuant: QuantizedRelu,
        QuantizedRelu: QuantizedMixingOp,
        QuantizedMixingOp: QuantizedBrevitasQuant,
    }

    def __init__(self, qmodule: QuantizedModule) -> None:
        self._qmodule = qmodule
        self._num_ignored_valid_patterns = 0

    @property
    def num_ignored_valid_patterns(self):
        """Get the number of optimizable patterns that were ignored.

        Patterns could be ignored since a number of rounding bits was
        set manually through the compilation function.

        Returns:
            result (int): number of patterns that could be optimized but were not
        """
        return self._num_ignored_valid_patterns

    def process(self) -> PatternDict:
        """Analyze an ONNX graph and detect Gemm/Conv patterns that can use RoundPBS.

        We want to detect a gemm/conv node whose weights/bias are Brevitas QAT, and whose
        input is produced by a Brevitas QAT node that is applied on the output of
        another Gemm/conv node. Optionally a Relu can be placed before this input
        quantization node.

        Nothing will be done if rounding is already specified.

        Returns:
            result (PatternDict): a dictionary containing for each Conv/Gemm node for which
                round PBS can be applied based on power-of-two scaling factors
        """

        # The Pattern can be described as follows
        # x = Quant(x) -> stored separately in the second member of the tuple in PatternDict
        # .... the following ops are stored in the List of the PatternDict
        # y = Gemm(x, w, b), with w, b produced by a Brevitas quant node
        #     +---> This is the node for which roundPBS can be adjusted
        # y = Relu(y)
        # y = Quant(y)
        # z = Gemm(y, w2, b2)  -> the output node of the pattern

        self._num_ignored_valid_patterns = 0

        predecessors = self.compute_op_predecessors()

        valid_paths = self.detect_patterns(predecessors)

        valid_paths = self.process_patterns(valid_paths)

        return valid_paths

    def compute_op_predecessors(self) -> PredecessorsType:
        """Compute the predecessors for each QuantizedOp in a QuantizedModule.

        Stores, for each quantized op, a list of quantized ops that produce its
        inputs. Currently only the first input of the operations is considered
        as it is, usually, the encrypted input.

        Returns:
            result (PredecessorsType): a dictionary containing a hierarchy of op
                predecessors
        """

        # Initialize the list of predecessors with tensors that are graph inputs
        predecessors: PredecessorsType = defaultdict(list)

        for node_inputs, node_op in self._qmodule.quant_layers_dict.values():
            # The first input node contains the encrypted data
            enc_input_node = node_inputs[0]

            assert_true(
                enc_input_node in self._qmodule.quant_layers_dict
                or enc_input_node in self._qmodule.ordered_module_input_names
            )
            pred = self._qmodule.quant_layers_dict.get(enc_input_node, (None, None))
            # Get the quantized op that produces the current op's input
            pred_with_output = (pred[1], enc_input_node)
            predecessors[node_op].append(pred_with_output)
        return predecessors

    def match_path_pattern(
        self,
        predecessors: PredecessorsType,
        nodes_in_path: List[Optional[QuantizedOp]],
        input_producer_of_path: Optional[QuantizedOp],
    ) -> bool:
        """Determine if a pattern has the structure that makes it viable for roundPBS.

        Args:
            predecessors (PredecessorsType): Module predecessor operation list
            nodes_in_path (List[QuantizedOp]): list of quantized ops in the pattern
            input_producer_of_path (Optional[QuantizedOp]): operation that produces the input

        Returns:
            result (bool): whether the pattern can be optimized
        """

        # Test if the list of operations in this pattern has not the right length
        if len(nodes_in_path) != 3:
            return False

        # If the input of this pattern is produced by a graph input then ignore it
        # as graph inputs are not always quantized with QAT. QAT networks
        # will have the input to the first gemm/conv op produced by a BrevitasQuant
        # op and it will be valid pattern
        if input_producer_of_path is None:
            return False

        for test_node in nodes_in_path:
            # Check the operations in the pattern are chained properly
            # for example if the Gemm op is preceded by a quantizer op, etc..
            for pattern_first, pattern_second in self.SUPPORTED_ROUND_PBS_OP_PREDECESSOR.items():
                pred_type = predecessors[test_node][0][0]
                if isinstance(test_node, pattern_first) and not isinstance(
                    pred_type, pattern_second
                ):
                    return False

        return True

    def detect_patterns(self, predecessors: PredecessorsType) -> PatternDict:
        """Detect the patterns that can be optimized with roundPBS in the QuantizedModule.

        Args:
            predecessors (PredecessorsType): Module predecessor operation list

        Returns:
            result (PatternDict): list of optimizable patterns
        """

        valid_paths: PatternDict = {}

        # pylint: disable-next=too-many-nested-blocks
        for _, node_op in self._qmodule.quant_layers_dict.values():
            # Only work with supported nodes that have a single
            # encrypted input (not supporting enc x enc matmul)
            if (
                isinstance(node_op, self.SUPPORTED_ROUND_PBS_OPS)
                and len(node_op.int_input_names) == 1
            ):
                prev_compatible_node_output = list(node_op.int_input_names)[0]
                if len(predecessors[node_op]) == 1:
                    back_node, back_node_output = predecessors[node_op][0]

                    # A pattern is a sequence of Gemm/Conv -> Relu -> Quant
                    # but we also need to store the Quant that quantizes
                    # the Gemm/Conv's input
                    nodes_in_path: List[Optional[QuantizedOp]] = []
                    integer_node_input_quant: Optional[QuantizedOp] = None

                    while back_node_output != prev_compatible_node_output:
                        assert back_node is not None
                        nodes_in_path.append(back_node)
                        assert_true(
                            back_node in predecessors,
                            "Power of Two adapter: Error during graph traversal",
                        )
                        # If multiple ops produced this node, the pattern is not matched

                        if len(predecessors[back_node]) == 1:
                            back_node, back_node_output = predecessors[back_node][0]

                            # Reached the previous integer node
                            if back_node_output == prev_compatible_node_output:
                                # The Gemm/Conv op that produces this integer node is the one
                                # onto which we apply the roundPBS optimization
                                nodes_in_path.append(back_node)
                                list_pred_of_path = predecessors[back_node]
                                if len(list_pred_of_path) == 1:
                                    integer_node_input_quant = list_pred_of_path[0][0]

                assert isinstance(node_op, QuantizedMixingOp)
                if self.match_path_pattern(predecessors, nodes_in_path, integer_node_input_quant):
                    # If rounding was manually set (usually globally for all layers)
                    # the do not override the requested number of rounding bits
                    # but keep statistics for testing purposes
                    path_start_node = nodes_in_path[-1]
                    assert isinstance(path_start_node, QuantizedMixingOp)
                    if path_start_node.rounding_threshold_bits is not None:
                        self._num_ignored_valid_patterns += 1
                    else:
                        valid_paths[path_start_node] = (nodes_in_path, integer_node_input_quant)
        return valid_paths

    def process_patterns(self, valid_paths: PatternDict) -> PatternDict:
        """Configure the rounding bits of roundPBS for the optimizable operations.

        Args:
            valid_paths (PatternDict): list of optimizable patterns

        Returns:
            result (PatternDict): list of patterns actually optimized with roundPBS
        """

        def integer_log2(value: float) -> Tuple[int, bool]:
            """Compute the log2 of the value and tests if its an integer.

            Args:
                value (float): the value for which to take the log2

            Returns:
                result: The integer log2 and a bool indicating whether
                    the input value was an integer power of two
            """
            log2_value = int(numpy.rint(numpy.log2(value)))
            # Check that the integer power of two is close to the original value
            # with a small percentage tolerance
            if numpy.allclose(numpy.power(2.0, log2_value), value, rtol=0.01):
                return log2_value, True
            return 0, False

        invalid_paths: List[QuantizedMixingOp] = []
        for path_start_node, (path, path_input_quant) in valid_paths.items():
            # Placeholders
            scale_input, scale_output, scale_weights = None, None, None
            # Populate placeholders
            for node in path:
                if isinstance(node, self.SUPPORTED_ROUND_PBS_OPS):
                    # Get the scale of the input of the Gemm/Conv node
                    # and of its weights
                    assert path_input_quant is not None
                    scale_input = path_input_quant.constant_inputs[1]
                    scale_weights = node.constant_inputs[1].quantizer.scale
                elif isinstance(node, QuantizedBrevitasQuant):
                    # Get the output scale that will be used to
                    # compute the compounded scale factor of the
                    # node that will apply roundPBS
                    scale_output = node.constant_inputs[1]

            # Check placeholders
            assert scale_input is not None, (
                "Power of two adapter: Can not determine input scale of pattern",
            )
            assert scale_weights is not None, (
                "Power of two adapter: Can not determine weight scale of pattern",
            )
            assert scale_output is not None, (
                "Power of two adapter: Can not determine output scale of pattern",
            )

            # Check if power of two
            log2_input, ok_input = integer_log2(scale_input)
            log2_weights, ok_weights = integer_log2(scale_weights)
            log2_output, ok_output = integer_log2(scale_output)

            # Modify rounding
            if ok_input and ok_weights and ok_output:
                assert_true(
                    path_start_node.rounding_threshold_bits is None,
                    "Power of two adapter: a global rounding configuration was unexpected here",
                )
                # The total scale factor is multiplied with the accumulator
                # but we want to use a division with a power-of-two (shift right)
                # operation to perform the scaling. Thus the
                # number of lsbs to round is the negative of the sum of log2
                # of the scale factors
                lsbs_to_round = -(log2_input + log2_weights - log2_output)
                if lsbs_to_round > 0:
                    path_start_node.rounding_threshold_bits = lsbs_to_round
                    # For mypy
                    assert isinstance(path_start_node.lsbs_to_remove, dict)
                    path_start_node.lsbs_to_remove["matmul"] = lsbs_to_round
            else:
                invalid_paths.append(path_start_node)

        for node in invalid_paths:
            valid_paths.pop(node)

        return valid_paths
