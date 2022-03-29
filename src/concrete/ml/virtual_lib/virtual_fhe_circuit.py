"""Virtual FHECircuit code."""

from typing import Dict, List, Tuple, Union

import numpy
from concrete.common.data_types import Integer
from concrete.common.debugging import format_operation_graph
from concrete.common.fhe_circuit import FHECircuit
from concrete.common.operator_graph import OPGraph
from concrete.common.representation.intermediate import IntermediateNode

from ..common.debugging import assert_true


class VirtualFHECircuit(FHECircuit):
    """Class simulating FHECircuit behavior in the clear, without any actual FHE computations."""

    # TODO: https://github.com/zama-ai/concrete-ml-internal/issues/640
    # remove this once 640 is done
    _has_warned: bool = False

    def __init__(self, op_graph: OPGraph):
        super().__init__(op_graph, None)

    def get_max_bit_width(self) -> int:
        """Get the max bit width of the simulated circuit.

        Returns:
            int: the max bit width of the simulated circuit.
        """
        return self.check_circuit_uses_n_bits_or_less(None)[1]

    def check_circuit_uses_n_bits_or_less(
        self, max_authorized_bit_width: Union[int, None]
    ) -> Tuple[bool, int, str]:
        """Check the bitwitdh of intermediate operations are <= to max_authorized_bit_width.

        Args:
            max_authorized_bit_width (Union[int, None]): number of bits the circuit must not go
                over. If None is passed the check is always successful, returned as the first
                element of the return tuple.

        Returns:
            Tuple[bool, int, Dict[IntermediateNode, List[str]]]: returns a tuple containing in order
                a boolean to indicate if the circuit conforms to the limit, the observed maximum
                bit width and in case of failure the formatted graph with highlighted offending
                nodes.
        """
        max_bit_width = 0
        offending_nodes: Dict[IntermediateNode, List[str]] = {}
        for node in self.op_graph.graph.nodes:
            for value_out in node.outputs:
                assert_true(isinstance(value_out.dtype, Integer))
                current_node_out_bit_width = value_out.dtype.bit_width

                max_bit_width = max(max_bit_width, current_node_out_bit_width)

                if (
                    max_authorized_bit_width is not None
                    and current_node_out_bit_width > max_authorized_bit_width
                ):
                    offending_nodes[node] = [
                        f"This node is over the {max_bit_width} bits limit with "
                        f"{current_node_out_bit_width} bits"
                    ]

        check_ok = len(offending_nodes) == 0

        bit_width_to_report = (
            max_authorized_bit_width if max_authorized_bit_width is not None else max_bit_width
        )

        formatted_graph_on_failure = (
            f"Graph had all intermediate values with a bit width <= {bit_width_to_report}"
            if check_ok
            else format_operation_graph(self.op_graph, highlighted_nodes=offending_nodes)
        )

        return check_ok, max_bit_width, formatted_graph_on_failure

    def run(self, *args: List[Union[int, numpy.ndarray]]) -> Union[int, numpy.ndarray]:
        """Simulate the FHE evaluation of the class's OPGraph.

        Args:
            *args (List[Union[int, numpy.ndarray]]): inputs to the simulated circuit

        Returns:
            Union[int, numpy.ndarray]: evaluation result on clear computations
        """

        # TODO: https://github.com/zama-ai/concrete-ml-internal/issues/640
        # warnings are not currently well handled in our package, use warnings when 640 is done
        if not self._has_warned:
            print(
                f"/!\\ WARNING /!\\: You are using a {self.__class__.__name__} "
                "meaning the execution is not done in FHE but in clear.",
            )
            self._has_warned = True

        return self.op_graph(*args)
