"""Graph pre-processor for automatic rounding.
"""

from copy import deepcopy
from typing import Callable, Dict, Optional, Union

import networkx as nx
import numpy as np
from concrete.fhe import Exactness, round_bit_pattern
from concrete.fhe.dtypes import Integer
from concrete.fhe.representation import Graph, GraphProcessor, Node, Operation


def is_node_tlu(node: Node) -> bool:
    """Determine if a graph node is a table lookup.

    Args:
        node (Node): graph node to check

    Returns:
        bool: boolean indicating whether the node is a TLU
    """
    return node.converted_to_table_lookup


def bit_width(x):
    return Integer.that_can_represent(x).bit_width


def add_rounding_node(
    a_node: Node,
    lsbs_to_remove: int,
    graph: nx.DiGraph,
    rounding_function: Callable = round_bit_pattern,
    exactness=Exactness.EXACT,
    overflow_protection: bool = False,
) -> Node:
    """Modify a computation graph to include a rounding node.

    Args:
        a_node (Node): the node whose output will be rounded
        lsbs_to_remove (int): the number of least significant bits to remove
        graph (nx.DiGraph): the graph containing the node
        rounding_function (Callable): the function to use for rounding
        exactness: FHE rounding mode, either Exactness.EXACT or Exactness.APPROXIMATE
        overflow_protection (bool): use FHE overflow protection

    Returns:
        Node: the rounding node that was added to the graph
    """

    if lsbs_to_remove <= 0:
        # No rounding node to add
        return a_node

    # Sanity check and mypy check
    assert isinstance(a_node.output.dtype, Integer)

    # Adding rounding node
    rounding_kwargs: Dict[str, Union[Exactness, int, bool]] = {
        "lsbs_to_remove": lsbs_to_remove,
        "overflow_protection": overflow_protection,
    }
    attributes = {
        "overflow_protection": overflow_protection,
    }

    # Only round_bit_pattern support exactness for now
    if rounding_function.__name__ == "round_bit_pattern":
        rounding_kwargs["exactness"] = exactness

    output_value = deepcopy(a_node.output)
    new_bounds_arr = rounding_function(np.array(a_node.bounds, dtype=np.int64), **rounding_kwargs)
    output_value.dtype = Integer.that_can_represent(new_bounds_arr)

    rounding_node = Node.generic(
        name=rounding_function.__name__,
        inputs=[deepcopy(a_node.output)],
        output=output_value,
        operation=rounding_function,
        kwargs=rounding_kwargs,
        attributes=attributes,
    )
    rounding_node.properties["final_lsbs_to_remove"] = lsbs_to_remove
    rounding_node.properties["overflow_detected"] = False
    rounding_node.properties["original_rounded_bit_width"] = a_node.output.dtype.bit_width
    rounding_node.properties["overflow_protection"] = overflow_protection

    # Compute new bounds and bit-width
    assert a_node.bounds is not None
    rounding_node.bounds = (new_bounds_arr[0], new_bounds_arr[1])
    rounding_node.properties["resulting_bit_width"] = output_value.dtype.bit_width
    if output_value.dtype.bit_width > a_node.output.dtype.bit_width:
        rounding_node.properties["overflow_detected"] = True

    # Add edge between node and rounding node
    graph.add_edge(a_node, rounding_node, input_idx=0)

    # Replace a -> o_i by rounding_node -> o_i
    edges = list(graph.out_edges(nbunch=a_node))  # type: ignore
    for in_node, out_node in edges:
        if out_node == rounding_node:
            continue

        # We should preserve the input_idx
        edge_data: Dict[int, Dict[str, int]] = dict(graph.get_edge_data(in_node, out_node))
        graph.remove_edge(in_node, out_node)
        input_idx: int = edge_data[0]["input_idx"]
        graph.add_edge(rounding_node, out_node, input_idx=input_idx)

    return rounding_node


class InsertRounding(GraphProcessor):
    """
    InsertRounding graph processor, to add rounding before TLUs if desired.
    """

    rounding_threshold: Optional[int]

    def __init__(
        self,
        msbs_to_keep: Optional[int],
        exactness: Exactness = Exactness.APPROXIMATE,
        overflow_protection: bool = True,
        rounding_function=round_bit_pattern,
    ):
        self.rounding_threshold = msbs_to_keep
        self.exactness = exactness
        self.overflow_protection = overflow_protection
        self.rounding_function = rounding_function
        assert self.rounding_function.__name__ in {"round_bit_pattern", "truncate_bit_pattern"}
        if self.rounding_function.__name__ == "truncate_bit_pattern":
            assert exactness == Exactness.EXACT

    def apply(self, graph: Graph):
        if self.rounding_threshold is None:
            # No rounding if None
            return

        # Get all nodes that will be converted to LUTs
        tlu_nodes = graph.query_nodes(
            custom_filter=is_node_tlu,
            ordered=True,
        )

        for tlu_node in tlu_nodes:
            # Predecessor nodes of LUT node
            pred_nodes = graph.ordered_preds_of(tlu_node)

            # Only take into account predecessor's that aren't constants
            variable_input_indices = []
            for pred_index, pred_node in enumerate(pred_nodes):
                if pred_node.operation != Operation.Constant:
                    variable_input_indices.append(pred_index)

            # Only one input should be non-constant per LUT
            if len(variable_input_indices) != 1:
                continue

            pred_node = pred_nodes[variable_input_indices[0]]

            # Continue if the predecessor node is rounding node
            if pred_node.properties["name"] in {"round_bit_pattern", "truncate_bit_pattern"}:
                continue

            # Continue if the node itself is a rounding node
            if tlu_node.properties["name"] in {"round_bit_pattern", "truncate_bit_pattern"}:
                continue

            # Sanity check
            if not isinstance(pred_node.output.dtype, Integer):
                raise ValueError(f"{pred_node.output.dtype=} is not 'Integer'")

            if pred_node.output.dtype.bit_width <= self.rounding_threshold:
                # No need to do anything if the bit-width is actually lower or equal
                # to the rounding threshold value
                continue

            # Compute lsbs to remove
            lsbs_to_remove = pred_node.output.dtype.bit_width - self.rounding_threshold

            # Add rounding node
            add_rounding_node(
                pred_node,
                lsbs_to_remove=lsbs_to_remove,
                graph=graph.graph,
                rounding_function=self.rounding_function,
                exactness=self.exactness,
                overflow_protection=self.overflow_protection,
            )
