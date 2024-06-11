"""Graph pre-processors for automatic rounding."""

# todo: handle uint (basically anything strictly negative or strictly positive will break atm)
# todo: check that we match the target-bit-width and not have the +1 anymore
# todo: add support for max-rounding-bit-width too

from collections import Counter
from copy import deepcopy
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np
from concrete.fhe import Exactness, round_bit_pattern
from concrete.fhe.dtypes import Float, Integer
from concrete.fhe.representation import Graph, GraphProcessor, Node, Operation
from concrete.fhe.representation.evaluator import ConstantEvaluator, GenericEvaluator
from concrete.fhe.values.value_description import ValueDescription


def add(x, y):
    """Add the values of two graph values.

    Args:
        x: left operand
        y: right operand

    Returns:
        res: sum of inputs
    """
    return x + y


def subtract(x, y):
    """Subtract the values of two graph values.

    Args:
        x: left operand
        y: right operand

    Returns:
        res: difference of inputs
    """
    return x - y


def multiply(x, y):
    """Multiply element-wise the values of two graph values.

    Args:
        x: left operand
        y: right operand

    Returns:
        res: product of inputs
    """
    return x * y


def divide(x, y):
    """Divide element-wise the values of two graph values.

    Args:
        x: left operand
        y: right operand

    Returns:
        res: quotient of division of inputs
    """
    return x / y


def is_node_tlu(node: Node) -> bool:
    """Determine if a graph node is a table lookup.

    Args:
        node (Node): graph node to check

    Returns:
        bool: boolean indicating whether the node is a TLU
    """
    return node.converted_to_table_lookup


class CycleDetector(GraphProcessor):
    """A graph processor that checks if cycles are found in graph."""

    def __init__(self):
        pass

    def apply(self, graph: Graph):
        """Check if the graph contains cycles.

        Args:
            graph (Graph): operation graph to analyze

        Raises:
            Exception: if the graph contains cycles
        """
        # Get all nodes that will be converted to LUTs
        tlu_nodes = graph.query_nodes(
            custom_filter=is_node_tlu,
        )
        cycles = nx.recursive_simple_cycles(graph.graph)
        if cycles:
            raise Exception()

        for tlu_node in tlu_nodes:
            evaluator = tlu_node.evaluator
            assert isinstance(evaluator, GenericEvaluator)

            # We check only on subgraphs
            if "subgraph" in evaluator.properties["kwargs"]:
                tlu_subgraph: Graph = evaluator.properties["kwargs"]["subgraph"]
                cycles = nx.recursive_simple_cycles(tlu_subgraph.graph)
                if cycles:
                    raise Exception()


def vectorized_graph_eval(
    graph: Graph,
    *inputs: np.ndarray,
    sorted_nodes: List,
    input_indices: Union[None, int, np.ndarray] = None,
) -> Union[Any, np.ndarray]:
    """Compute the output of a subgraph on a tensor input.

    Args:
        graph (Graph): the computation graph to evaluate
        inputs (Tuple[np.ndarray,...]): list of inputs to the graph
        sorted_nodes (List): the graph nodes in sorted order
        input_indices (Optional[np.ndarray]): indices that map the input list to the graph inputs

    Returns:
        Union[Any, np.ndarray]: tensor with the result of the graph on the inputs

    Raises:
        RuntimeError: if the subgraph could not be evaluated
    """

    node_results: Dict[Node, Union[np.bool_, np.integer, np.floating, np.ndarray]] = {}

    for node in sorted_nodes:
        if node.operation == Operation.Input:
            # Deepcopy on the fhe.Graph doesn't modify the input node/indices mappings
            indices = input_indices if input_indices is not None else graph.input_indices[node]
            node_results[node] = node.evaluator(inputs[indices])
            continue

        pred_results = [deepcopy(node_results[pred]) for pred in graph.ordered_preds_of(node)]

        node_results[node] = node.evaluator(*pred_results)

    result = tuple(node_results[node] for node in graph.ordered_outputs())
    assert len(result) > 0, "Empty results"
    return result if len(result) > 1 else result[0]


def merge_tlu_constant_shapes(constant_shapes: List[Tuple[int, ...]]) -> Tuple[int, ...]:
    """Determine the maximally broadcasted shape of a subgraph output.

    The analysis is based on the constants that the TLU uses.

    Args:
        constant_shapes (List[Tuple[int, ...]]): list of tuples containing the shapes of
            constants found in a subgraph.

    Returns:
        res: tuple containing the shape of a tensor upon which a TLU can be evaluated
    """

    # For each axis take the max value for the constant
    if not constant_shapes:
        return tuple()

    n_dims_max = max(len(constant_shape) for constant_shape in constant_shapes)    
    fixed_constant_shapes = []
    for constant_shape in constant_shapes:
        if len(constant_shape) < n_dims_max:
            fixed_shape = tuple([1] * (n_dims_max - len(constant_shape)) + list(constant_shape))
            fixed_constant_shapes.append(fixed_shape)
        else:
            fixed_constant_shapes.append(constant_shape)
    fixed_constant_shapes = np.asarray(fixed_constant_shapes)
    
    merged_shape = np.max(fixed_constant_shapes, axis=0)
    for dim in range(len(merged_shape)):
        unique_vals = np.unique(fixed_constant_shapes[:,dim])
        assert len(unique_vals) <= 2

    return tuple(merged_shape)
#    return tuple(
#        max(constant_shape[idx] for constant_shape in constant_shapes if len(constant_shape) > idx)
#        for idx in range(max(len(elt) for elt in constant_shapes))
#    )


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
        # No  rounding node to add
        return a_node

    # Adding rounding node
    assert isinstance(a_node.output.dtype, Integer)
    rounding_kwargs = {
        "lsbs_to_remove": lsbs_to_remove,
    }
    attributes = {}
    if rounding_function.__name__ == "round_bit_pattern":
        # These kwargs are not supported atm
        rounding_kwargs["exactness"] = exactness
        rounding_kwargs["overflow_protection"] = overflow_protection
        attributes = {
            "overflow_protection": overflow_protection,
        }
    output_value = deepcopy(a_node.output)
    output_value.dtype = Integer.that_can_represent(
        round_bit_pattern(
            np.array(a_node.bounds, dtype=np.int64),
            exactness=exactness,
            overflow_protection=overflow_protection,
            lsbs_to_remove=lsbs_to_remove,
        )
    )

    rounding_node = Node.generic(
        name=rounding_function.__name__,
        inputs=[deepcopy(output_value)],
        output=output_value,
        operation=rounding_function,
        kwargs=rounding_kwargs,
        attributes=attributes,
    )
    rounding_node.properties["final_lsbs_to_remove"] = lsbs_to_remove
    rounding_node.properties["resulting_bit_width"] = output_value.dtype.bit_width - lsbs_to_remove
    rounding_node.properties["overflow_detected"] = False
    rounding_node.properties["original_rounded_bit_width"] = a_node.output.dtype.bit_width
    if rounding_function.__name__ == "round_bit_pattern":
        rounding_node.properties["overflow_protection"] = overflow_protection
        rounding_node.properties["exactness"] = exactness

    # todo: fix this
    rounding_node.bounds = a_node.bounds  # Might be over/under-estimated

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


def add_leveled_op_with_cst(
    a_node: Node,
    b: Union[np.number, np.ndarray],
    function: Callable[[np.ndarray, np.ndarray], np.ndarray],
    graph: nx.DiGraph,
) -> Node:
    """Add a levelled encrypted operation with a constant to a graph.

    Can also add a float operation to a TLU subgraph.

    Args:
        a_node (Node): The node whose output to apply the new op to
        b (Union[int, np.ndarray]): The constant to use in the operation
        function (Callable[[np.ndarray, np.ndarray], np.ndarray]): the operation's function
        graph (nx.DiGraph): The graph which contains the node to modify

    Returns:
        res (Node): the levelled op node

    Raises:
        ValueError: if the inputs are not of the correct type
    """
    assert isinstance(a_node, Node)

    # When processing a single subgraph b is a float scalar, but when
    # we're processing the main graph, b is an int or an array corresponding
    # to a broadcasted value
    assert isinstance(b, (float, int, np.int64, np.float64)) or (
        isinstance(b, np.ndarray) and b.dtype in (np.float64, np.int64)
    ), f"Constant {b} should be of dtype np.int64 or np.float64, not {b.dtype}"

    assert isinstance(a_node.output.dtype, (Float, Integer))

    constant_node = Node.constant(b)

    # Handle dtype
    if b.dtype == np.float64:
        constant_dtype = Float(64)
        result_dtype = Float(64)
        res_bounds = np.array([0, 0], dtype=np.float64)
    elif b.dtype == np.int64:
        # Compute bounds
        assert isinstance(a_node.bounds, tuple) and len(a_node.bounds) == 2, (
            f"{a_node.bounds=} from {a_node.properties['name']=} is "
            "not a Tuple or doesn't have length 2"
        )
        some_inputs = np.zeros((2,) + a_node.output.shape)
        some_inputs[0] = a_node.bounds[0]  # min
        some_inputs[1] = a_node.bounds[1]  # max
        results = function(np.array(some_inputs), np.asarray(b)[np.newaxis, ...])
        res_bounds = np.array(
            [results.min(), results.max()],
            dtype=np.int64,
        )
        constant_dtype = Integer.that_can_represent(b)
        result_dtype = Integer.that_can_represent(results.astype(np.int64))
    else:
        raise ValueError(f"{b.dtype=} is not supported")

    constant_node.output = ValueDescription(
        dtype=constant_dtype,
        is_encrypted=False,
        shape=b.shape,
    )

    assert isinstance(constant_node.output.dtype, (Float, Integer))

    # Create op node
    new_node = Node.generic(
        name=function.__name__,
        inputs=[
            deepcopy(a_node.output),
            deepcopy(constant_node.output),
        ],
        output=ValueDescription(
            dtype=result_dtype,
            shape=a_node.output.shape,
            is_encrypted=a_node.output.is_encrypted,
        ),
        operation=function,
    )
    x_min: int = res_bounds[0]
    x_max: int = res_bounds[1]
    new_node.bounds = (x_min, x_max)

    # Create new edges
    graph.add_edge(a_node, new_node, input_idx=0)
    graph.add_edge(constant_node, new_node, input_idx=1)

    # Replace a -> o_i by new_node -> o_i
    edges = list(graph.out_edges(a_node))  # type: ignore
    for in_node, out_node in edges:
        if out_node == new_node:
            continue
        # We should preserve the input_idx
        edge_data: Dict[int, Dict[str, int]] = dict(graph.get_edge_data(in_node, out_node))
        graph.remove_edge(in_node, out_node)
        input_idx: int = edge_data[0]["input_idx"]
        graph.add_edge(new_node, out_node, input_idx=input_idx)

    return new_node


def add_clipping(
    a_node: Node,
    x_min: Union[np.number, np.ndarray],
    x_max: Union[np.number, np.ndarray],
    graph: nx.DiGraph,
) -> Node:
    if x_min is None and x_max is None:
        return a_node
    elif x_min is None:
        raise ValueError
    elif x_max is None:
        raise ValueError
    assert isinstance(a_node, Node)
    # When processing a single subgraph b is a float scalar, but when
    # we're processing the main graph, b is an int or an array corresponding
    # to a broadcasted value
    assert isinstance(x_min, (float, int, np.int64, np.float64)) or (
        isinstance(x_min, np.ndarray) and x_min.dtype in (np.float64, np.int64)
    ), f"Constant {x_min} should be of dtype np.int64 or np.float64, not {x_min.dtype}"
    assert isinstance(x_max, (float, int, np.int64, np.float64)) or (
        isinstance(x_max, np.ndarray) and x_max.dtype in (np.float64, np.int64)
    ), f"Constant {x_max} should be of dtype np.int64 or np.float64, not {x_max.dtype}"

    assert isinstance(a_node.output.dtype, (Float, Integer))

    constant_node_x_min = Node.constant(x_min)
    constant_node_x_max = Node.constant(x_max)

    assert x_min.dtype == x_max.dtype
    bounds_dtype = x_min.dtype

    # Handle dtype
    if bounds_dtype == np.float64:
        # If we have floats we are in a TLU  so we don't really care about the bounds
        constant_dtype = Float(64)
        result_dtype = Float(64)
        res_bounds = np.array([0, 0], dtype=np.float64)
    elif bounds_dtype == np.int64:
        # Compute bounds
        assert isinstance(a_node.bounds, tuple) and len(a_node.bounds) == 2, (
            f"{a_node.bounds=} from {a_node.properties['name']=} is "
            "not a Tuple or doesn't have length 2"
        )
        bounds = (
            np.max(a_node.bounds[0], np.min(x_min)),
            np.min(a_node.bounds[1], np.max(x_max)),
        )  # max
        res_bounds = np.array(
            bounds,
            dtype=np.int64,
        )
        constant_dtype = Integer.that_can_represent(np.array([x_min, x_max], dtype=np.int64))
        result_dtype = Integer.that_can_represent(res_bounds.astype(np.int64))
    else:
        raise ValueError(f"{bounds_dtype.dtype=} is not supported")

    constant_node_x_min.output = ValueDescription(
        dtype=constant_dtype,
        is_encrypted=False,
        shape=x_min.shape,
    )
    constant_node_x_max.output = ValueDescription(
        dtype=constant_dtype,
        is_encrypted=False,
        shape=x_max.shape,
    )

    assert isinstance(constant_node_x_max.output.dtype, (Float, Integer))
    assert isinstance(constant_node_x_min.output.dtype, (Float, Integer))

    # Create op node
    function = np.clip
    new_node = Node.generic(
        name=function.__name__,
        inputs=[
            deepcopy(a_node.output),
            deepcopy(constant_node_x_min.output),
            deepcopy(constant_node_x_max.output),
        ],
        output=ValueDescription(
            dtype=result_dtype,
            shape=a_node.output.shape,
            is_encrypted=a_node.output.is_encrypted,
        ),
        operation=function,
    )
    new_node.bounds = (res_bounds[0], res_bounds[1])

    # Create new edges
    graph.add_edge(a_node, new_node, input_idx=0)
    graph.add_edge(constant_node_x_min, new_node, input_idx=1)
    graph.add_edge(constant_node_x_max, new_node, input_idx=2)

    # Replace a -> o_i by new_node -> o_i
    edges = list(graph.out_edges(a_node))  # type: ignore
    for in_node, out_node in edges:
        if out_node == new_node:
            continue
        # We should preserve the input_idx
        edge_data: Dict[int, Dict[str, int]] = dict(graph.get_edge_data(in_node, out_node))
        graph.remove_edge(in_node, out_node)
        input_idx: int = edge_data[0]["input_idx"]
        graph.add_edge(new_node, out_node, input_idx=input_idx)

    return new_node


def argmin(d):
    if not d:
        return None
    min_val = min(d.values())
    return [k for k in d if d[k] == min_val][0]


def scale_up(
    x: np.ndarray,
    scaling_factor: Union[int, np.ndarray] = 1,
    bias: Union[int, np.ndarray] = 0,
):
    return (x * scaling_factor) - bias


def scale_down(
    x: np.ndarray,
    scaling_factor: Union[int, np.ndarray] = 1,
    bias: Union[int, np.ndarray] = 0,
):
    return (x + bias) / scaling_factor


# Faster more explicit implementation of rounding
def truncation(x, lsbs_to_remove: int):
    return x - (x % 2**lsbs_to_remove)


def rounding(x, lsbs_to_remove: int):
    offsetted = x + 2 ** (lsbs_to_remove - 1)
    return offsetted - (offsetted % 2**lsbs_to_remove)


def bit_width(x):
    return Integer.that_can_represent(x).bit_width


def transform_inputs(
    input_range: np.ndarray,
    inputset: np.ndarray,
    msbs_to_keep: int,
    scaling_factor=1,
    bias=0,
):
    # TODO: add some asserts based on the expected bit-width of the range
    scaled_bit_width = bit_width(scale_up(input_range, scaling_factor=scaling_factor, bias=bias))
    lsbs_to_remove = scaled_bit_width - msbs_to_keep
    return scale_down(
        rounding(
            scale_up(inputset - 1, scaling_factor=scaling_factor, bias=bias),
            lsbs_to_remove=lsbs_to_remove,
        ),
        scaling_factor=scaling_factor,
        bias=bias,
    )


def find_msbs_to_keep(inputset: np.ndarray, thresholds: np.ndarray, deltas: np.ndarray) -> int:

    # todo: this should be reworked since we now update msbs-to-keep in the bias computation
    msbs_to_keep_set = set()

    for delta in deltas:

        # How many elements it takes to divide [x-min, x-max] in parts of size delta
        # then take the log2 of that
        msbs_to_keep = np.ceil(
            np.log2(np.ceil((inputset.max() - inputset.min() + 1) / delta))
        ).astype(np.int64)

        # Off-setting the values so that the thresholds are on a proper power of two
        # can result in an added bit
        lsbs_to_remove = int(bit_width(inputset) - msbs_to_keep)
        if lsbs_to_remove:
            for offset in np.unique(
                thresholds - rounding(thresholds, bit_width(inputset) - msbs_to_keep)
            ):
                msbs_to_keep = np.ceil(
                    np.log2(
                        np.ceil(((inputset.max() - inputset.min() + 1) + np.abs(offset)) / delta)
                    )
                ).astype(np.int64)

                msbs_to_keep_set.add(msbs_to_keep)
        else:
            msbs_to_keep_set.add(msbs_to_keep)

    msbs_to_keep = max(msbs_to_keep_set)

    return msbs_to_keep


def compute_scaling_factor(deltas, target_bit_width, msbs_to_keep):
    a_candidates = Counter()
    for delta in deltas:
        a_candidate = np.floor(2 ** (target_bit_width - msbs_to_keep) / delta).astype(np.int64)
        a_candidates[a_candidate] += 1

        a_candidate = np.ceil(2 ** (target_bit_width - msbs_to_keep) / delta).astype(np.int64)
        a_candidates[a_candidate] += 1
    return a_candidates.most_common(1)[0][0]


def bias_closed_form(
    input_range: np.ndarray,
    msbs_to_keep: int,
    thresholds: np.ndarray,
    scaling_factor: int = 0,
):
    # todo: somehow this isn't the optimal bias solution as shown by grid-searching the optimal parameter
    # could be worth adding some exploration around it
    # the first step could be to check for the step size of the error(bias) function
    # nothing really explicit came to mind just yet

    # Find each threshold (ideally all thresholds) on the same value even with thresholds

    scaled_bit_width = bit_width(input_range)  # Range bit-width
    lsbs_to_remove = scaled_bit_width - msbs_to_keep  # Lsbs to remove

    rounded_thresholds = rounding(thresholds, lsbs_to_remove)

    threshold_diff = thresholds - rounded_thresholds

    # If the range is skewed left or right we should either remove 2**(lsbs-1) or add it
    if np.mean(input_range) >= 0:
        func = add
    else:
        func = subtract

    # todo: remove this print debug statement
    # print(f"debug: {func.__name__=}")

    # todo
    # I somehow have an offset by one to the right that needs to be fixed, maybe a - scaling factor would fix it

    # Should we sub or add the scaling factor -> probably depends on the sign of the mean after the first bias
    # bias = np.floor(
    #     func(func(np.mean(threshold_diff), int(2 ** (lsbs_to_remove - 1))), np.ceil(scaling_factor/2))
    # ).astype(np.int64)

    # todo: check if there isn't a better bias?
    # I feel like somewhere around this value there is a better solution
    # mean really?
    # ceil/floor/rint?
    # also is it - or + scaling factor
    # exhaustive search in this range
    # there is some factor here that I need to fix 2**(lsbs-2) is weird -> would probably be easier with truncation
    bias = np.rint(
        func(
            np.mean(threshold_diff),
            # func(
            (int(2 ** (lsbs_to_remove - 1))),
            #     scaling_factor - int(2 ** (lsbs_to_remove - 2)),
            # ),
        )
    ).astype(np.int64)

    # todo: remove the need for this
    # hackish -> breaks the target-bit-width assumption
    if bit_width(input_range - bias) != bit_width(input_range):
        msbs_to_keep += 1
        print(f"debug: adding 1 to {msbs_to_keep=}")

    return bias, msbs_to_keep


def find_best_params(target, input_range: np.ndarray, target_bit_width: int = 24):
    # Compute the function over the full range
    inputset = np.arange(input_range[0], input_range[1] + 1, 1)
    assert len(target) == len(inputset), f"{len(target)=} != {len(inputset)=}"

    # Compute thresholds and deltas
    change_mask = np.concatenate([[False], np.diff(target).astype(bool)])
    thresholds = inputset[change_mask]
    deltas = np.diff(thresholds)
    # todo: remove this print debug statement
    # print(f"debug: {thresholds=}, {deltas=}")
    # Compute msbs_to_keep
    msbs_to_keep = find_msbs_to_keep(inputset, thresholds, deltas)
    scaling_factor = compute_scaling_factor(deltas, target_bit_width, msbs_to_keep)
    bias, msbs_to_keep = bias_closed_form(
        input_range=input_range * scaling_factor,
        msbs_to_keep=msbs_to_keep,
        thresholds=thresholds * scaling_factor,
        scaling_factor=scaling_factor,
    )
    return bias, scaling_factor, msbs_to_keep


def scale_and_round(x: np.ndarray, scaling_factor=1, bias=0, msbs_to_keep: int = 1):
    scaled_up = scale_up(x, scaling_factor=scaling_factor, bias=bias)
    acc_bit_width = bit_width(scaled_up)
    lsbs_to_remove = acc_bit_width - msbs_to_keep
    rounded = rounding(scaled_up, lsbs_to_remove=lsbs_to_remove)
    scaled_down = scale_down(rounded, scaling_factor=scaling_factor, bias=bias)
    return scaled_down


def delta_optimize(
    subgraph_inputs: np.ndarray,
    subgraph_outputs: np.ndarray,
    shape_: Tuple[int, ...],
    bounds: Tuple[int, int],
    target_bit_width: int,
    rounding_function: Callable = round_bit_pattern,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Optimize a TLU by analyzing the steps (deltas) in the output of the TLU function.

    Args:
        subgraph_inputs (np.ndarray): The possible inputs of the
            TLU subgraph, determined using calibration
        reference (np.ndarray): The TLU outputs on the possible inputs of the
            original TLU without rounding
        shape_ (Tuple[int, ...]): the shape of constants that broadcast to the inputs of the TLU
        bounds (Tuple[int, int]): calibrated min/max of the possible inputs
        tlu_subgraph (nx.DiGraph): the subgraph that contains the TLU operations
        raised_bit_width (int): the desired raised bit_width used for approximate rounding
        rounding_function (Callable): the function used to round down the raised precision

    Returns:
        tuple: scaling factors, offsets and global tensor rounding bits

    Raises:
        ValueError: if the rounding function is not supported
    """

    assert (
        rounding_function.__name__ == "round_bit_pattern"
    ), "Only round is supported for TLU adjustment"

    x_min, x_max = bounds
    assert x_min < x_max
    assert subgraph_inputs.min() == x_min, f"{subgraph_inputs.min()} != {x_min}"
    assert subgraph_inputs.max() == x_max, f"{subgraph_inputs.max()} != {x_max}"

    if (x_min <= 0) == (x_max <= 0):
        raise ValueError(f"Same sign bounds is not supported yet {x_min=}, {x_max=}")

    # Initialize a and b such that no changes are done
    best_a = np.ones(shape_, dtype=np.int64)
    best_b = np.zeros(shape_, dtype=np.int64)

    ref_diff = np.diff(subgraph_outputs, axis=0).astype(bool)

    # Compute mask of values for which there is a change
    change_mask = np.concatenate(
        [
            np.zeros(subgraph_outputs[:1].shape).astype(bool),
            ref_diff.astype(bool),
        ]
    ).astype(bool)

    # Some accumulators
    best_lsbs = np.ones(shape_, dtype=np.int64)
    best_msbs = np.ones(shape_, dtype=np.int64)
    best_acc = np.ones(shape_, dtype=np.int64)

    # This doesn't pass
    # assert subgraph_inputs.shape == subgraph_outputs.shape, f"{subgraph_inputs.shape=} != {subgraph_outputs.shape=}"

    # We create the input_range object here but we should probably have it per element
    # instead of per tensor as we have now
    input_range = np.array([bounds[0], bounds[1]], dtype=np.int64)

    # For each axis along which the TLU is different we should compute the best set of parameters
    for indexes in product(*[range(elt) for elt in shape_]):
        triggered = False
        best_indexes = tuple([*indexes])
        selection = tuple([slice(0, subgraph_outputs.shape[0]), *indexes[1:]])

        # Slice things up
        subgraph_inputs_selected = subgraph_inputs[selection]
        subgraph_outputs_selected = subgraph_outputs[selection]
        change_mask_selection = change_mask[selection]
        thresholds_selected = subgraph_inputs_selected[change_mask_selection]

        assert len(thresholds_selected) == len(
            np.unique(thresholds_selected)
        ), "not unique steps, shouldn't occur"

        # todo: maybe we should wrap this if-else thingy in `find_best_params`?
        # todo: support corner-cases

        # 0-jump
        if len(thresholds_selected) == 0:
            # constant tlu
            msbs_to_keep = 1
            acc_size = bit_width(input_range)
            lsbs_to_remove = acc_size - msbs_to_keep
            scaling_factor = 1
            bias = 0

        # 1-jump
        elif len(thresholds_selected) == 1:
            # single-jump tlu

            # todo: atm we are filing the values in the range with the one from the value even if outside the range
            # this leads to a wrong behavior in some situations
            # i don't know if there is a way to fix that from our side without changing the filling behavior
            # of concrete-python

            # todo: verify
            msbs_to_keep = 1
            scaling_factor = 1
            bias = thresholds_selected[0] + (2 ** (bit_width(input_range)))
            acc_size = bit_width(scale_up(input_range, scaling_factor=scaling_factor, bias=bias))
            lsbs_to_remove = acc_size - msbs_to_keep

        else:
            deltas = np.diff(thresholds_selected).astype(np.int64)
            assert len(deltas) > 0
            if (
                np.all(deltas == 1)
                and len(thresholds_selected) == len(subgraph_inputs_selected) - 1
            ):
                # all delta-1 jumps taking the full range

                # If all values are different in the range then maybe there is nothing that can be done
                # todo: check if correct
                acc_size = bit_width(input_range)
                msbs_to_keep = acc_size
                lsbs_to_remove = 0
                scaling_factor = 1
                bias = 0
            else:
                # 'normal' tlu
                # todo: implement something that limits the bit the msbs_to_keep but still
                # finds adequate parameters
                # todo: check that optimization is indeed needed
                bias, scaling_factor, msbs_to_keep = find_best_params(
                    subgraph_outputs_selected,
                    input_range,
                    target_bit_width=target_bit_width,
                )
                acc_size = bit_width(scale_up(input_range, scaling_factor, bias))
                lsbs_to_remove = acc_size - msbs_to_keep

        # todo: re-activate this assert as this shouldn't happen
        if acc_size > target_bit_width:
            print(f"{acc_size=} > {target_bit_width=}")
            triggered = True

        # todo: remove this after check that the accuracy is fine
        if lsbs_to_remove and triggered:
            lsbs_to_remove -= 1

        (
            best_a[best_indexes],
            best_b[best_indexes],
            best_lsbs[indexes],
            best_msbs[indexes],
            best_acc[indexes],
        ) = (
            scaling_factor,
            bias,
            lsbs_to_remove,
            msbs_to_keep,
            acc_size,
        )

        # todo: remove debug print
        # print(
        #     f"{best_a[best_indexes]=}",
        #     f"{best_b[best_indexes]=}",
        #     f"{best_lsbs[indexes]=}",
        #     f"{best_msbs[indexes]=}",
        #     f"{best_acc[indexes]=}",
        # )

    # As rounding can be applied only for the entire tensor,
    # a single rounding threshold must be found. The smallest
    # number of lsbs to remove is used.
    if best_lsbs.shape:
        n_round_idx = np.argmax(best_msbs)
        lsbs_to_remove = best_lsbs.flatten()[n_round_idx]
        msbs_to_keep = best_msbs.flatten()[n_round_idx]
    else:
        lsbs_to_remove = best_lsbs
        msbs_to_keep = best_msbs

    lsbs_to_remove = int(lsbs_to_remove)
    msbs_to_keep = int(msbs_to_keep)

    return best_a, best_b, lsbs_to_remove, msbs_to_keep


def modify_subgraph_for_rounded_inputs(
    tlu_subgraph: Graph,
    a: np.ndarray,
    c: np.ndarray,
    rounding_node,
    x_min: Optional[np.ndarray] = None,
    x_max: Optional[np.ndarray] = None,
):
    """Modify a TLU subgraph to cancel out the rounding.

    A TLU will cancel out the rounding transformation by dividing
    its inputs by ``a`` and subtracting the offset ``c``.

    Args:
        tlu_subgraph (Graph): the subgraph that contains the TLU operations
        a (np.ndarray): broadcasted scaling factors
        c (np.ndarray): the offset of the rounding approximation
    """
    previous_node = get_subgraph_input(tlu_subgraph)

    if not previous_node.output.shape:
        if not isinstance(a, np.ndarray):
            a = np.array(a, dtype=np.int64)
        if not isinstance(c, np.ndarray):
            c = np.array(c, dtype=np.int64)

        assert a.shape == (1,) or a.shape == tuple(), f"{a.shape=}"
        assert c.shape == (1,) or c.shape == tuple(), f"{c.shape=}"
        if a.shape == (1,):
            a = a[0]
        if c.shape == (1,):
            c = c[0]

    c = np.array(c, dtype=np.int64)
    a = np.array(a, dtype=np.int64)

    # Add threshold offset
    if np.any(c != 0):
        previous_node = add_leveled_op_with_cst(
            previous_node, c.astype(np.float64), add, graph=tlu_subgraph.graph
        )

    # Divide by scaling factor
    if np.any(a != 1):
        previous_node = add_leveled_op_with_cst(
            previous_node, a.astype(np.float64), divide, graph=tlu_subgraph.graph
        )

    previous_node = add_clipping(
        previous_node,
        x_min=x_min.astype(np.float64),
        x_max=x_max.astype(np.float64),
        graph=tlu_subgraph.graph,
    )


def eval_subgraph_on_rounded_inputs(
    tlu_subgraph: Graph,
    subgraph_inputs: np.ndarray,
    rounding_function: Callable,
    a: np.ndarray,
    b: np.ndarray,
    lsbs_to_remove: int,
) -> np.ndarray:
    """Compute the output of a subgraph into which rounding is inserted.

    This simulates the result of applying the subgraph on inputs
    that are rounded. The subgraph is modified to cancel out the rounding.
    This function allows to evaluate a candidate scale (a) and offset (b).

    Args:
        tlu_subgraph (Graph): Executable subgraph to be evaluated
        subgraph_inputs (np.ndarray): Tensor with full range of possible inputs
        rounding_function (Callable): Rounding function to use to round the subgraph inputs
        a (np.ndarray): Scale of the rounding transformation
        b (np.ndarray): Offset of the rounding transformation
        lsbs_to_remove (int): Number of rounding bits

    Returns:
        np.ndarray: result of applying the rounded TLU
    """
    tlu_subgraph_copy = deepcopy(tlu_subgraph)
    modify_subgraph_for_rounded_inputs(tlu_subgraph_copy, a, b)

    # TODO: find a way to make sure that scaling is always the same in the code
    # i.e. not mix x * a - b and x - b * a
    rounded_inputs = (
        rounding_function(
            (subgraph_inputs * a) - b,
            lsbs_to_remove=int(lsbs_to_remove),
        ).astype(np.float64)
        + b
    ) / a

    # Get the input indices of the original graph
    sorted_nodes = list(nx.topological_sort(tlu_subgraph.graph))
    indices = None
    for node in sorted_nodes:
        if node.operation == Operation.Input:
            # Deepcopy on the fhe.Graph doesn't modify the input node/indices mappings
            indices = tlu_subgraph.input_indices[node]
            break

    # Reuse the input indices when evaluation the modified graph copy
    sorted_nodes = list(nx.topological_sort(tlu_subgraph_copy.graph))
    approx_reference = vectorized_graph_eval(
        tlu_subgraph_copy,
        rounded_inputs,
        sorted_nodes=sorted_nodes,
        input_indices=indices,
    )
    return approx_reference


def extract_tlu_input_bounds(variable_input_node: Node) -> Tuple[np.int64, np.int64]:
    """Extract TLU input bounds.

    Args:
        variable_input_node (Node): Tlu subgraph input node

    Returns:
        tuple: tuple of min/max values
    """

    assert variable_input_node.bounds is not None, "Bounds not found during TLU optimization"

    min_bound, max_bound = variable_input_node.bounds
    min_bound = np.int64(min_bound)
    max_bound = np.int64(max_bound)

    return min_bound, max_bound


def compute_tlu_output_shapes(
    tlu_subgraph: Graph,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Infer the reduced output shape of a tlu.

    The reduced output shape reduces dimensions which are broadcasted, to
    eliminate superfluous computations. A tlu
    can only apply univariate functions but broadcasting is allowed. Thus,
    the shapes of the constants used in the TLU, but also the values of the
    tlu output can be used to infer the minimal shape of the TLU input
    that can is broadcasted to the TLU output shape.

    E.g. Applying a univariate function to [[1, 1, 1], [2, 2, 2]] gives the the same
    result as applying a broadcasting function to [[1], [2]], but much faster.

    Args:
        tlu_subgraph (Graph): The executable subgraph of the tlu

    Returns:
        res: the reduced input shape, the original output shape without reduction
    """

    constant_shapes = []
    orig_constant_shapes = []
    for elt in tlu_subgraph.graph.nodes:
        assert isinstance(elt, Node)
        if isinstance(elt.evaluator, ConstantEvaluator):
            constant_shape = list(elt.output.shape)
            value = elt.evaluator.properties["constant"]
            if constant_shape:
                orig_constant_shapes.append(tuple(constant_shape))
                for axis in range(len(value.shape)):  # pylint: disable=consider-using-enumerate
                    unique_values_per_axis = np.unique(value, axis=axis)
                    if unique_values_per_axis.shape[axis] == 1:
                        constant_shape[axis] = 1

                constant_shapes.append(tuple(constant_shape))

    # This shape includes constant axes folding and only reduces broadcasted axes
    shape_ = merge_tlu_constant_shapes(constant_shapes)

    # This shape excludes constant axes
    orig_shape_ = merge_tlu_constant_shapes(orig_constant_shapes)

    return shape_, orig_shape_


def get_subgraph_input(subgraph: Graph) -> Node:
    """Find the encrypted input node of a subgraph.

    Args:
        subgraph: Executable tlu subgraph

    Returns:
        Node: encrypted input node

    Raises:
        ValueError: if the subgraph input could not be found
    """
    input_node = None
    for node in subgraph.graph:
        if "name" in node.properties and node.properties["name"] == "input":
            assert input_node is None, "More than one astype float node detected"
            input_node = node

    assert input_node is not None, f"Couldn't detect input node in:\n{subgraph.format()}"
    return input_node


def get_tlu_node_subgraph_input_node(graph: Graph, tlu_node: Node) -> Node:
    """Get the TLU subgraph node that contains the input to the TLU.

    Args:
        graph (Graph): The graph containing the TLU
        tlu_node (Node): The executable subgraph of the TLU

    Returns:
        res (Node): The node in the TLU subgraph that contains the TLU input
    """

    pred_nodes = graph.ordered_preds_of(tlu_node)
    variable_input_indices = []
    for pred_index, pred_node in enumerate(pred_nodes):
        if pred_node.operation != Operation.Constant:
            variable_input_indices.append(pred_index)

    assert len(variable_input_indices) == 1, "TLU node got more than one variable input"
    variable_input_index = variable_input_indices[0]
    variable_input_node: Node = pred_nodes[variable_input_index]
    assert isinstance(
        variable_input_node.output.dtype, Integer
    ), "TLU node got input dtype that isn't integer"
    return variable_input_node


def make_subgraph_input_tensor(
    min_bound: np.int64,
    max_bound: np.int64,
    orig_shape_: Tuple[int, ...],
    expected_shape: Tuple[int, ...],
) -> np.ndarray:
    """Build a tensor of a broadcastable shape that contains all possible inputs to a TLU.

    Args:
        min_bound (np.int64): minimum TLU input calibrated value
        max_bound (np.int64): maximum TLU input calibrated value
        orig_shape_ (Tuple[int, ...]): the non reduce shape of the TLU input
        expected_shape (Tuple[int, ...]): the TLU output shape

    Returns:
        res: Tensor with a shape that broadcasts during TLU execution
    """

    subgraph_inputs = np.array(list(range(int(min_bound), int(max_bound) + 1)))
    subgraph_input_shape = tuple([len(subgraph_inputs), *orig_shape_[1:]])

    reps = list(expected_shape)
    reps[0] = 1

    if len(expected_shape) > 1:
        to_tile = subgraph_inputs[
                tuple(
                    [
                        slice(0, len(subgraph_inputs), 1),
                        *[np.newaxis for _ in range(len(expected_shape[1:]))],
                    ]
                )
            ]
        subgraph_inputs = np.tile(
            to_tile,
            reps,
        )

#        subgraph_inputs = subgraph_inputs[tuple(slice(0, elt, 1) for elt in subgraph_input_shape)]

    return subgraph_inputs


def reduce_output_tensor(
    subgraph_inputs: np.ndarray, reference: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Reduce the TLU output tensor to eliminate redundant dimensions.

    Analyzes the contents of the tensor and determines if some
    dimensions have constant values and thus can be reduced to
    single values. Since TLUs can broadcast constants, it's
    faster to reduce dimensions in the input.

    Args:
        subgraph_inputs (np.ndarray): The TLU possible inputs
        reference (np.ndarray): The result of executing the original TLU on the inputs

    Returns:
        res: reduced subgraph inputs, reduced reference tensor
    """

    axes_to_test = reversed(range(1, reference.ndim))
    axis_to_start_reduce = reference.ndim
    for axis in axes_to_test:
        unique_per_axis = np.unique(reference, axis=axis)
        if unique_per_axis.shape[axis] == 1:
            axis_to_start_reduce = axis
        else:
            break

    if axis_to_start_reduce < reference.ndim:
        slices_reduce = []
        for axis in range(reference.ndim):
            if axis < axis_to_start_reduce:
                slices_reduce.append(slice(None))
            else:
                slices_reduce.append(slice(0, 1, 1))
        slices_reduce_tpl = tuple(slices_reduce)
        reference = reference[slices_reduce_tpl]
        subgraph_inputs = subgraph_inputs[slices_reduce_tpl]
    return subgraph_inputs, reference


# todo: fix to use proper function to add rounding
class InsertRounding(GraphProcessor):
    """
    InsertRounding graph processor, to add rounding before TLUs if desired.
    """

    rounding_threshold: Optional[int]

    def __init__(
        self,
        threshold: Optional[int],
        exactness: Exactness = Exactness.EXACT,
        overflow_protection: bool = True,
    ):
        self.rounding_threshold = threshold
        self.exactness = exactness
        self.overflow_protection = overflow_protection

    def apply(self, graph: Graph):
        if self.rounding_threshold is None:
            # No rounding
            return

        # Get all nodes that will be converted to LUTs
        tlu_nodes = graph.query_nodes(
            custom_filter=is_node_tlu,
        )
        for tlu_node in tlu_nodes:
            # Predecessor nodes
            pred_nodes = graph.ordered_preds_of(tlu_node)

            # Only take into accound predecessor's that aren't constants
            variable_input_indices = []
            for pred_index, pred_node in enumerate(pred_nodes):
                if pred_node.operation != Operation.Constant:
                    variable_input_indices.append(pred_index)

            # Only one input should be non-constant per LUT
            if len(variable_input_indices) != 1:
                continue

            # Get variable input
            variable_input_index = variable_input_indices[0]
            variable_input_node = pred_nodes[variable_input_index]
            variable_input_dtype = variable_input_node.output.dtype

            if not isinstance(variable_input_dtype, Integer):
                raise ValueError(f"{variable_input_dtype=} is not 'Integer'")

            variable_input_bit_width = variable_input_dtype.bit_width
            if variable_input_bit_width <= self.rounding_threshold:
                # No need to do anything if the bit-width is actually lower or equal
                # to the rounding threshold value
                continue

            # Compute lsbs to remove
            lsbs_to_remove = variable_input_bit_width - self.rounding_threshold

            # Rounding node
            rounding_node = Node.generic(
                "round_bit_pattern",
                [deepcopy(variable_input_node.output)],
                deepcopy(variable_input_node.output),
                round_bit_pattern,
                kwargs={
                    "lsbs_to_remove": lsbs_to_remove,
                    "overflow_protection": self.overflow_protection,
                    "exactness": self.exactness,
                },
                attributes={
                    "overflow_protection": self.overflow_protection,
                },
            )
            rounding_node.properties["final_lsbs_to_remove"] = lsbs_to_remove
            rounding_node.properties["resulting_bit_width"] = self.rounding_threshold
            rounding_node.properties["overflow_protection"] = self.overflow_protection
            rounding_node.properties["overflow_detected"] = False
            rounding_node.properties["exactness"] = self.exactness

            nx_graph = graph.graph
            nx_graph.add_edge(variable_input_node, rounding_node, input_idx=0)

            edge_data = nx_graph.get_edge_data(variable_input_node, tlu_node).values()
            for data in list(edge_data):
                input_idx = data["input_idx"]
                nx_graph.add_edge(rounding_node, tlu_node, input_idx=input_idx)

            nx_graph.remove_edge(variable_input_node, tlu_node)


class TLUDeltaBasedOptimizer(GraphProcessor):
    """TLU Step-based optimizer.

    Determines rounding bit-width based on TLU shape. Adds approximate
    rounding and scaling before/in TLUs if desired.

    A TLU is analyzed as follows:

    |                               +----
    |                               |
    |                    +----------+
    |                    |
    |             +------+
    |             |
    +------------------------------------
    ^---offset1---^
    ..............^delta1^
    ^--------offset2---^ ^---delta2-^

    Pairs of (offset, delta) are found by analyzing the output of the TLU.
    The inputs of the tlu are raised to a high precision (internal_bit_width_target)
    and the offset is removed. Next, rounding to a bit_width that is determined
    based on delta is applied. The effects of the scaling, offsetting and rounding
    are inversed in a modified TLU so the results obtained are the same as those
    of the original TLU.
    """

    def __init__(
        self,
        exactness: Exactness = Exactness.APPROXIMATE,
        overflow_protection: bool = True,
        internal_bit_width_target: int = 24,
        verbose: int = 0,
    ):
        self.exactness = exactness
        self.overflow_protection = overflow_protection
        self.rounding_function = round_bit_pattern
        self.internal_bit_width_target = internal_bit_width_target
        self.verbose = verbose
        # Store per PBS statistics to see how bit_widths were reduced
        self.statistics: Dict[Node, Dict[str, Union[int, np.ndarray, Tuple[int, ...]]]] = {}

    def apply(self, graph: Graph) -> None:
        """Apply the TLU optimization to a Graph for all TLUs.

        Args:
            graph (Graph): The executable graph containing TLU nodes
        """
        # Get all nodes in the graph that will be converted to a TLU
        tlu_nodes = graph.query_nodes(
            custom_filter=is_node_tlu,  # TLU filter function
            ordered=True,  # Not strictly necessary but easier to debug
        )

        for tlu_node in tlu_nodes:
            # On each tlu we do:
            # 1. Optimize a and b for the subgraph
            # 2. Insert a and b in the graph
            # 3. Insert rounding to the graph
            # 4. Insert a and b in the subgraph of the TLU
            #   We could also just modify the evalutor function to
            #   include the scale-up.
            #   That would require another way to do the optimization
            #   since we rely on the subgraph to reduce the dimension
            #   of the TLU.

            # Get TLU sub-graph
            evaluator = tlu_node.evaluator
            assert isinstance(evaluator, GenericEvaluator)

            if "subgraph" not in evaluator.properties["kwargs"]:
                if self.verbose:
                    print("skipping because tlu is not a subgraph")
                # Not supported for now
                # skipping because tlu is not a subgraph
                continue

            tlu_subgraph: Graph = evaluator.properties["kwargs"]["subgraph"]

            # TLU input node (multiple inputs using the same subgraph is not supported)
            variable_input_node = get_tlu_node_subgraph_input_node(graph, tlu_node)

            # Bounds are per-tensor for now.
            # Using per-element bounds would allow us to improve the optimization
            min_bound, max_bound = extract_tlu_input_bounds(variable_input_node)

            # # Create input with proper shape on the bounds for optimization
            # expected_shape = variable_input_node.output.shape

            # We reduce to 1 the dimension of the axis if no constant rely on it
            # This allows the computation to be done on less elements
            shape_, orig_shape_ = compute_tlu_output_shapes(
                tlu_subgraph,
            )

            # Create an input which the full input range
            subgraph_inputs = make_subgraph_input_tensor(
                min_bound,
                max_bound,
                orig_shape_,
                tuple([int(max_bound - min_bound) + 1, *shape_[1:]]),
            )
            # shape_ -> expected_shape

            # Compute TLU output on bounds without rounding or calibration for reference
            sorted_nodes = list(nx.topological_sort(tlu_subgraph.graph))
            reference = vectorized_graph_eval(
                tlu_subgraph, subgraph_inputs, sorted_nodes=sorted_nodes
            )
            assert isinstance(reference, np.ndarray)
            reference = reference.astype(np.int64)

            # If the broadcasting shape of the TLU could not be determined
            # try to find axes with constant values in the results of the tlu
            # so that its shape can be reduced to eliminate duplicate values
            if not shape_:
                subgraph_inputs, reference = reduce_output_tensor(subgraph_inputs, reference)

            best_a, best_b, lsbs_to_remove, msbs_to_keep = delta_optimize(
                subgraph_inputs,
                reference,
                shape_,
                (int(min_bound), int(max_bound)),
                self.internal_bit_width_target,
            )

            # For testing purposes we had some properties
            tlu_node.properties["attributes"]["opt_round_a"] = best_a
            tlu_node.properties["attributes"]["opt_round_b"] = best_b
            tlu_node.properties["attributes"]["round_bits"] = lsbs_to_remove

            all_multipliers_are_1 = np.all(best_a == 1)
            all_offsets_are_0 = np.all(best_b == 0)
            if (all_multipliers_are_1 and all_offsets_are_0) and lsbs_to_remove == 0:
                # Skipping because there is no benefit to upscaling
                continue

            # We need to make sure that we have the correct shape when adding constants in the graph
            # As Concrete Python doesn't handle broadcasting at the graph level
            assert isinstance(variable_input_node.output, ValueDescription)
            best_a = np.broadcast_to(best_a, shape=(1,) + variable_input_node.output.shape[1:])
            best_b = np.broadcast_to(best_b, shape=(1,) + variable_input_node.output.shape[1:])

            rounding_node = self.modify_graph_to_round_subgraph_inputs(
                graph, variable_input_node, best_a, best_b, lsbs_to_remove
            )

            self.statistics[tlu_node] = {
                "shape": variable_input_node.output.shape,
                "size": variable_input_node.output.size,
                "original_bit_width": int(variable_input_node.output.dtype.bit_width),
                "lsbs_to_remove": int(lsbs_to_remove),
                "msbs_to_keep": int(msbs_to_keep),
                "scaling_factor": best_a,
                "bias": best_b,
            }

            # Store some statistics for testing/debugging in the object itself
            tlu_node.properties["attributes"]["msbs_to_keep"] = int(
                variable_input_node.output.dtype.bit_width - lsbs_to_remove
            )
            tlu_node.properties["attributes"]["lsbs_to_remove"] = lsbs_to_remove

            tlu_node.inputs[0] = deepcopy(rounding_node.output)

            modify_subgraph_for_rounded_inputs(
                tlu_subgraph, best_a, best_b, rounding_node, x_min=min_bound, x_max=max_bound
            )

    def modify_graph_to_round_subgraph_inputs(
        self,
        graph: Graph,
        variable_input_node: Node,
        a: np.ndarray,
        b: np.ndarray,
        n_round: int,
    ):
        """Modify a graph to add rounding before a TLU.

        Remove the offset of TLU inputs and raise them to a higher precision. This
        is achieved by subtracting ``b`` and then multiplying with ``a``. Finally,
        the rounding to ``n_round`` bits is applied on the encrypted values that are
        input to the TLU. The transformation is cancelled out in the TLU by doing the
        inverse computation.

        Args:
            graph (Graph): The full graph with non-rounded TLUs
            variable_input_node (Node): The input to the TLU node that will be rounded
            a (np.ndarray): The scaling factor to raise precision
            b (np.ndarray): The offset to remove from the TLU input before raising precision
            n_round (np.int64): The number of bits to round off

        """

        if variable_input_node.output.shape:
            assert (
                a.shape == variable_input_node.output.shape
            ), f"{a.shape=} != {variable_input_node.output.shape=}"
            assert (
                b.shape == variable_input_node.output.shape
            ), f"{b.shape=} != {variable_input_node.output.shape=}"
        else:
            assert a.shape == (1,)
            assert b.shape == (1,)

            a = a[0]
            b = b[0]

        previous_node = variable_input_node

        # Multiply to match step size to rounding step size
        a_int = a.astype(np.int64)
        if np.any(a_int != 1):
            previous_node = add_leveled_op_with_cst(previous_node, a_int, multiply, graph.graph)

        # Subtract offset that matches step threshold to rounding step treshold
        b_int = b.astype(np.int64)
        if np.any(b_int != 0):
            previous_node = add_leveled_op_with_cst(previous_node, b_int, subtract, graph.graph)

        # # Compute the offset needed to cancel out the rounding offset
        # # Then broadcast it to match the rounded tensor shape
        # round_offset_scalar = np.int64(2 ** (n_round - 1))
        # round_offset = np.broadcast_to(round_offset_scalar, shape=b.shape)

        # Add offset for rounding correctness
        # previous_node = add_leveled_op_with_cst(previous_node, round_offset, subtract, graph.graph)

        # Round by n_round
        assert isinstance(previous_node.output.dtype, Integer)
        lsbs_to_remove = int(n_round)
        if lsbs_to_remove:
            previous_node = add_rounding_node(
                previous_node,
                lsbs_to_remove,
                graph.graph,
                rounding_function=self.rounding_function,
                exactness=self.exactness,
                overflow_protection=self.overflow_protection,
            )
        return previous_node
