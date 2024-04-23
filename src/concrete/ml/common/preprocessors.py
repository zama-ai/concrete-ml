"""Graph pre-processors for automatic rounding."""

from copy import deepcopy
from itertools import product
from typing import Any, Callable, Dict, List, Tuple, Union

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

    return tuple(
        max(constant_shape[idx] for constant_shape in constant_shapes if len(constant_shape) > idx)
        for idx in range(max(len(elt) for elt in constant_shapes))
    )


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
        return a_node

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
    rounding_node = Node.generic(
        name=rounding_function.__name__,
        inputs=[deepcopy(a_node.output)],
        output=deepcopy(a_node.output),
        operation=rounding_function,
        kwargs=rounding_kwargs,
        attributes=attributes,
    )
    rounding_node.properties["final_lsbs_to_remove"] = lsbs_to_remove
    rounding_node.properties["resulting_bit_width"] = a_node.output.dtype.bit_width - lsbs_to_remove
    rounding_node.properties["overflow_detected"] = False
    rounding_node.properties["original_rounded_bit_width"] = a_node.output.dtype.bit_width
    if rounding_function.__name__ == "round_bit_pattern":
        rounding_node.properties["overflow_protection"] = overflow_protection
        rounding_node.properties["exactness"] = exactness

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
    assert isinstance(b, (float, int)) or (
        isinstance(b, np.ndarray) and b.dtype in (np.float64, np.int64)
    ), f"Constant {b} should be of dtype np.int64 or np.float64, not {b.dtype}"

    assert isinstance(a_node.output.dtype, (Float, Integer))

    constant_node = Node.constant(b)

    # Handle dtype
    if b.dtype == np.float64:
        constant_dtype = Float(64)
        result_dtype = Float(64)
        bounds = None
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
        bounds = (
            results.min(),
            results.max(),
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
    new_node.bounds = bounds

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


def compute_best_rounding_for_single_tlu(
    x_min: int,
    x_max: int,
    steps_indexes: np.ndarray,
    per_tensor_a: np.int64,
    raised_bitwidth: int,
    tlu_evaluation_function: Callable,
) -> Tuple[int, int, int]:
    """Finds the optimal rounding based on unidimensional outputs of a TLU.

    Args:
        x_min (int): minimal calibrated value of TLU inputs
        x_max (int): maximal calibrated value of TLU inputs
        steps_indexes (np.ndarray): the indices of the TLU output changes
        per_tensor_a (np.int64): the scaling factor that raises precision to the desired one
        raised_bitwidth (int): the desired raised precision
        tlu_evaluation_function (Callable): the function that computes the error incurred
            when rounding this TLU with a certain scale and offset

    Returns:
        res: tuple containing the scale, offset and rounding bits
    """

    if len(steps_indexes) == 0:
        # The function is constant so nothing to do here
        # We can just the raised precision value down to one bit
        return (1, 0, raised_bitwidth - 1)

    assert len(set(steps_indexes)) == len(steps_indexes), "Steps indexes are not unique"

    step_thresholds = steps_indexes[0:-1]  # all step thresholds

    delta_axis = np.diff(steps_indexes, axis=0)  # all step sizes

    assert step_thresholds.size == delta_axis.size

    if len(delta_axis) == 0:
        # Single jump, we can just offset by the threshold and round to 1-bit
        return (1, steps_indexes[0], raised_bitwidth - 1)

    low_side_step_size = steps_indexes[0] - x_min
    high_side_step_size = x_max - steps_indexes[-1]
    min_side_step_size = min(low_side_step_size, high_side_step_size)
    delta_axis = np.minimum(delta_axis, min_side_step_size)

    if np.all(delta_axis <= 1):
        # Do not round
        return (1, 0, 0)

    all_combinations = np.stack([step_thresholds, delta_axis]).T
    all_a_b_r = np.zeros((all_combinations.shape[0], 3), np.int64)
    for idx, comb in enumerate(all_combinations):
        threshold, delta = comb[0], comb[1]

        b = threshold

        r_bits = int(np.ceil(np.log2(per_tensor_a * delta)))

        all_a_b_r[idx, 0] = per_tensor_a
        all_a_b_r[idx, 1] = b
        all_a_b_r[idx, 2] = r_bits

    unique_a_b_r = np.unique(all_a_b_r, axis=0)

    all_comb_mse = np.zeros((unique_a_b_r.shape[0],))
    for idx, comb in enumerate(unique_a_b_r):
        a, b, r_bits = comb[0], comb[1], comb[2]
        mse = tlu_evaluation_function(a, b, r_bits)
        all_comb_mse[idx] = mse

    best_a_b_r = np.argmin(all_comb_mse)

    return (unique_a_b_r[best_a_b_r, 0], unique_a_b_r[best_a_b_r, 1], unique_a_b_r[best_a_b_r, 2])


def delta_optimize(
    subgraph_inputs: np.ndarray,
    reference: np.ndarray,
    shape_: Tuple[int, ...],
    bounds: Tuple[int, int],
    tlu_subgraph: Graph,
    raised_bitwidth: int,
    rounding_function: Callable = round_bit_pattern,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Optimize a TLU by analyzing the steps (deltas) in the output of the TLU function.

    Args:
        subgraph_inputs (np.ndarray): The possible inputs of the
            TLU subgraph, determined using calibration
        reference (np.ndarray): The TLU outputs on the possible inputs of the
            original TLU without rounding
        shape_ (Tuple[int, ...]): the shape of constants that broadcast to the inputs of the TLU
        bounds (Tuple[int, int]): calibrated min/max of the possible inputs
        tlu_subgraph (nx.DiGraph): the subgraph that contains the TLU operations
        raised_bitwidth (int): the desired raised bitwidth used for approximate rounding
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
    bitwidth_input = Integer.that_can_represent([x_min, x_max]).bit_width

    # Initialize a and b such that no changes are done
    best_a = np.ones(shape_, dtype=np.int64)
    best_b = np.zeros(shape_, dtype=np.int64)

    ref_diff = np.diff(reference, axis=0).astype(bool)

    # Compute mask of values for which there is a change
    change_mask = np.concatenate(
        [
            np.zeros(reference[:1].shape).astype(bool),
            ref_diff.astype(bool),
        ]
    ).astype(bool)

    # Some accumulators
    n_rounds = np.ones(shape_, dtype=np.int64)

    per_tensor_a = 2 ** (raised_bitwidth - bitwidth_input)

    for indexes in product(*[range(elt) for elt in shape_]):
        selection = tuple([slice(0, reference.shape[0]), *indexes[1:]])
        best_indexes = tuple([*indexes])

        subgraph_inputs_selected = subgraph_inputs[selection]
        change_mask_selected = change_mask[selection]
        steps_indexes = subgraph_inputs_selected[change_mask_selected]
        assert len(steps_indexes) == len(np.unique(steps_indexes)), "not unique steps"

        def evaluate_tlu(a, b, r_bits):
            rounded_output = eval_subgraph_on_rounded_inputs(
                tlu_subgraph, subgraph_inputs, rounding_function, a, b, r_bits
            )
            return np.sqrt(np.sum((rounded_output - reference) ** 2))

        (
            best_a[best_indexes],
            best_b[best_indexes],
            n_rounds[indexes],
        ) = compute_best_rounding_for_single_tlu(
            x_min, x_max, steps_indexes, per_tensor_a, raised_bitwidth, evaluate_tlu
        )

    # As rounding can be applied only for the entire tensor,
    # a single rounding threshold must be found. The smallest
    # number of lsbs to remove is used. The scaling factor
    # is chosen for that threshold, while the step
    # offsets are preserved
    if n_rounds.shape:
        n_round_idx = np.argmin(n_rounds)
        n_round = n_rounds.flatten()[n_round_idx]
        best_a[::] = best_a.flatten()[n_round_idx]
    else:
        n_round = n_rounds
    n_round = int(n_round)
    return best_a, best_b, n_round


def modify_subgraph_for_rounded_inputs(tlu_subgraph: Graph, a: np.ndarray, c: np.ndarray):
    """Modify a TLU subgraph to cancel out the rounding.

    A TLU will cancel out the rounding transformation by dividing
    its inputs by ``a`` and subtracting the offset ``c``.

    Args:
        tlu_subgraph (Graph): the subgraph that contains the TLU operations
        a (np.ndarray): broadcasted scaling factors
        c (np.ndarray): the offset of the rounding approximation
    """

    previous_node = get_subgraph_input(tlu_subgraph)

    # Use floats in the subgraph
    a = a.astype(np.float64)
    c = c.astype(np.float64)

    # Divide by scaling factor
    previous_node = add_leveled_op_with_cst(previous_node, a, divide, graph=tlu_subgraph.graph)

    # Add threshold offset
    previous_node = add_leveled_op_with_cst(previous_node, c, add, graph=tlu_subgraph.graph)


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

    rounded_inputs = rounding_function(
        (subgraph_inputs - b) * a - 2 ** (lsbs_to_remove - 1),
        lsbs_to_remove=int(lsbs_to_remove),
    ).astype(np.float64)

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
        tlu_subgraph_copy, rounded_inputs, sorted_nodes=sorted_nodes, input_indices=indices
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


def compute_tlu_output_shapes(tlu_subgraph: Graph) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
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

    # TODO: assert that there isn't any rounding nodes before

    variable_input_index = variable_input_indices[0]
    variable_input_node: Node = pred_nodes[variable_input_index]
    assert isinstance(
        variable_input_node.output.dtype, Integer
    ), "TLU node got input dtype that isn't integer"
    return variable_input_node


# TODO: the issue is HERE or in the usage of this function
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

    print(f"{subgraph_input_shape=}")

    if len(expected_shape) > 1:
        subgraph_inputs = np.tile(
            subgraph_inputs[
                tuple(
                    [
                        slice(0, len(subgraph_inputs), 1),
                        *[np.newaxis for _ in range(len(expected_shape[1:]))],
                    ]
                )
            ],
            expected_shape,
        )

        subgraph_inputs = subgraph_inputs[tuple(slice(0, elt, 1) for elt in subgraph_input_shape)]

    print(f"{subgraph_inputs.shape=}")
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
    and the offset is removed. Next, rounding to a bitwidth that is determined
    based on delta is applied. The effects of the scaling, offsetting and rounding
    are inversed in a modified TLU so the results obtained are the same as those
    of the original TLU.
    """

    def __init__(
        self,
        exactness: Exactness = Exactness.APPROXIMATE,
        overflow_protection: bool = True,
        internal_bit_width_target: int = 24,
    ):
        self.exactness = exactness
        self.overflow_protection = overflow_protection
        self.rounding_function = round_bit_pattern
        self.internal_bit_width_target = internal_bit_width_target
        # Store per PBS statistics to see how bitwidths were reduced
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
                # Not supported for now
                continue
            tlu_subgraph: Graph = evaluator.properties["kwargs"]["subgraph"]

            # TLU input node (multiple inputs using the same subgraph is not supported)
            variable_input_node = get_tlu_node_subgraph_input_node(graph, tlu_node)

            # Bounds are per-tensor for now.
            # Using per-element bounds would allow us to improve the optimization
            min_bound, max_bound = extract_tlu_input_bounds(variable_input_node)

            # Create input with proper shape on the bounds for optimization
            expected_shape = variable_input_node.output.shape

            # We reduce to 1 the dimension of the axis if no constant rely on it
            # This allows the computation to be done on less elements
            shape_, orig_shape_ = compute_tlu_output_shapes(
                tlu_subgraph,
            )

            # Create an input which the full input range
            subgraph_inputs = make_subgraph_input_tensor(
                min_bound, max_bound, orig_shape_, tuple([max_bound - min_bound, *shape_[1:]]),
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

            best_a, best_b, n_round = delta_optimize(
                subgraph_inputs,
                reference,
                shape_,
                (int(min_bound), int(max_bound)),
                tlu_subgraph,
                self.internal_bit_width_target,
            )

            # For testing purposes we had some properties
            tlu_node.properties["attributes"]["opt_round_a"] = best_a
            tlu_node.properties["attributes"]["opt_round_b"] = best_b
            tlu_node.properties["attributes"]["round_bits"] = n_round

            all_multipliers_are_1 = np.all(best_a == 1)
            all_offsets_are_0 = np.all(best_b == 0)
            if (all_multipliers_are_1 and all_offsets_are_0) or n_round == 0:
                continue

            # We need to make sure that we have the correct shape when adding constants in the graph
            # As Concrete Python doesn't handle broadcasting at the graph level
            assert isinstance(variable_input_node.output, ValueDescription)
            best_a = np.broadcast_to(best_a, shape=(1,) + variable_input_node.output.shape[1:])
            best_b = np.broadcast_to(best_b, shape=(1,) + variable_input_node.output.shape[1:])

            self.modify_graph_to_round_subgraph_inputs(
                graph, variable_input_node, best_a, best_b, n_round
            )

            self.statistics[tlu_node] = {
                "shape": variable_input_node.output.shape,
                "size": variable_input_node.output.size,
                "original_bitwidth": int(variable_input_node.output.dtype.bit_width),
                "optimized_bitwidth": int(self.internal_bit_width_target - n_round),
            }

            # Store some statistics for testing/debugging in the object itself
            tlu_node.properties["attributes"]["msbs_to_keep"] = int(
                variable_input_node.output.dtype.bit_width - n_round
            )
            tlu_node.properties["attributes"]["lsbs_to_remove"] = n_round

            modify_subgraph_for_rounded_inputs(tlu_subgraph, best_a, best_b)

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
        assert a.shape == variable_input_node.output.shape
        assert b.shape == variable_input_node.output.shape

        # Subtract offset that matches step threshold to rounding step treshold
        previous_node = add_leveled_op_with_cst(
            variable_input_node, b.astype(np.int64), subtract, graph.graph
        )

        # Multiply to match step size to rounding step size
        previous_node = add_leveled_op_with_cst(
            previous_node, a.astype(np.int64), multiply, graph.graph
        )

        # Compute the offset needed to cancel out the rounding offset
        # Then broadcast it to match the rounded tensor shape
        round_offset_scalar = np.int64(2 ** (n_round - 1))
        round_offset = np.broadcast_to(round_offset_scalar, shape=b.shape)

        # Add offset for rounding correctness
        previous_node = add_leveled_op_with_cst(previous_node, round_offset, subtract, graph.graph)

        # Round by n_round
        assert isinstance(previous_node.output.dtype, Integer)
        lsbs_to_remove = int(n_round)
        previous_node = add_rounding_node(
            previous_node,
            lsbs_to_remove,
            graph.graph,
            rounding_function=self.rounding_function,
            exactness=self.exactness,
            overflow_protection=self.overflow_protection,
        )
