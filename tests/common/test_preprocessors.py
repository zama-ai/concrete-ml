"""Unit tests for TLU optimization preprocessors."""

from typing import List

import numpy
import pytest
from concrete.fhe import Compiler, Configuration, Exactness, univariate

from concrete.ml.common.preprocessors import CycleDetector, GraphProcessor, TLUDeltaBasedOptimizer


def make_step_function(n_thresholds, delta, x_min, x_max, power_of_two=False):
    """Make a step function using a TLU."""
    thresholds_ = []  # First threshold

    if power_of_two:
        th0 = numpy.random.randint(0, delta)
    else:
        th0 = numpy.random.randint(x_min, x_max)

    for index in range(n_thresholds):
        thresholds_.append(th0 + index * delta)

    thresholds = tuple(thresholds_)

    # Step size function to optimize
    def util(x):
        return sum([numpy.where(x >= float(threshold), 1.0, 0.0) for threshold in thresholds])

    def step_function(x):
        return univariate(util)(x).astype(numpy.int64)

    def f(x):
        return step_function(x.astype(numpy.float64))

    def constant_f(x):
        return univariate(lambda x: th0 * (1.0 - (x.astype(numpy.float64) * 0.0)))(x).astype(
            numpy.int64
        )

    if n_thresholds == 0:
        return constant_f
    return f


def make_identity_function():
    """Make an identity function using TLUs."""

    def f(x):
        return (x * 1.0).astype(numpy.int64)

    return f


def make_random_function(x_min, x_max):
    """Make a function that maps to a TLU that that looks random."""

    freq = numpy.random.random(size=(3,))

    def f(x):
        x = x.astype(numpy.float64)
        y = numpy.sin(x * freq[0]) + numpy.sin(x * freq[1]) + numpy.sin(x * freq[2])
        y = y * (x_max - x_min) + x_max
        y = y.astype(numpy.int64)
        return y

    return f


def make_tlu_optimizer_function(execution_number: int, function_name: str):
    """Make a function to be tested by the TLU optimizer."""

    # Create function
    if function_name == "staircase_pot":
        n_bits_from = execution_number + 2
        x_min, x_max = -(2 ** (n_bits_from)), (2 ** (n_bits_from)) - 1
        delta = 2 ** (n_bits_from // 2)  # Constant step size assumption
        f = make_step_function(execution_number, delta, x_min, x_max, True)

    elif function_name == "staircase":
        n_bits_from = numpy.random.choice(range(3, 9))
        # 2**n range
        x_min, x_max = -(2**n_bits_from), (2**n_bits_from) - 1

        # Real sub-range
        x_min = numpy.random.randint(x_min, x_max // 2)
        x_max = numpy.random.randint(x_max, x_max + 1)

        # Function thresholds
        delta = 2 ** (n_bits_from // 2) - 1  # Constant step size assumption

        f = make_step_function(execution_number, delta, x_min, x_max)
    elif function_name == "identity":
        n_bits_from = execution_number + 1
        x_min, x_max = -(2**n_bits_from), (2**n_bits_from) - 1

        f = make_identity_function()
    elif function_name == "random":
        n_bits_from = execution_number + 1
        x_min, x_max = -(2**n_bits_from), (2**n_bits_from) - 1
        f = make_random_function(x_min, x_max)
    else:
        raise AssertionError(f"Invalid function to test for TLU optimization {function_name}")

    return f, x_min, x_max


@pytest.mark.parametrize("execution_number", range(1, 7))
@pytest.mark.parametrize("function_name", ["staircase_pot", "staircase", "identity", "random"])
def test_tlu_optimizer(execution_number: int, function_name: str):
    """Tests the tlu optimizer with various functions."""

    curr_seed = numpy.random.randint(0, 2**32)
    numpy.random.seed(curr_seed + execution_number)

    f, x_min, x_max = make_tlu_optimizer_function(execution_number, function_name)

    # Function definition bounds
    input_set = numpy.arange(x_min, x_max + 1, 1, dtype=numpy.int64)
    input_set_as_list_of_array = [numpy.array([elt]) for elt in input_set]

    # Optim, Rounding
    tlu_optimizer = TLUDeltaBasedOptimizer(
        verbose=False, exactness=Exactness.EXACT, overflow_protection=False
    )
    cycle_detector = CycleDetector()
    additional_pre_processors: List[GraphProcessor] = [tlu_optimizer]
    additional_post_processors: List[GraphProcessor] = [cycle_detector]
    compilation_configuration = Configuration(
        additional_pre_processors=additional_pre_processors,
        additional_post_processors=additional_post_processors,
    )
    compiler = Compiler(
        f,
        parameter_encryption_statuses={"x": "encrypted"},
    )

    circuit = compiler.compile(
        input_set_as_list_of_array,
        configuration=compilation_configuration,
    )

    # Find optimized TLUs
    tlu_nodes = circuit.graph.query_nodes(
        custom_filter=lambda node: node.converted_to_table_lookup
        and "opt_round_a" in node.properties["attributes"],
        ordered=True,
    )

    # A single TLU should be present
    assert len(tlu_nodes) == 1, f"No TLU found in mlir or \n {circuit.graph.format()}"
    tlu_node = tlu_nodes[0]
    lsbs_to_remove = tlu_node.properties["attributes"].get("lsbs_to_remove", 0)

    # No-optim, no-rounding
    compiler = Compiler(
        f,
        parameter_encryption_statuses={"x": "encrypted"},
    )
    circuit_no_optim_no_rounding = compiler.compile(
        input_set_as_list_of_array,
    )

    reference = f(input_set)

    if function_name in ["staircase_pot", "staircase"] and execution_number == 0:
        # TODO: round to 1b or eliminate these TLUs entirely
        assert (
            circuit.mlir == circuit_no_optim_no_rounding.mlir
        ), "Constant TLUs should not have optimization"
        return

    if function_name == "identity":
        assert circuit.mlir == circuit_no_optim_no_rounding.mlir
        return

    simulated = numpy.array([circuit.simulate(numpy.array([elt])) for elt in input_set])[..., 0]
    simulated_no_optim_no_rounding = numpy.array(
        [circuit_no_optim_no_rounding.simulate(numpy.array([elt])) for elt in input_set]
    )[..., 0]

    graph_res = numpy.array([circuit.graph(numpy.array([elt])) for elt in input_set])[..., 0]
    graph_res_no_optim_no_rounding = numpy.array(
        [circuit_no_optim_no_rounding.graph(numpy.array([elt])) for elt in input_set]
    )[..., 0]

    not_equal = reference != simulated
    if not_equal.sum() > 0 and function_name == "staircase_pot":
        raise Exception(
            f"TLU Optimizer is not exact: "
            f"{not_equal.mean()=} = {not_equal.sum()}/{not_equal.size}\n"
            f"{tlu_optimizer.statistics=}\n"
            f"{execution_number=}, {lsbs_to_remove=} \n{'#'*20}\n"
            f"{(simulated == graph_res).mean()=}, "
            f"{(simulated_no_optim_no_rounding == graph_res_no_optim_no_rounding).mean()=}\n"
            f"{circuit.graph.format()}\n{'#'*20}\n"
            f"{circuit_no_optim_no_rounding.graph.format()}\n"
        )
