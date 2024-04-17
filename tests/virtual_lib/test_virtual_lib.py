"""Test file for FHE simulation specific tests."""

import numpy
from concrete.fhe.compilation.circuit import Circuit
from concrete.fhe.compilation.compiler import Compiler


def test_torch_matmul_fhe_simulation(default_configuration):
    """Test special cases of matmul compilation with FHE simulation"""

    def f(x, weights):
        return x @ weights

    thousand_ones = numpy.ones((1000,), dtype=numpy.int64)

    matmul_thousand_ones_compiler = Compiler(lambda x: f(x, thousand_ones), {"x": "encrypted"})

    # Special input-set
    inputset = [thousand_ones]
    fhe_simulation_circuit = matmul_thousand_ones_compiler.compile(
        inputset,
        default_configuration,
    )

    assert isinstance(fhe_simulation_circuit, Circuit)

    # 10 is >= log2(1000), so we expect the sum of 1000 ones to have bit width <= 10
    max_bit_width = fhe_simulation_circuit.graph.maximum_integer_bit_width()
    assert max_bit_width == 10

    # Test to check that canceling out ones give the expected bit width as well
    two_thousand_ones_and_a_thousand_minus_ones = numpy.ones((3000,), dtype=numpy.int64)
    two_thousand_ones_and_a_thousand_minus_ones[-1000:] = -1

    matmul_three_thousand_plus_minus_ones_compiler = Compiler(
        lambda x: f(x, two_thousand_ones_and_a_thousand_minus_ones),
        {"x": "encrypted"},
    )

    # Special input-set
    inputset = [numpy.ones((3000,), dtype=numpy.int64)]
    fhe_simulation_circuit = matmul_three_thousand_plus_minus_ones_compiler.compile(
        inputset,
        default_configuration,
    )

    assert isinstance(fhe_simulation_circuit, Circuit)

    # 10 is >= log2(1000), so we expect (2000 - 1000) == 1000 to have bit width <= 10
    max_bit_width = fhe_simulation_circuit.graph.maximum_integer_bit_width()
    assert max_bit_width == 10

    # Additional test with a simulated PBS with 10 bits
    def g(x, weights):
        return numpy.rint(numpy.sin(f(x, weights))).astype(numpy.int64)

    sin_matmul_three_thousand_plus_minus_ones_compiler = Compiler(
        lambda x: g(x, two_thousand_ones_and_a_thousand_minus_ones),
        {"x": "encrypted"},
    )

    # Special input-set
    inputset = [numpy.ones((3000,), dtype=numpy.int64)]
    fhe_simulation_circuit = sin_matmul_three_thousand_plus_minus_ones_compiler.compile(
        inputset,
        default_configuration,
    )

    assert isinstance(fhe_simulation_circuit, Circuit)

    max_bit_width = fhe_simulation_circuit.graph.maximum_integer_bit_width()
    assert max_bit_width == 10
