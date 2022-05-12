"""Test file for virtual lib specific tests."""
import numpy
from concrete.numpy.compilation.circuit import Circuit
from concrete.numpy.compilation.compiler import Compiler


def test_torch_matmul_virtual_lib(default_configuration):
    """Test special cases of matmul compilation with virtual_lib"""

    def f(x, weights):
        return x @ weights

    thousand_ones = numpy.ones((1000,), dtype=numpy.int64)

    matmul_thousand_ones_compiler = Compiler(lambda x: f(x, thousand_ones), {"x": "encrypted"})

    # Special inputset
    inputset = [thousand_ones]
    virtual_fhe_circuit = matmul_thousand_ones_compiler.compile(
        inputset, default_configuration, virtual=True
    )

    assert isinstance(virtual_fhe_circuit, Circuit)

    # 10 is >= log2(1000), so we expect the sum of 1000 ones to have bit width <= 10
    max_bit_width = virtual_fhe_circuit.graph.maximum_integer_bit_width()
    assert max_bit_width == 10

    # Test to check that ones cancelling out give the expected bit width as well
    two_thousand_ones_and_a_thousand_minus_ones = numpy.ones((3000,), dtype=numpy.int64)
    two_thousand_ones_and_a_thousand_minus_ones[-1000:] = -1

    matmul_three_thousand_plus_minus_ones_compiler = Compiler(
        lambda x: f(x, two_thousand_ones_and_a_thousand_minus_ones),
        {"x": "encrypted"},
    )

    # Special inputset
    inputset = [numpy.ones((3000,), dtype=numpy.int64)]
    virtual_fhe_circuit = matmul_three_thousand_plus_minus_ones_compiler.compile(
        inputset, default_configuration, virtual=True
    )

    assert isinstance(virtual_fhe_circuit, Circuit)

    # 10 is >= log2(1000), so we expect (2000 - 1000) == 1000 to have bit width <= 10
    max_bit_width = virtual_fhe_circuit.graph.maximum_integer_bit_width()
    assert max_bit_width == 10

    # Additional test with a simulated PBS with 10 bits
    def g(x, weights):
        return numpy.rint(numpy.sin(f(x, weights))).astype(numpy.int64)

    sin_matmul_three_thousand_plus_minus_ones_compiler = Compiler(
        lambda x: g(x, two_thousand_ones_and_a_thousand_minus_ones),
        {"x": "encrypted"},
    )

    # Special inputset
    inputset = [numpy.ones((3000,), dtype=numpy.int64)]
    virtual_fhe_circuit = sin_matmul_three_thousand_plus_minus_ones_compiler.compile(
        inputset, default_configuration, virtual=True
    )

    assert isinstance(virtual_fhe_circuit, Circuit)

    max_bit_width = virtual_fhe_circuit.graph.maximum_integer_bit_width()
    assert max_bit_width == 10
