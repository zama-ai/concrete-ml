"""Test file for virtual lib specific tests."""

# FIXME, Concrete Numpy 0.6 integration, #795
# pylint: disable=E1101

import numpy

from concrete.ml.virtual_lib import VirtualCircuit, VirtualCompiler


# FIXME, Concrete Numpy 0.6 integration, #795, restore this test
def te_st_torch_matmul_virtual_lib(default_configuration):
    """Test special cases of matmul compilation with virtual_lib"""

    def f(x, weights):
        return x @ weights

    thousand_ones = numpy.ones((1000,), dtype=numpy.int64)

    matmul_thousand_ones_compiler = VirtualCompiler(
        lambda x: f(x, thousand_ones), {"x": "encrypted"}, default_configuration
    )

    # Special inputset
    inputset = [thousand_ones]

    virtual_fhe_circuit = matmul_thousand_ones_compiler.compile(inputset)

    assert isinstance(virtual_fhe_circuit, VirtualCircuit)

    # 10 is >= log2(1000), so we expect the sum of 1000 ones to have bit width <= 10
    check_ok, max_bit_width, _ = virtual_fhe_circuit.check_circuit_uses_n_bits_or_less(10)
    other_max_bit_width = virtual_fhe_circuit.get_max_bit_width()
    assert check_ok
    assert max_bit_width == 10
    assert other_max_bit_width == max_bit_width

    # Test to check that ones cancelling out give the expected bit width as well
    two_thousand_ones_and_a_thousand_minus_ones = numpy.ones((3000,), dtype=numpy.int64)
    two_thousand_ones_and_a_thousand_minus_ones[-1000:] = -1

    matmul_three_thousand_plus_minus_ones_compiler = VirtualCompiler(
        lambda x: f(x, two_thousand_ones_and_a_thousand_minus_ones),
        {"x": "encrypted"},
        default_configuration,
    )

    # Special inputset
    inputset = [numpy.ones((3000,), dtype=numpy.int64)]
    virtual_fhe_circuit = matmul_three_thousand_plus_minus_ones_compiler.compile(inputset)

    assert isinstance(virtual_fhe_circuit, VirtualCircuit)

    # 10 is >= log2(1000), so we expect (2000 - 1000) == 1000 to have bit width <= 10
    check_ok, max_bit_width, _ = virtual_fhe_circuit.check_circuit_uses_n_bits_or_less(10)
    other_max_bit_width = virtual_fhe_circuit.get_max_bit_width()
    assert check_ok
    assert max_bit_width == 10
    assert other_max_bit_width == max_bit_width

    # Additional test with a simulated PBS with 10 bits
    def g(x, weights):
        return numpy.rint(numpy.sin(f(x, weights))).astype(numpy.int64)

    sin_matmul_three_thousand_plus_minus_ones_compiler = VirtualCompiler(
        lambda x: g(x, two_thousand_ones_and_a_thousand_minus_ones),
        {"x": "encrypted"},
        default_configuration,
    )

    # Special inputset
    inputset = [numpy.ones((3000,), dtype=numpy.int64)]
    virtual_fhe_circuit = sin_matmul_three_thousand_plus_minus_ones_compiler.compile(inputset)

    assert isinstance(virtual_fhe_circuit, VirtualCircuit)

    check_ok, max_bit_width, _ = virtual_fhe_circuit.check_circuit_uses_n_bits_or_less(10)
    other_max_bit_width = virtual_fhe_circuit.get_max_bit_width()
    assert check_ok
    assert max_bit_width == 10
    assert other_max_bit_width == max_bit_width
