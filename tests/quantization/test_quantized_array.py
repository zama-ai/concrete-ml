"""Tests for the quantized array/tensors."""
import numpy
import pytest

from concrete.ml.quantization import QuantizedArray


@pytest.mark.parametrize(
    "n_bits",
    [32, 28, 20, 16, 8, 4],
)
@pytest.mark.parametrize(
    "is_signed, is_symmetric",
    [pytest.param(True, True), pytest.param(True, False), pytest.param(False, False)],
)
@pytest.mark.parametrize("values", [pytest.param(numpy.random.randn(2000))])
def test_quant_dequant_update(values, n_bits, is_signed, is_symmetric, check_array_equality):
    """Test the quant and dequant function."""

    quant_array = QuantizedArray(n_bits, values, is_signed=is_signed, is_symmetric=is_symmetric)
    qvalues = quant_array.quant()

    # Quantized values must be contained between 0 and 2**n_bits
    assert numpy.max(qvalues) <= 2 ** (n_bits) - 1 - quant_array.quantizer.offset
    assert numpy.min(qvalues) >= -quant_array.quantizer.offset

    # Dequantized values must be close to original values
    dequant_values = quant_array.dequant()

    # Check that all values are close
    tolerance = quant_array.quantizer.scale / 2
    assert numpy.isclose(dequant_values, values, atol=tolerance).all()

    # Explain the choice of tolerance
    # This test checks the values are quantized and dequantized correctly
    # Each quantization have a maximum error per quantized value an it's `scale / 2`

    # To give an intuition, let's say you have the scale of 0.5
    #     the range `[a + 0.00, a + 0.25]` will be quantized into 0, dequantized into `a + 0.00`
    #     the range `[a + 0.25, a + 0.75]` will be quantized into 1, dequantized into `a + 0.50`
    #     the range `[a + 0.75, a + 1.25]` will be quantized into 2, dequantized into `a + 1.00`
    #     ...

    # So for each quantization-then-dequantization operation,
    # the maximum error is `0.25`, which is `scale / 2`

    # Test update functions
    new_values = numpy.array([0.3, 0.5, -1.2, -3.4])
    new_qvalues_ = quant_array.update_values(new_values)

    # Make sure the shape changed for the qvalues
    assert new_qvalues_.shape != qvalues.shape

    new_qvalues = numpy.array([1, 4, 7, 29])
    new_values_updated = quant_array.update_quantized_values(new_qvalues)

    # Make sure that we can see at least one change.
    assert not numpy.array_equal(new_qvalues, new_qvalues_)
    assert not numpy.array_equal(new_values, new_values_updated)

    # Check that the __call__ returns also the qvalues.
    check_array_equality(quant_array(), new_qvalues)


@pytest.mark.parametrize(
    "n_bits",
    [32, 28, 20, 16, 8, 4],
)
@pytest.mark.parametrize("is_signed", [pytest.param(True), pytest.param(False)])
@pytest.mark.parametrize("value_shape", [(10,), (1, 3, 224, 224)])
def test_quantized_array_all_zeros(n_bits, is_signed, value_shape):
    """Test case where all values are close to 0 in an interval smaller than STABILITY_CONST."""

    values = numpy.random.uniform(0, QuantizedArray.STABILITY_CONST * 0.99, size=value_shape)

    quant_array = QuantizedArray(
        n_bits,
        values,
        is_signed=is_signed,
    )

    assert quant_array.quantizer.scale == 1
    assert quant_array.quantizer.zero_point == 0

    qvalues = quant_array.quant()

    # Quantized values must be contained between 0 and 2**n_bits
    assert numpy.max(qvalues) <= 2 ** (n_bits) - 1 - quant_array.quantizer.offset
    assert numpy.min(qvalues) >= -quant_array.quantizer.offset


def test_quantized_array_constructor():
    """Test various QuantizedArray construction semantics."""

    value_shape = (10,)
    values = numpy.random.uniform(0, 1, size=value_shape)

    # Create an array with precomputed statistics
    qarr = QuantizedArray(2, values, stats=None, rmax=2, rmin=-1, uvalues=[0, 1, 2])

    # Verify that the statistics were not recomputed
    assert qarr.quantizer.rmax == 2
    assert qarr.quantizer.rmin == -1

    # Create an array with an invalid quantizer parameter
    with pytest.raises(TypeError):
        QuantizedArray(2, values, stats=None, __InvalidParam=2)

    # Test an incomplete stats structure, should throw an error
    with pytest.raises(TypeError):
        QuantizedArray(2, values, stats=None, rmax=2)
