"""Test utils functions."""
import numpy as np
import pytest

from concrete.ml.common.debugging.custom_assert import assert_true
from concrete.ml.common.utils import compute_bits_precision


@pytest.mark.parametrize(
    "x, expected_n_bits", [(range(2**8), 8), (range(2**8 + 1), 9), (range(8), 3)]
)
def test_compute_bits_precision(x, expected_n_bits):
    """Test the function that computes the number of bits necessary to represent an array."""
    assert_true(compute_bits_precision(np.array(x)) == expected_n_bits)
