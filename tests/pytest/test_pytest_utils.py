"""Test pytest utility functions."""

import numpy
import pytest
from numpy.random import RandomState

from concrete.ml.pytest.utils import values_are_equal


@pytest.mark.parametrize(
    "value_1, value_2, expected_output",
    [
        pytest.param(None, None, True),
        pytest.param(None, 0, False),
        pytest.param(numpy.array([0]), numpy.array([0]), True),
        pytest.param(
            numpy.array([0], dtype=numpy.int64), numpy.array([0], dtype=numpy.int32), False
        ),
        pytest.param(numpy.array([0]), 0, False),
        pytest.param(numpy.int64(0), numpy.int64(0), True),
        pytest.param(numpy.int64(0), numpy.int32(0), False),
        pytest.param(numpy.int64(0), 0, False),
        pytest.param(numpy.float64(0), numpy.float64(0), True),
        pytest.param(numpy.float64(0), numpy.float32(0), False),
        pytest.param(numpy.float64(0), 0, False),
        pytest.param(RandomState(0), RandomState(0), True),
        pytest.param(RandomState(0), RandomState(1), False),
        pytest.param(RandomState(0), 0, False),
        pytest.param(0, 0, True),
        pytest.param(0, 1, False),
    ],
)
def test_values_are_equal(value_1, value_2, expected_output):
    """Check that values_are_equal works properly."""

    assert values_are_equal(value_1, value_2) == expected_output, (
        f"The values_are_equal utility function does not match the expected output: {value_1} and "
        f"{value_2} were expected to be "
        + "equal " * expected_output
        + "different " * (1 - expected_output)
    )
