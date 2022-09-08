"""Tests for our inputs checks"""

import numpy
import pytest

from concrete.ml.common.check_inputs import check_array_and_assert, check_X_y_and_assert

x_with_nans = [numpy.array([[numpy.nan], [0.0]]), numpy.array([[numpy.nan], [numpy.nan]])]
x_clean = [numpy.array([[1.0], [2.0]]), numpy.array([[0], [10]])]
x_1_element = [numpy.array([[2.0]])]
y_1_element = [numpy.array([2.0])]
y_with_nans = [numpy.array([numpy.nan, 0.0]), numpy.array([numpy.nan, numpy.nan])]
y_clean = [numpy.array([1.0, 2.0]), numpy.array([0, 10])]


@pytest.mark.parametrize("x", x_with_nans)
def test_check_array_and_assert_nan(x):
    """Checks that with an input containing a Nan the function raises an error"""
    with pytest.raises(ValueError, match=r"Input .*contains NaN.*"):
        check_array_and_assert(x)


@pytest.mark.parametrize("x", x_clean)
def test_check_array_and_assert_clean(x):
    """Checks that with a correct input no change is done to it"""
    checked_x = check_array_and_assert(x)
    assert numpy.array_equal(checked_x, x)


@pytest.mark.parametrize("x", x_with_nans)
@pytest.mark.parametrize("y", y_clean)
def test_check_x_y_and_assert_nan_clean(x, y):
    """Checks that if X contains a Nan the function raises an error"""
    with pytest.raises(ValueError, match=r"Input .*contains NaN.*"):
        check_X_y_and_assert(x, y)


@pytest.mark.parametrize("x", x_clean)
@pytest.mark.parametrize("y", y_with_nans)
def test_check_x_y_and_assert_clean_nan(x, y):
    """Checks that if y contains a Nan the function raises an error"""
    with pytest.raises(ValueError, match=r"Input .*contains NaN.*"):
        check_X_y_and_assert(x, y)


@pytest.mark.parametrize("x", x_with_nans)
@pytest.mark.parametrize("y", y_with_nans)
def test_check_x_y_and_assert_nan_nan(x, y):
    """Checks that if X and y contain Nan values the function raises an error"""
    with pytest.raises(ValueError, match=r"Input .*contains NaN.*"):
        check_X_y_and_assert(x, y)


@pytest.mark.parametrize("x", x_clean)
@pytest.mark.parametrize("y", y_clean)
def test_check_x_y_and_assert_clean_clean(x, y):
    """Checks that with a correct input no change is done to it"""
    checked_x, checked_y = check_X_y_and_assert(x, y)
    assert numpy.array_equal(checked_x, x)
    assert numpy.array_equal(checked_y, y)


@pytest.mark.parametrize("x", x_1_element)
@pytest.mark.parametrize("y", y_clean)
def test_check_x_y_and_assert_1elt_clean(x, y):
    """Checks that if X and y don't have the same length the function raises an error"""
    with pytest.raises(ValueError):
        check_X_y_and_assert(x, y)


@pytest.mark.parametrize("x", x_clean)
@pytest.mark.parametrize("y", y_1_element)
def test_check_x_y_and_assert_clean_1elt(x, y):
    """Checks that if X and y don't have the same length the function raises an error"""
    with pytest.raises(ValueError):
        check_X_y_and_assert(x, y)
