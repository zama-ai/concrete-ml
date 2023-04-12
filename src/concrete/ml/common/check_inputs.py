"""Check and conversion tools.

Utils that are used to check (including convert) some data types which are compatible with
scikit-learn to numpy types.
"""

import numpy
import sklearn

from ..common.debugging.custom_assert import assert_true

# Disable pylint invalid name since scikit learn uses "X" as variable name for data
# pylint: disable=invalid-name


def check_array_and_assert(X, *args, **kwargs):
    """sklearn.utils.check_array with an assert.

    Equivalent of sklearn.utils.check_array, with a final assert that the type is one which
    is supported by Concrete ML.

    Args:
        X (object): Input object to check / convert
        *args: The arguments to pass to check_array
        **kwargs: The keyword arguments to pass to check_array

    Returns:
        The converted and validated array
    """
    X = sklearn.utils.check_array(X, *args, **kwargs)
    assert_true(isinstance(X, numpy.ndarray), f"wrong type {type(X)}")
    return X


def check_X_y_and_assert(X, y, *args, **kwargs):
    """sklearn.utils.check_X_y with an assert.

    Equivalent of sklearn.utils.check_X_y, with a final assert that the type is one which
    is supported by Concrete ML.

    Args:
        X (ndarray, list, sparse matrix): Input data
        y (ndarray, list, sparse matrix): Labels
        *args: The arguments to pass to check_X_y
        **kwargs: The keyword arguments to pass to check_X_y

    Returns:
        The converted and validated arrays
    """

    X, y = sklearn.utils.check_X_y(X, y, *args, **kwargs)
    assert_true(isinstance(X, numpy.ndarray), f"wrong type {type(X)}")
    assert_true(isinstance(y, numpy.ndarray), f"wrong type {type(y)}")
    return X, y


def check_X_y_and_assert_multi_output(X, y, *args, **kwargs):
    """sklearn.utils.check_X_y with an assert and multi-output handling.

    Equivalent of sklearn.utils.check_X_y, with a final assert that the type is one which
    is supported by Concrete ML. If y is 2D, allows multi-output.

    Args:
        X (ndarray, list, sparse matrix): Input data
        y (ndarray, list, sparse matrix): Labels
        *args: The arguments to pass to check_X_y
        **kwargs: The keyword arguments to pass to check_X_y

    Returns:
        The converted and validated arrays with multi-output targets.
    """
    multi_output = isinstance(y[0], list) if isinstance(y, list) else len(y.shape) > 1
    X, y = check_X_y_and_assert(X, y, *args, multi_output=multi_output, **kwargs)
    return X, y


# pylint: enable=invalid-name
