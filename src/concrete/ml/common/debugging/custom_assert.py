"""Provide some variants of assert."""
from typing import Type


def _custom_assert(
    condition: bool, on_error_msg: str = "", error_type: Type[Exception] = AssertionError
):
    """Provide a custom assert which is kept even if the optimized python mode is used.

    See https://docs.python.org/3/reference/simple_stmts.html#assert for the documentation
    on the classical assert function

    Args:
        condition(bool): the condition. If False, raise AssertionError
        on_error_msg(str): optional message for precising the error, in case of error
        error_type (Type[Exception]): the type of error to raise, if condition is not fulfilled.
            Default to AssertionError

    Raises:
        error_type: Raises an error if condition is False

    """

    if not condition:
        raise error_type(on_error_msg)


def assert_true(
    condition: bool, on_error_msg: str = "", error_type: Type[Exception] = AssertionError
):
    """Provide a custom assert to check that the condition is True.

    Args:
        condition(bool): the condition. If False, raise AssertionError
        on_error_msg(str): optional message for precising the error, in case of error
        error_type (Type[Exception]): the type of error to raise, if condition is not fulfilled.
            Default to AssertionError

    """
    _custom_assert(condition, on_error_msg, error_type)


def assert_false(
    condition: bool, on_error_msg: str = "", error_type: Type[Exception] = AssertionError
):
    """Provide a custom assert to check that the condition is False.

    Args:
        condition(bool): the condition. If True, raise AssertionError
        on_error_msg(str): optional message for precising the error, in case of error
        error_type (Type[Exception]): the type of error to raise, if condition is not fulfilled.
            Default to AssertionError

    """
    _custom_assert(not condition, on_error_msg, error_type)


def assert_not_reached(on_error_msg: str, error_type: Type[Exception] = AssertionError):
    """Provide a custom assert to check that a piece of code is never reached.

    Args:
        on_error_msg(str): message for precising the error
        error_type (Type[Exception]): the type of error to raise, if condition is not fulfilled.
            Default to AssertionError

    """
    _custom_assert(False, on_error_msg, error_type)
