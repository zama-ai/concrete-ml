<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/common/debugging/custom_assert.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.common.debugging.custom_assert`

Provide some variants of assert.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/debugging/custom_assert.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `assert_true`

```python
assert_true(
    condition: bool,
    on_error_msg: str = '',
    error_type: Type[Exception] = <class 'AssertionError'>
)
```

Provide a custom assert to check that the condition is True.

**Args:**

- <b>`condition`</b> (bool):  the condition. If False, raise AssertionError
- <b>`on_error_msg`</b> (str):  optional message for precising the error, in case of error
- <b>`error_type`</b> (Type\[Exception\]):  the type of error to raise, if condition is not fulfilled.  Default to AssertionError

______________________________________________________________________

<a href="../../../src/concrete/ml/common/debugging/custom_assert.py#L43"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `assert_false`

```python
assert_false(
    condition: bool,
    on_error_msg: str = '',
    error_type: Type[Exception] = <class 'AssertionError'>
)
```

Provide a custom assert to check that the condition is False.

**Args:**

- <b>`condition`</b> (bool):  the condition. If True, raise AssertionError
- <b>`on_error_msg`</b> (str):  optional message for precising the error, in case of error
- <b>`error_type`</b> (Type\[Exception\]):  the type of error to raise, if condition is not fulfilled.  Default to AssertionError

______________________________________________________________________

<a href="../../../src/concrete/ml/common/debugging/custom_assert.py#L58"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `assert_not_reached`

```python
assert_not_reached(
    on_error_msg: str,
    error_type: Type[Exception] = <class 'AssertionError'>
)
```

Provide a custom assert to check that a piece of code is never reached.

**Args:**

- <b>`on_error_msg`</b> (str):  message for precising the error
- <b>`error_type`</b> (Type\[Exception\]):  the type of error to raise, if condition is not fulfilled.  Default to AssertionError
