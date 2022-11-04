<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/common/check_inputs.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.common.check_inputs`

Check and conversion tools.

Utils that are used to check (including convert) some data types which are compatible with scikit-learn to numpy types.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/common/check_inputs.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_array_and_assert`

```python
check_array_and_assert(X)
```

sklearn.utils.check_array with an assert.

Equivalent of sklearn.utils.check_array, with a final assert that the type is one which is supported by Concrete-ML.

**Args:**

- <b>`X`</b> (object):  Input object to check / convert

**Returns:**
The converted and validated array

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.5.x/src/concrete/ml/common/check_inputs.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `check_X_y_and_assert`

```python
check_X_y_and_assert(X, y, *args, **kwargs)
```

sklearn.utils.check_X_y with an assert.

Equivalent of sklearn.utils.check_X_y, with a final assert that the type is one which is supported by Concrete-ML.

**Args:**

- <b>`X`</b> (ndarray, list, sparse matrix):  Input data
- <b>`y`</b> (ndarray, list, sparse matrix):  Labels
- <b>`*args`</b>:  The arguments to pass to check_X_y
- <b>`**kwargs`</b>:  The keyword arguments to pass to check_X_y

**Returns:**
The converted and validated arrays
