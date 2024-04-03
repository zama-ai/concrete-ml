<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/pandas/__init__.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.pandas`

Public API for encrypted data-frames.

## **Global Variables**

- **dataframe**
- **client_engine**

______________________________________________________________________

<a href="../../../src/concrete/ml/pandas/__init__.py#L9"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_encrypted_dataframe`

```python
load_encrypted_dataframe(path: Union[Path, str]) → EncryptedDataFrame
```

Load a serialized encrypted data-frame.

**Args:**

- <b>`path`</b> (Union\[Path, str\]):  The path to consider for loading the serialized encrypted  data-frame.

**Returns:**

- <b>`EncryptedDataFrame`</b>:  The loaded encrypted data-frame.

______________________________________________________________________

<a href="../../../src/concrete/ml/pandas/__init__.py#L23"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `merge`

```python
merge(
    left_encrypted: EncryptedDataFrame,
    right_encrypted: EncryptedDataFrame,
    how: str = 'left',
    on: Optional[str] = None,
    left_on: Optional[Hashable, Sequence[Hashable]] = None,
    right_on: Optional[Hashable, Sequence[Hashable]] = None,
    left_index: bool = False,
    right_index: bool = False,
    sort: bool = False,
    suffixes: Tuple[Optional[str], Optional[str]] = ('_x', '_y'),
    copy: Optional[bool] = None,
    indicator: Union[bool, str] = False,
    validate: Optional[str] = None
) → EncryptedDataFrame
```

Merge two encrypted data-frames in FHE using Pandas parameters.

Note that for now, only a left and right join is implemented. Additionally, only some Pandas parameters are supported, and joining on multiple columns is not available.

Pandas documentation for version 2.0 can be found here: https://pandas.pydata.org/pandas-docs/version/2.0/reference/api/pandas.DataFrame.merge.html

**Args:**

- <b>`left_encrypted`</b> (EncryptedDataFrame):  The left encrypted data-frame.
- <b>`right_encrypted`</b> (EncryptedDataFrame):  The right encrypted data-frame.
- <b>`how`</b> (str):  Type of merge to be performed, one of {'left', 'right'}.
- <b>`* left`</b>:  use only keys from left frame, similar to a SQL left outer join; preserve key order.
- <b>`* right`</b>:  use only keys from right frame, similar to a SQL right outer join; preserve key order.
- <b>`on`</b> (Optional\[str\]):  Column name to join on. These must be found in both DataFrames. If  it is None then this defaults to the intersection of the columns in both DataFrames.  Default to None.
- <b>`left_on`</b> (Optional\[Union\[Hashable, Sequence\[Hashable\]\]\]):  Currently not supported, please  keep the default value. Default to None.
- <b>`right_on`</b> (Optional\[Union\[Hashable, Sequence\[Hashable\]\]\]):  Currently not supported,  please keep the default value. Default to None.
- <b>`left_index`</b> (bool):  Currently not supported, please keep the default value. Default to  False.
- <b>`right_index`</b> (bool):  Currently not supported, please keep the default value. Default  to False.
- <b>`sort`</b> (bool):  Currently not supported, please keep the default value. Default to False.
- <b>`suffixes`</b> (Tuple\[Optional\[str\], Optional\[str\]\]):  A length-2 sequence where each element  is optionally a string indicating the suffix to add to overlapping column names in  `left` and `right` respectively. Pass a value of `None` instead of a string to  indicate that the column name from `left` or `right` should be left as-is, with no  suffix. At least one of the values must not be None.. Default to ("\_x", "\_y").
- <b>`copy`</b> (Optional\[bool\]):  Currently not supported, please keep the default value. Default  to None.
- <b>`indicator`</b> (Union\[bool, str\]):  Currently not supported, please keep the default value.  Default to False.
- <b>`validate`</b> (Optional\[str\]):  Currently not supported, please keep the default value.  Default to None.

**Returns:**

- <b>`EncryptedDataFrame`</b>:  The joined encrypted data-frame.
