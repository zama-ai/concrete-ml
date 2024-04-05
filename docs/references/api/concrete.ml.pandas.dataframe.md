<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/pandas/dataframe.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.pandas.dataframe`

Define the encrypted data-frame framework.

## **Global Variables**

- **ZIP_STORED**

______________________________________________________________________

<a href="../../../src/concrete/ml/pandas/dataframe.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `EncryptedDataFrame`

Define an encrypted data-frame framework that supports Pandas operators and parameters.

<a href="../../../src/concrete/ml/pandas/dataframe.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    encrypted_values: ndarray,
    encrypted_nan: Value,
    evaluation_keys: EvaluationKeys,
    column_names: List[str],
    dtype_mappings: Dict,
    api_version: int
)
```

______________________________________________________________________

#### <kbd>property</kbd> api_version

Get the API version used when instantiating this instance.

**Returns:**

- <b>`int`</b>:  The data-frame's API version.

______________________________________________________________________

#### <kbd>property</kbd> column_names

Get the data-frame's column names in order.

**Returns:**

- <b>`List[str]`</b>:  The data-frame's column names in order.

______________________________________________________________________

#### <kbd>property</kbd> column_names_to_position

Get the mapping between each column's name and its index position.

**Returns:**

- <b>`Dict[str, int]`</b>:  Mapping between column names and their position.

______________________________________________________________________

#### <kbd>property</kbd> dtype_mappings

Get the mappings for non-integer dtypes used in pre and post-processing.

**Returns:**

- <b>`Dict`</b>:  The mappings for non-integers dtypes.

______________________________________________________________________

#### <kbd>property</kbd> encrypted_nan

Get the encrypted value representing a NaN.

**Returns:**

- <b>`fhe.Value`</b>:  The encrypted representation of a NaN.

______________________________________________________________________

#### <kbd>property</kbd> encrypted_values

Get the encrypted values.

**Returns:**

- <b>`numpy.ndarray`</b>:  The array containing all encrypted values.

______________________________________________________________________

#### <kbd>property</kbd> evaluation_keys

Get the evaluation keys.

**Returns:**

- <b>`fhe.EvaluationKeys`</b>:  The evaluation keys.

______________________________________________________________________

<a href="../../../src/concrete/ml/pandas/dataframe.py#L133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_schema`

```python
get_schema() → DataFrame
```

Get the encrypted data-frame's scheme.

The scheme can include column names, dtypes or dtype mappings. It is displayed as a Pandas data-frame for better readability.

**Returns:**

- <b>`pandas.DataFrame`</b>:  The encrypted data-frame's scheme.

______________________________________________________________________

<a href="../../../src/concrete/ml/pandas/dataframe.py#L334"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load`

```python
load(path: Union[Path, str])
```

Load an encrypted data-frame from disk.

**Args:**

- <b>`path`</b> (Union\[Path, str\]):  The path where to load the encrypted data-frame.

**Returns:**

- <b>`EncryptedDataFrame`</b>:  The loaded encrypted data-frame.

______________________________________________________________________

<a href="../../../src/concrete/ml/pandas/dataframe.py#L171"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `merge`

```python
merge(
    other,
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
)
```

Merge two encrypted data-frames in FHE using Pandas parameters.

Note that for now, only a left and right join is implemented. Additionally, only some Pandas parameters are supported, and joining on multiple columns is not available.

Pandas documentation for version 2.0 can be found here: https://pandas.pydata.org/pandas-docs/version/2.0/reference/api/pandas.DataFrame.merge.html

**Args:**

- <b>`other`</b> (EncryptedDataFrame):  The other encrypted data-frame.
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

______________________________________________________________________

<a href="../../../src/concrete/ml/pandas/dataframe.py#L315"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save`

```python
save(path: Union[Path, str])
```

Save the encrypted data-frame on disk.

**Args:**

- <b>`path`</b> (Union\[Path, str\]):  The path where to save the encrypted data-frame.
