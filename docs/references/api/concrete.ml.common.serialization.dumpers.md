<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/common/serialization/dumpers.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.common.serialization.dumpers`

Dump functions for serialization.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/serialization/dumpers.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `dumps`

```python
dumps(obj: Any) → str
```

Dump any object as a string.

**Arguments:**

- <b>`obj`</b> (Any):  Object to dump.

**Returns:**

- <b>`str`</b>:  A string representation of the object.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/serialization/dumpers.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `dump`

```python
dump(obj: Any, file: <class 'TextIO'>)
```

Dump any Concrete ML object in a file.

**Arguments:**

- <b>`obj`</b> (Any):  The object to dump.
- <b>`file`</b> (TextIO):  The file to dump the serialized object into.
