<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/common/serialization/dumpers.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.common.serialization.dumpers`

Dump functions for serialization.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/serialization/dumpers.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `dumps_random_state`

```python
dumps_random_state(random_state: Optional[RandomState, int]) → str
```

Dump random state to string.

**Arguments:**

- <b>`random_state`</b> (Union\[RandomState, int, None\]):  a random state

**Returns:**

- <b>`str`</b>:  a serialized version of the random state

______________________________________________________________________

<a href="../../../src/concrete/ml/common/serialization/dumpers.py#L28"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `dump`

```python
dump(obj: Any, file: <class 'TextIO'>)
```

Dump any Concrete ML object that has a dump method.

**Arguments:**

- <b>`obj`</b> (Any):  the object to dump.
- <b>`file`</b> (TextIO):  a file containing the serialized object.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/serialization/dumpers.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `dumps`

```python
dumps(obj: Any) → str
```

Dump as string any object.

If the object has some `dumps` method then it uses that. Otherwise the object is casted as `str`.

**Arguments:**

- <b>`obj`</b> (Any):  any object.

**Returns:**

- <b>`str`</b>:  a string representation of the object.
