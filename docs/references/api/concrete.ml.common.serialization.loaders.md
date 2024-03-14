<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/common/serialization/loaders.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.common.serialization.loaders`

Load functions for serialization.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/serialization/loaders.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `loads`

```python
loads(content: Union[str, bytes]) → Any
```

Load any Concrete ML object that provide a `dump_dict` method.

**Arguments:**

- <b>`content`</b> (Union\[str, bytes\]):  A serialized object.

**Returns:**

- <b>`Any`</b>:  The object itself.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/serialization/loaders.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load`

```python
load(file: Union[IO[str], IO[bytes]])
```

Load any Concrete ML object that provide a `load_dict` method.

**Arguments:**

- <b>`file`</b> (Union\[IO\[str\], IO\[bytes\]):  The file containing the serialized object.

**Returns:**

- <b>`Any`</b>:  The object itself.
