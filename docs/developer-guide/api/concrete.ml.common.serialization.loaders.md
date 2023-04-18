<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/common/serialization/loaders.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.common.serialization.loaders`

Load functions for serialization.

## **Global Variables**

- **LOADS_METHODS**

______________________________________________________________________

<a href="../../../src/concrete/ml/common/serialization/loaders.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_dict`

```python
load_dict(metadata: Dict[str, Any]) → Any
```

Load any Concrete ML object that has a dump method.

**Arguments:**

- <b>`metadata`</b> (Dict\[str, Any\]):  a dict of a serialized object.

**Returns:**

- <b>`Any`</b>:  the object itself.

**Raises:**

- <b>`ValueError`</b>:  if "cml_dumped_class_name" key is not in the serialized object.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/serialization/loaders.py#L47"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `loads`

```python
loads(content: str) → Any
```

Load any Concrete ML object that has a dump method.

**Arguments:**

- <b>`content`</b> (str):  a serialized object.

**Returns:**

- <b>`Any`</b>:  the object itself.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/serialization/loaders.py#L60"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load`

```python
load(file: <class 'TextIO'>)
```

Load any Concrete ML object that has a dump method.

**Arguments:**

- <b>`file`</b> (TextIO):  a file containing the serialized object.

**Returns:**

- <b>`Any`</b>:  the object itself.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/serialization/loaders.py#L73"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `loads_onnx`

```python
loads_onnx(serialized_onnx: str) → ModelProto
```

Load serialized onnx model.

**Arguments:**

- <b>`serialized_onnx`</b> (str):  a serialized onnx model.

**Returns:**

- <b>`onnx.ModelProto`</b>:  the onnx model

______________________________________________________________________

<a href="../../../src/concrete/ml/common/serialization/loaders.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `loads_random_state`

```python
loads_random_state(
    serialized_random_state: str
) → Union[RandomState, int, NoneType]
```

Load random state from string.

**Arguments:**

- <b>`serialized_random_state`</b> (str):  a serialized version of the random state

**Returns:**

- <b>`random_state`</b> (Union\[RandomState, int, None\]):  a random state
