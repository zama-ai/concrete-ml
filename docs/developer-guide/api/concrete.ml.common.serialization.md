<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/common/serialization.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.common.serialization`

Serialization.

## **Global Variables**

- **LOADS_METHODS**

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/common/serialization.py#L65"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `dumps_onnx`

```python
dumps_onnx(onnx_model: ModelProto) → str
```

Dump onnx model as string.

**Arguments:**

- <b>`onnx_model`</b> (onnx.ModelProto):  an onnx model.

**Returns:**

- <b>`str`</b>:  a serialized version of the onnx model.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/common/serialization.py#L77"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/common/serialization.py#L90"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/common/serialization.py#L107"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/common/serialization.py#L131"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `loads`

```python
loads(content: str) → Any
```

Load any CML object that has a dump method.

**Arguments:**

- <b>`content`</b> (str):  a serialized object.

**Returns:**

- <b>`Any`</b>:  the object itself.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/common/serialization.py#L144"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load_dict`

```python
load_dict(metadata: Dict[str, Any]) → Any
```

Load any CML object that has a dump method.

**Arguments:**

- <b>`metadata`</b> (Dict\[str, Any\]):  a dict of a serialized object.

**Returns:**

- <b>`Any`</b>:  the object itself.

**Raises:**

- <b>`ValueError`</b>:  if "cml_dumped_class_name" key is not in the serialized object.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/common/serialization.py#L182"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `load`

```python
load(file: <class 'TextIO'>)
```

Load any CML object that has a dump method.

**Arguments:**

- <b>`file`</b> (TextIO):  a file containing the serialized object.

**Returns:**

- <b>`Any`</b>:  the object itself.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/common/serialization.py#L195"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `dump`

```python
dump(obj: Any, file: <class 'TextIO'>)
```

Dump any CML object that has a dump method.

**Arguments:**

- <b>`obj`</b> (Any):  the object to dump.
- <b>`file`</b> (TextIO):  a file containing the serialized object.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/common/serialization.py#L205"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/common/serialization.py#L15"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CustomEncoder`

CustomEncoder: custom json encoder to handle non-native types.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/common/serialization.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `default`

```python
default(o: Any) → Any
```

Overload default serialization.

**Arguments:**

- <b>`o`</b> (Any):  the object to serialize.

**Returns:**
The serialized object.

**Raises:**

- <b>`NotImplementedError`</b>:  if a cnp.Circuit is given.
