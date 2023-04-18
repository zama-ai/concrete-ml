<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/common/serialization/encoder.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.common.serialization.encoder`

Custom encoder for serialization.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/serialization/encoder.py#L66"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/common/serialization/encoder.py#L16"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CustomEncoder`

CustomEncoder: custom json encoder to handle non-native types.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/serialization/encoder.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

- <b>`NotImplementedError`</b>:  if a fhe.Circuit is given.
