<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/common/serialization/decoder.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.common.serialization.decoder`

Custom decoder for serialization.

## **Global Variables**

- **ALL_QUANTIZED_OPS**
- **SUPPORTED_TORCH_ACTIVATIONS**
- **USE_SKOPS**
- **TRUSTED_SKOPS**
- **SERIALIZABLE_CLASSES**

______________________________________________________________________

<a href="../../../src/concrete/ml/common/serialization/decoder.py#L103"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `object_hook`

```python
object_hook(d: Any) → Any
```

Define a custom object hook that enables loading any supported serialized values.

If the input's type is non-native, then we expect it to have the following format.More information is available in the ConcreteEncoder class.

**Args:**

- <b>`d`</b> (Any):  The serialized value to load.

**Returns:**

- <b>`Any`</b>:  The loaded value.

**Raises:**

- <b>`NotImplementedError`</b>:  If the serialized object does not provides a `dump_dict` method as  expected.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/serialization/decoder.py#L224"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ConcreteDecoder`

Custom json decoder to handle non-native types found in serialized Concrete ML objects.

<a href="../../../src/concrete/ml/common/serialization/decoder.py#L227"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*args, **kwargs)
```
