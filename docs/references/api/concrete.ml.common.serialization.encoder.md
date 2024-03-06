<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/common/serialization/encoder.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.common.serialization.encoder`

Custom encoder for serialization.

## **Global Variables**

- **INFINITY**
- **USE_SKOPS**

______________________________________________________________________

<a href="../../../src/concrete/ml/common/serialization/encoder.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `dump_name_and_value`

```python
dump_name_and_value(name: str, value: Any, **kwargs) → Dict
```

Dump the value into a custom dict format.

**Args:**

- <b>`name`</b> (str):  The custom name to use. This name should be unique for each type to encode, as  it is used in the ConcreteDecoder class to detect the initial type and apply the proper  load method to the serialized object.
- <b>`value`</b> (Any):  The serialized value to dump.
- <b>`**kwargs (dict)`</b>:  Additional arguments to dump.

**Returns:**

- <b>`Dict`</b>:  The serialized custom format that includes both the serialized value and its type  name.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/serialization/encoder.py#L57"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ConcreteEncoder`

Custom json encoder to handle non-native types found in serialized Concrete ML objects.

Non-native types are serialized manually and dumped in a custom dict format that stores both the serialization value of the object and its associated type name.

The name should be unique for each type, as it is used in the ConcreteDecoder class to detect the initial type and apply the proper load method to the serialized object. The serialized value is the value that was serialized manually in a native type. Additional arguments such as a numpy array's dtype are also properly serialized. If an object has an unexpected type or is not serializable, an error is thrown.

The ConcreteEncoder is only meant to encode Concrete-ML's built-in models and therefore only supports the necessary types. For example, torch.Tensor objects are not serializable using this encoder as built-in models only use numpy arrays. However, the list of supported types might expand in future releases if new models are added and need new types.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/serialization/encoder.py#L176"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `default`

```python
default(o: Any) → Any
```

Define a custom default method that enables dumping any supported serialized values.

**Arguments:**

- <b>`o`</b> (Any):  The object to serialize.

**Returns:**

- <b>`Any`</b>:  The serialized object. Non-native types are returned as a dict of a specific  format.

**Raises:**

- <b>`NotImplementedError`</b>:  If an FHE.Circuit, a Callable or a Generator object is given.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/serialization/encoder.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `isinstance`

```python
isinstance(o: Any, cls: Type) → bool
```

Define a custom isinstance method.

Natively, among other types, the JSONENcoder handles integers, floating points and tuples. However, a numpy.integer (resp. numpy.floating) object is automatically casted to a built-in int (resp. float) object, without keeping their dtype information. Similarly, a tuple is casted to a list, meaning that it will then be loaded as a list, which notably does not have the uniqueness property and therefore might cause issues in complex structures such as QuantizedModule instances. This is an issue as JSONEncoder only calls its customizable `default` method at the end of the parsing. We thus need to provide this custom isinstance method in order to make the encoder avoid handling these specific types until `default` is reached (where they are properly serialized using our custom format).

**Args:**

- <b>`o`</b> (Any):  The object to serialize.
- <b>`cls`</b> (Type):  The type to compare the object with.

**Returns:**

- <b>`bool`</b>:  If the object is of the given type. False if it is a numpy.floating, numpy.integer  or a tuple.

______________________________________________________________________

<a href="../../../src/concrete/ml/common/serialization/encoder.py#L110"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `iterencode`

```python
iterencode(o: Any, _one_shot: bool = False) → Generator
```

Encode the given object and yield each string representation as available.

This method overrides the JSONEncoder's native iterencode one in order to pass our custom isinstance method to the `_make_iterencode` function. More information in `isinstance`'s docstring. For simplicity, iterencode does not give the ability to use the initial `c_make_encoder` function, as it would required to override it in C.

**Args:**

- <b>`o`</b> (Any):  The object to serialize.
- <b>`_one_shot`</b> (bool):  This parameter is not used since the `_make_iterencode` function has  been removed from the method.

**Returns:**

- <b>`Generator`</b>:  Yield each string representation as available.
