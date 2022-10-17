<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/onnx/onnx_utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.onnx.onnx_utils`

Utils to interpret an ONNX model with numpy.

## **Global Variables**

- **ATTR_TYPES**
- **ATTR_GETTERS**
- **ONNX_OPS_TO_NUMPY_IMPL**
- **ONNX_COMPARISON_OPS_TO_NUMPY_IMPL_FLOAT**
- **ONNX_COMPARISON_OPS_TO_NUMPY_IMPL_BOOL**
- **ONNX_OPS_TO_NUMPY_IMPL_BOOL**
- **IMPLEMENTED_ONNX_OPS**

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/onnx/onnx_utils.py#L207"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_attribute`

```python
get_attribute(attribute: AttributeProto) → Any
```

Get the attribute from an ONNX AttributeProto.

**Args:**

- <b>`attribute`</b> (onnx.AttributeProto):  The attribute to retrieve the value from.

**Returns:**

- <b>`Any`</b>:  The stored attribute value.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/onnx/onnx_utils.py#L219"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_op_name`

```python
get_op_name(node)
```

Construct the qualified name of the ONNX operator.

**Args:**

- <b>`node`</b> (Any):  ONNX graph node

**Returns:**

- <b>`result`</b> (str):  qualified name

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/onnx/onnx_utils.py#L231"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `execute_onnx_with_numpy`

```python
execute_onnx_with_numpy(graph: GraphProto, *inputs: ndarray) → Tuple[ndarray, ]
```

Execute the provided ONNX graph on the given inputs.

**Args:**

- <b>`graph`</b> (onnx.GraphProto):  The ONNX graph to execute.
- <b>`*inputs`</b>:  The inputs of the graph.

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  The result of the graph's execution.
