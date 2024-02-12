<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/onnx/onnx_utils.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.onnx.onnx_utils`

Utils to interpret an ONNX model with numpy.

## **Global Variables**

- **ATTR_TYPES**
- **ATTR_GETTERS**
- **ONNX_OPS_TO_NUMPY_IMPL**
- **ONNX_COMPARISON_OPS_TO_NUMPY_IMPL_FLOAT**
- **ONNX_COMPARISON_OPS_TO_NUMPY_IMPL_BOOL**
- **ONNX_COMPARISON_OPS_TO_ROUNDED_TREES_NUMPY_IMPL_BOOL**
- **ONNX_OPS_TO_NUMPY_IMPL_BOOL**
- **IMPLEMENTED_ONNX_OPS**

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/onnx_utils.py#L433"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/onnx/onnx_utils.py#L445"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_op_type`

```python
get_op_type(node)
```

Construct the qualified type name of the ONNX operator.

**Args:**

- <b>`node`</b> (Any):  ONNX graph node

**Returns:**

- <b>`result`</b> (str):  qualified name

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/onnx_utils.py#L457"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/onnx_utils.py#L486"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `execute_onnx_with_numpy_trees`

```python
execute_onnx_with_numpy_trees(
    graph: GraphProto,
    lsbs_to_remove_for_trees: Optional[Tuple[int, int]],
    *inputs: ndarray
) → Tuple[ndarray, ]
```

Execute the provided ONNX graph on the given inputs for tree-based models only.

**Args:**

- <b>`graph`</b> (onnx.GraphProto):  The ONNX graph to execute.
- <b>`lsbs_to_remove_for_trees`</b> (Optional\[Tuple\[int, int\]\]):  This parameter is exclusively used for  optimizing tree-based models. It contains the values of the least significant bits to  remove during the tree traversal, where the first value refers to the first comparison  (either "less" or "less_or_equal"), while the second value refers to the "Equal"  comparison operation.  Default to None.
- <b>`*inputs`</b>:  The inputs of the graph.

**Returns:**

- <b>`Tuple[numpy.ndarray]`</b>:  The result of the graph's execution.

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/onnx_utils.py#L545"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `remove_initializer_from_input`

```python
remove_initializer_from_input(model: ModelProto)
```

Remove initializers from model inputs.

In some cases, ONNX initializers may appear, erroneously, as graph inputs. This function searches all model inputs and removes those that are initializers.

**Args:**

- <b>`model`</b> (onnx.ModelProto):  the model to clean

**Returns:**

- <b>`onnx.ModelProto`</b>:  the cleaned model
