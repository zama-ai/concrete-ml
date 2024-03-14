<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/onnx/onnx_model_manipulations.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.onnx.onnx_model_manipulations`

Some code to manipulate models.

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/onnx_model_manipulations.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `simplify_onnx_model`

```python
simplify_onnx_model(onnx_model: ModelProto)
```

Simplify an ONNX model, removes unused Constant nodes and Identity nodes.

**Args:**

- <b>`onnx_model`</b> (onnx.ModelProto):  the model to simplify.

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/onnx_model_manipulations.py#L21"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `remove_unused_constant_nodes`

```python
remove_unused_constant_nodes(onnx_model: ModelProto)
```

Remove unused Constant nodes in the provided onnx model.

**Args:**

- <b>`onnx_model`</b> (onnx.ModelProto):  the model for which we want to remove unused Constant nodes.

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/onnx_model_manipulations.py#L53"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `remove_identity_nodes`

```python
remove_identity_nodes(onnx_model: ModelProto)
```

Remove identity nodes from a model.

**Args:**

- <b>`onnx_model`</b> (onnx.ModelProto):  the model for which we want to remove Identity nodes.

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/onnx_model_manipulations.py#L78"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `keep_following_outputs_discard_others`

```python
keep_following_outputs_discard_others(
    onnx_model: ModelProto,
    outputs_to_keep: Iterable[str]
)
```

Keep the outputs given in outputs_to_keep and remove the others from the model.

**Args:**

- <b>`onnx_model`</b> (onnx.ModelProto):  the ONNX model to modify.
- <b>`outputs_to_keep`</b> (Iterable\[str\]):  the outputs to keep by name.

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/onnx_model_manipulations.py#L109"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `remove_node_types`

```python
remove_node_types(onnx_model: ModelProto, op_types_to_remove: List[str])
```

Remove unnecessary nodes from the ONNX graph.

**Args:**

- <b>`onnx_model`</b> (onnx.ModelProto):  The ONNX model to modify.
- <b>`op_types_to_remove`</b> (List\[str\]):  The node types to remove from the graph.

**Raises:**

- <b>`ValueError`</b>:  Wrong replacement by an Identity node.

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/onnx_model_manipulations.py#L163"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `clean_graph_at_node_op_type`

```python
clean_graph_at_node_op_type(
    onnx_model: ModelProto,
    node_op_type: str,
    fail_if_not_found: bool = True
)
```

Remove the first node matching node_op_type and its following nodes from the ONNX graph.

**Args:**

- <b>`onnx_model`</b> (onnx.ModelProto):  The onnx model.
- <b>`node_op_type`</b> (str):  The node's op_type whose following nodes as well as itself will be  removed.
- <b>`fail_if_not_found`</b> (bool):  If true, raise an error if no node matched the given op_type.

**Raises:**

- <b>`ValueError`</b>:  If no node matched the given op_type and fail_if_not_found is set to True

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/onnx_model_manipulations.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `clean_graph_after_node_op_type`

```python
clean_graph_after_node_op_type(
    onnx_model: ModelProto,
    node_op_type: str,
    fail_if_not_found: bool = True
)
```

Remove the nodes following first node matching node_op_type from the ONNX graph.

**Args:**

- <b>`onnx_model`</b> (onnx.ModelProto):  The onnx model.
- <b>`node_op_type`</b> (str):  The node's op_type whose following nodes will be removed.
- <b>`fail_if_not_found`</b> (bool):  If true, raise an error if no node matched the given op_type.

**Raises:**

- <b>`ValueError`</b>:  If no node matched the given op_type and fail_if_not_found is set to True
