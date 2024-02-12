<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/onnx/convert.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.onnx.convert`

ONNX conversion related code.

## **Global Variables**

- **IMPLEMENTED_ONNX_OPS**
- **OPSET_VERSION_FOR_ONNX_EXPORT**

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/convert.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `fuse_matmul_bias_to_gemm`

```python
fuse_matmul_bias_to_gemm(onnx_model: ModelProto)
```

Fuse sequence of matmul -> add into a gemm node.

**Args:**

- <b>`onnx_model`</b> (onnx.ModelProto):  A onnx model to optimize using Mat-Mult + Add -> Gemm

**Returns:**

- <b>`onnx.ModelProto`</b>:  the optimized onnx model

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/convert.py#L118"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_equivalent_numpy_forward_from_torch`

```python
get_equivalent_numpy_forward_from_torch(
    torch_module: Module,
    dummy_input: Union[Tensor, Tuple[Tensor, ]],
    output_onnx_file: Union[NoneType, Path, str] = None
) → Tuple[Callable[, Tuple[ndarray, ]], ModelProto]
```

Get the numpy equivalent forward of the provided torch Module.

**Args:**

- <b>`torch_module`</b> (torch.nn.Module):  the torch Module for which to get the equivalent numpy  forward.
- <b>`dummy_input`</b> (Union\[torch.Tensor, Tuple\[torch.Tensor, ...\]\]):  dummy inputs for ONNX export.
- <b>`output_onnx_file`</b> (Optional\[Union\[Path, str\]\]):  Path to save the ONNX file to. Will  use a temp file if not provided.  Defaults to None.

**Returns:**

- <b>`Tuple[Callable[..., Tuple[numpy.ndarray, ...]], onnx.GraphProto]`</b>:  The function that will  execute the equivalent numpy code to the passed torch_module and the generated ONNX  model.

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/convert.py#L178"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `preprocess_onnx_model`

```python
preprocess_onnx_model(onnx_model: ModelProto, check_model: bool) → ModelProto
```

Get the numpy equivalent forward of the provided ONNX model.

**Args:**

- <b>`onnx_model`</b> (onnx.ModelProto):  the ONNX model for which to get the equivalent numpy  forward.
- <b>`check_model`</b> (bool):  set to True to run the onnx checker on the model.  Defaults to True.

**Raises:**

- <b>`ValueError`</b>:  Raised if there is an unsupported ONNX operator required to convert the torch  model to numpy.

**Returns:**

- <b>`onnx.ModelProto`</b>:  The preprocessed ONNX model.

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/convert.py#L240"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_equivalent_numpy_forward_from_onnx`

```python
get_equivalent_numpy_forward_from_onnx(
    onnx_model: ModelProto,
    check_model: bool = True
) → Tuple[Callable[, Tuple[ndarray, ]], ModelProto]
```

Get the numpy equivalent forward of the provided ONNX model.

**Args:**

- <b>`onnx_model`</b> (onnx.ModelProto):  the ONNX model for which to get the equivalent numpy  forward.
- <b>`check_model`</b> (bool):  set to True to run the onnx checker on the model.  Defaults to True.

**Returns:**

- <b>`Callable[..., Tuple[numpy.ndarray, ...]]`</b>:  The function that will execute  the equivalent numpy function.

______________________________________________________________________

<a href="../../../src/concrete/ml/onnx/convert.py#L265"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_equivalent_numpy_forward_from_onnx_tree`

```python
get_equivalent_numpy_forward_from_onnx_tree(
    onnx_model: ModelProto,
    check_model: bool = True,
    lsbs_to_remove_for_trees: Optional[Tuple[int, int]] = None
) → Tuple[Callable[, Tuple[ndarray, ]], ModelProto]
```

Get the numpy equivalent forward of the provided ONNX model for tree-based models only.

**Args:**

- <b>`onnx_model`</b> (onnx.ModelProto):  the ONNX model for which to get the equivalent numpy  forward.
- <b>`check_model`</b> (bool):  set to True to run the onnx checker on the model.  Defaults to True.
- <b>`lsbs_to_remove_for_trees`</b> (Optional\[Tuple\[int, int\]\]):  This parameter is exclusively used for  optimizing tree-based models. It contains the values of the least significant bits to  remove during the tree traversal, where the first value refers to the first comparison  (either "less" or "less_or_equal"), while the second value refers to the "Equal"  comparison operation. Default to None, as it is not applicable to other types of models.

**Returns:**

- <b>`Tuple[Callable[..., Tuple[numpy.ndarray, ...]], onnx.ModelProto]`</b>:  The function that will  execute the equivalent numpy function.
