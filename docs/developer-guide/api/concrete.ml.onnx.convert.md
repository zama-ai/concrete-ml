<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/onnx/convert.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.onnx.convert`

ONNX conversion related code.

## **Global Variables**

- **IMPLEMENTED_ONNX_OPS**
- **OPSET_VERSION_FOR_ONNX_EXPORT**

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/onnx/convert.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_equivalent_numpy_forward_and_onnx_model`

```python
get_equivalent_numpy_forward_and_onnx_model(
    torch_module: Module,
    dummy_input: Union[Tensor, Tuple[Tensor, ]],
    output_onnx_file: Optional[Path, str] = None
) → Tuple[Callable[, Tuple[ndarray, ]], GraphProto]
```

Get the numpy equivalent forward of the provided torch Module.

**Args:**

- <b>`torch_module`</b> (torch.nn.Module):  the torch Module for which to get the equivalent numpy  forward.
- <b>`dummy_input`</b> (Union\[torch.Tensor, Tuple\[torch.Tensor, ...\]\]):  dummy inputs for ONNX export.
- <b>`output_onnx_file`</b> (Optional\[Union\[Path, str\]\], optional):  Path to save the ONNX file to. Will  use a temp file if not provided.  Defaults to None.

**Returns:**

- <b>`Tuple[Callable[..., Tuple[numpy.ndarray, ...]], onnx.GraphProto]`</b>:  The function that will  execute the equivalent numpy code to the passed torch_module and the generated ONNX  model.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/onnx/convert.py#L68"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_equivalent_numpy_forward`

```python
get_equivalent_numpy_forward(
    onnx_model: ModelProto,
    check_model: bool = True
) → Callable[, Tuple[ndarray, ]]
```

Get the numpy equivalent forward of the provided ONNX model.

**Args:**

- <b>`onnx_model`</b> (onnx.ModelProto):  the ONNX model for which to get the equivalent numpy  forward.
- <b>`check_model`</b> (bool):  set to True to run the onnx checker on the model.  Defaults to True.

**Raises:**

- <b>`ValueError`</b>:  Raised if there is an unsupported ONNX operator required to convert the torch  model to numpy.

**Returns:**

- <b>`Callable[..., Tuple[numpy.ndarray, ...]]`</b>:  The function that will execute  the equivalent numpy function.
