<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/torch/numpy_module.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.torch.numpy_module`

A torch to numpy module.

## **Global Variables**

- **OPSET_VERSION_FOR_ONNX_EXPORT**

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/numpy_module.py#L19"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NumpyModule`

General interface to transform a torch.nn.Module to numpy module.

**Args:**

- <b>`torch_model`</b> (Union\[nn.Module, onnx.ModelProto\]):  A fully trained, torch model along with  its parameters or the onnx graph of the model.
- <b>`dummy_input`</b> (Union\[torch.Tensor, Tuple\[torch.Tensor, ...\]\]):  Sample tensors for all the  module inputs, used in the ONNX export to get a simple to manipulate nn representation.
- <b>`debug_onnx_output_file_path`</b>:  (Optional\[Union\[Path, str\]\], optional): An optional path to  indicate where to save the ONNX file exported by torch for debug.  Defaults to None.

<a href="../../../src/concrete/ml/torch/numpy_module.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    model: Union[Module, ModelProto],
    dummy_input: Optional[Tensor, Tuple[Tensor, ]] = None,
    debug_onnx_output_file_path: Optional[Path, str] = None
)
```

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

.. # noqa: DAR201

**Returns:**

- <b>`_onnx_model`</b> (onnx.ModelProto):  the ONNX model

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/numpy_module.py#L78"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(*args: ndarray) → Union[ndarray, Tuple[ndarray, ]]
```

Apply a forward pass on args with the equivalent numpy function only.

**Args:**

- <b>`*args`</b>:  the inputs of the forward function

**Returns:**

- <b>`Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]`</b>:  result of the forward on the given  inputs
