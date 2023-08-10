<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.torch.hybrid_model`

Implement the conversion of a torch model to a hybrid fhe/torch inference.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `convert_conv1d_to_linear`

```python
convert_conv1d_to_linear(layer_or_module)
```

Convert all Conv1D layers in a module or a Conv1D layer itself to nn.Linear.

**Args:**

- <b>`layer_or_module`</b> (nn.Module or Conv1D):  The module which will be recursively searched  for Conv1D layers, or a Conv1D layer itself.

**Returns:**

- <b>`nn.Module or nn.Linear`</b>:  The updated module with Conv1D layers converted to Linear layers,  or the Conv1D layer converted to a Linear layer.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L56"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `RemoteModule`

A wrapper class for the modules to be done remotely with FHE.

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L59"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(module=None, server_remote_address=None)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x: Tensor) → Tensor
```

Forward pass of the remote module.

**Args:**

- <b>`x`</b> (torch.Tensor):  The input tensor.

**Returns:**

- <b>`(torch.Tensor)`</b>:  The output tensor.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `init_fhe_client`

```python
init_fhe_client(path_to_client: str, path_to_keys: str)
```

Set the clients keys.

**Args:**

- <b>`path_to_client`</b> (str):  Path where the client.zip is located.
- <b>`path_to_keys`</b> (str):  Path where keys are located.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L110"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `remote_call`

```python
remote_call(x: Tensor)
```

Call the remote server to get the private module inference.

**Args:**

- <b>`x`</b> (torch.Tensor):  The input tensor.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L120"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `HybridFHEModel`

Convert a model to a hybrid model.

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L123"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    model: Module,
    module_names: Union[str, List[str]],
    server_remote_address=None
)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L206"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile_model`

```python
compile_model(
    x: Tensor,
    n_bits: int = 8,
    rounding_threshold_bits: int = 8,
    p_error=0.01,
    configuration: Configuration = None
)
```

Compiles the specific layers to FHE.

**Args:**

- <b>`x`</b> (torch.Tensor):  The input tensor for the model. This is used to run the model  once for calibration.
- <b>`n_bits`</b> (int):  The bit precision for quantization during FHE model compilation.  Default is 8.
- <b>`rounding_threshold_bits`</b> (int):  The number of bits to use for rounding threshold during  FHE model compilation. Default is 8.
- <b>`p_error`</b> (float):  Error allowed for each table look-up in the circuit.
- <b>`configuration`</b> (Configuration):  A concrete Configuration object specifying the FHE  encryption parameters. If not specified, a default configuration is used.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L196"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `init_client`

```python
init_client(path_to_client: str, path_to_keys: str)
```

Initialize client for all remote modules.

**Args:**

- <b>`path_to_client`</b> (str):  Path to the client.zip files.
- <b>`path_to_keys`</b> (str):  Path to the keys folder.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L285"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `publish_to_hub`

```python
publish_to_hub()
```

Allow the user to push the model and FHE required files to HF Hub.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L262"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_and_clear_private_info`

```python
save_and_clear_private_info(path: Path)
```

Save the PyTorch model to the provided path and also saves the corresponding FHE circuit.

**Args:**

- <b>`path`</b> (Path):  The directory where the model and the FHE circuit will be saved.
