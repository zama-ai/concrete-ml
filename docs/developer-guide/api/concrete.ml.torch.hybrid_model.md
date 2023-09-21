<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.torch.hybrid_model`

Implement the conversion of a torch model to a hybrid fhe/torch inference.

## **Global Variables**

- **MAX_BITWIDTH_BACKWARD_COMPATIBLE**

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `tuple_to_underscore_str`

```python
tuple_to_underscore_str(tup: Tuple) → str
```

Convert a tuple to a string representation.

**Args:**

- <b>`tup`</b> (Tuple):  a tuple to change into string representation

**Returns:**

- <b>`str`</b>:  a string representing the tuple

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L46"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L24"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `HybridFHEMode`

Simple enum for different modes of execution of HybridModel.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `RemoteModule`

A wrapper class for the modules to be done remotely with FHE.

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L89"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    module: Optional[Module] = None,
    server_remote_address: Optional[str] = None,
    module_name: Optional[str] = None,
    model_name: Optional[str] = None,
    verbose: int = 0
)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L195"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x: Tensor) → Tensor
```

Forward pass of the remote module.

To change the behavior of this forward function one must change the fhe_local_mode attribute. Choices are:

- disable: forward using torch module
- remote: forward with fhe client-server
- simulate: forward with local fhe simulation
- calibrate: forward for calibration

**Args:**

- <b>`x`</b> (torch.Tensor):  The input tensor.

**Returns:**

- <b>`(torch.Tensor)`</b>:  The output tensor.

**Raises:**

- <b>`ValueError`</b>:  if local_fhe_mode is not supported

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L111"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `init_fhe_client`

```python
init_fhe_client(
    path_to_client: Optional[Path] = None,
    path_to_keys: Optional[Path] = None
)
```

Set the clients keys.

**Args:**

- <b>`path_to_client`</b> (str):  Path where the client.zip is located.
- <b>`path_to_keys`</b> (str):  Path where keys are located.

**Raises:**

- <b>`ValueError`</b>:  if anything goes wrong with the server.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L252"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `remote_call`

```python
remote_call(x: Tensor) → Tensor
```

Call the remote server to get the private module inference.

**Args:**

- <b>`x`</b> (torch.Tensor):  The input tensor.

**Returns:**

- <b>`torch.Tensor`</b>:  The result of the FHE computation

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L315"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `HybridFHEModel`

Convert a model to a hybrid model.

This is done by converting targeted modules by RemoteModules. This will modify the model in place.

**Args:**

- <b>`model`</b> (nn.Module):  The model to modify (in-place modification)
- <b>`module_names`</b> (Union\[str, List\[str\]\]):  The module name(s) to replace with FHE server.
- <b>`server_remote_address)`</b>:  The remote address of the FHE server
- <b>`model_name`</b> (str):  Model name identifier
- <b>`verbose`</b> (int):  If logs should be printed when interacting with FHE server

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L329"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    model: Module,
    module_names: Union[str, List[str]],
    server_remote_address=None,
    model_name: str = 'model',
    verbose: int = 0
)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L431"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile_model`

```python
compile_model(
    x: Tensor,
    n_bits: int = 8,
    rounding_threshold_bits: Optional[int] = None,
    p_error: Optional[float] = None,
    configuration: Optional[Configuration] = None
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

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L414"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `init_client`

```python
init_client(
    path_to_clients: Optional[Path] = None,
    path_to_keys: Optional[Path] = None
)
```

Initialize client for all remote modules.

**Args:**

- <b>`path_to_clients`</b> (Optional\[Path\]):  Path to the client.zip files.
- <b>`path_to_keys`</b> (Optional\[Path\]):  Path to the keys folder.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L528"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `publish_to_hub`

```python
publish_to_hub()
```

Allow the user to push the model and FHE required files to HF Hub.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L504"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_and_clear_private_info`

```python
save_and_clear_private_info(path: Path, via_mlir=False)
```

Save the PyTorch model to the provided path and also saves the corresponding FHE circuit.

**Args:**

- <b>`path`</b> (Path):  The directory where the model and the FHE circuit will be saved.
- <b>`via_mlir`</b> (bool):  if fhe circuits should be serialized using via_mlir option  useful for cross-platform (compile on one architecture and run on another)
