<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.torch.hybrid_model`

Implement the conversion of a torch model to a hybrid fhe/torch inference.

## **Global Variables**

- **MAX_BITWIDTH_BACKWARD_COMPATIBLE**

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L43"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L55"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `underscore_str_to_tuple`

```python
underscore_str_to_tuple(tup: str) → Tuple
```

Convert a a string representation of a tuple to a tuple.

**Args:**

- <b>`tup`</b> (str):  a string representing the tuple

**Returns:**

- <b>`Tuple`</b>:  a tuple to change into string representation

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L68"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `HybridFHEMode`

Simple enum for different modes of execution of HybridModel.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L108"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `RemoteModule`

A wrapper class for the modules to be evaluated remotely with FHE.

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L111"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L217"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x: Tensor) → Union[Tensor, QuantTensor]
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

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L133"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L278"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L339"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `HybridFHEModel`

Convert a model to a hybrid model.

This is done by converting targeted modules by RemoteModules. This will modify the model in place.

**Args:**

- <b>`model`</b> (nn.Module):  The model to modify (in-place modification)
- <b>`module_names`</b> (Union\[str, List\[str\]\]):  The module name(s) to replace with FHE server.
- <b>`server_remote_address)`</b>:  The remote address of the FHE server
- <b>`model_name`</b> (str):  Model name identifier
- <b>`verbose`</b> (int):  If logs should be printed when interacting with FHE server

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L353"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L455"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile_model`

```python
compile_model(
    x: Tensor,
    n_bits: Union[int, Dict[str, int]] = 8,
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

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L438"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L562"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `publish_to_hub`

```python
publish_to_hub()
```

Allow the user to push the model and FHE required files to HF Hub.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L538"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_and_clear_private_info`

```python
save_and_clear_private_info(path: Path, via_mlir=False)
```

Save the PyTorch model to the provided path and also saves the corresponding FHE circuit.

**Args:**

- <b>`path`</b> (Path):  The directory where the model and the FHE circuit will be saved.
- <b>`via_mlir`</b> (bool):  if fhe circuits should be serialized using via_mlir option  useful for cross-platform (compile on one architecture and run on another)

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L566"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_fhe_mode`

```python
set_fhe_mode(hybrid_fhe_mode: Union[str, HybridFHEMode])
```

Set Hybrid FHE mode for all remote modules.

**Args:**

- <b>`hybrid_fhe_mode`</b> (Union\[str, HybridFHEMode\]):  Hybrid FHE mode to set to all  remote modules.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L577"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LoggerStub`

Placeholder type for a typical logger like the one from loguru.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L580"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `info`

```python
info(msg: str)
```

Placholder function for logger.info.

**Args:**

- <b>`msg`</b> (str):  the message to output

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L619"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `HybridFHEModelServer`

Hybrid FHE Model Server.

This is a class object to server FHE models serialized using HybridFHEModel.

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L625"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(key_path: Path, model_dir: Path, logger: Optional[LoggerStub])
```

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L772"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `add_key`

```python
add_key(key: bytes, model_name: str, module_name: str, input_shape: str)
```

Add public key.

**Arguments:**

- <b>`key`</b> (bytes):  public key
- <b>`model_name`</b> (str):  model name
- <b>`module_name`</b> (str):  name of the module in the model
- <b>`input_shape`</b> (str):  input shape of said module

**Returns:**
Dict\[str, str\]
\- uid: uid a personal uid

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L694"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_inputs`

```python
check_inputs(
    model_name: str,
    module_name: Optional[str],
    input_shape: Optional[str]
)
```

Check that the given configuration exist in the compiled models folder.

**Args:**

- <b>`model_name`</b> (str):  name of the model
- <b>`module_name`</b> (Optional\[str\]):  name of the module in the model
- <b>`input_shape`</b> (Optional\[str\]):  input shape of the module

**Raises:**

- <b>`ValueError`</b>:  if the given configuration does not exist.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L796"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compute`

```python
compute(
    model_input: bytes,
    uid: str,
    model_name: str,
    module_name: str,
    input_shape: str
)
```

Compute the circuit over encrypted input.

**Arguments:**

- <b>`model_input`</b> (bytes):  input of the circuit
- <b>`uid`</b> (str):  uid of the public key to use
- <b>`model_name`</b> (str):  model name
- <b>`module_name`</b> (str):  name of the module in the model
- <b>`input_shape`</b> (str):  input shape of said module

**Returns:**

- <b>`bytes`</b>:  the result of the circuit

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L668"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_key`

```python
dump_key(key_bytes: bytes, uid: Union[UUID, str]) → None
```

Dump a public key to a stream.

**Args:**

- <b>`key_bytes`</b> (bytes):  stream to dump the public serialized key to
- <b>`uid`</b> (Union\[str, uuid.UUID\]):  uid of the public key to dump

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L678"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_circuit`

```python
get_circuit(model_name, module_name, input_shape)
```

Get circuit based on model name, module name and input shape.

**Args:**

- <b>`model_name`</b> (str):  name of the model
- <b>`module_name`</b> (str):  name of the module in the model
- <b>`input_shape`</b> (str):  input shape of the module

**Returns:**

- <b>`FHEModelServer`</b>:  a fhe model server of the given module of the given model  for the given shape

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L750"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_client`

```python
get_client(model_name: str, module_name: str, input_shape: str)
```

Get client.

**Args:**

- <b>`model_name`</b> (str):  name of the model
- <b>`module_name`</b> (str):  name of the module in the model
- <b>`input_shape`</b> (str):  input shape of the module

**Returns:**

- <b>`Path`</b>:  the path to the correct client

**Raises:**

- <b>`ValueError`</b>:  if client couldn't be found

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L725"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `list_modules`

```python
list_modules(model_name: str)
```

List all modules in a model.

**Args:**

- <b>`model_name`</b> (str):  name of the model

**Returns:**
Dict\[str, Dict\[str, Dict\]\]

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L737"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `list_shapes`

```python
list_shapes(model_name: str, module_name: str)
```

List all modules in a model.

**Args:**

- <b>`model_name`</b> (str):  name of the model
- <b>`module_name`</b> (str):  name of the module in the model

**Returns:**
Dict\[str, Dict\]

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_model.py#L657"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_key`

```python
load_key(uid: Union[str, UUID]) → bytes
```

Load a public key from the key path in the file system.

**Args:**

- <b>`uid`</b> (Union\[str, uuid.UUID\]):  uid of the public key to load

**Returns:**

- <b>`bytes`</b>:  the bytes of the public key
