<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/torch/compile.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.torch.compile`

torch compilation function.

## **Global Variables**

- **MAX_BITWIDTH_BACKWARD_COMPATIBLE**
- **OPSET_VERSION_FOR_ONNX_EXPORT**

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/compile.py#L36"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `has_any_qnn_layers`

```python
has_any_qnn_layers(torch_model: Module) → bool
```

Check if a torch model has QNN layers.

This is useful to check if a model is a QAT model.

**Args:**

- <b>`torch_model`</b> (torch.nn.Module):  a torch model

**Returns:**

- <b>`bool`</b>:  whether this torch model contains any QNN layer.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/compile.py#L52"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `convert_torch_tensor_or_numpy_array_to_numpy_array`

```python
convert_torch_tensor_or_numpy_array_to_numpy_array(
    torch_tensor_or_numpy_array: Union[Tensor, ndarray]
) → ndarray
```

Convert a torch tensor or a numpy array to a numpy array.

**Args:**

- <b>`torch_tensor_or_numpy_array`</b> (Tensor):  the value that is either  a torch tensor or a numpy array.

**Returns:**

- <b>`numpy.ndarray`</b>:  the value converted to a numpy array.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/compile.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `build_quantized_module`

```python
build_quantized_module(
    model: Union[Module, ModelProto],
    torch_inputset: Union[Tensor, ndarray, Tuple[Union[Tensor, ndarray], ]],
    import_qat: bool = False,
    n_bits: Union[int, Dict[str, int]] = 8,
    rounding_threshold_bits: Union[NoneType, int, Dict[str, Union[str, int]]] = None,
    reduce_sum_copy=False
) → QuantizedModule
```

Build a quantized module from a Torch or ONNX model.

Take a model in torch or ONNX, turn it to numpy, quantize its inputs / weights / outputs and retrieve the associated quantized module.

**Args:**

- <b>`model`</b> (Union\[torch.nn.Module, onnx.ModelProto\]):  The model to quantize, either in torch or  in ONNX.
- <b>`torch_inputset`</b> (Dataset):  the calibration input-set, can contain either torch  tensors or numpy.ndarray
- <b>`import_qat`</b> (bool):  Flag to signal that the network being imported contains quantizers in  in its computation graph and that Concrete ML should not re-quantize it
- <b>`n_bits`</b>:  the number of bits for the quantization
- <b>`rounding_threshold_bits`</b> (Union\[None, int, Dict\[str, Union\[str, int\]\]\]):  Defines precision  rounding for model accumulators. Accepts None, an int, or a dict.  The dict can specify 'method' (fhe.Exactness.EXACT or fhe.Exactness.APPROXIMATE)  and 'n_bits' ('auto' or int)
- <b>`reduce_sum_copy`</b> (bool):  if the inputs of QuantizedReduceSum should be copied to avoid  bit-width propagation

**Returns:**

- <b>`QuantizedModule`</b>:  The resulting QuantizedModule.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/compile.py#L237"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compile_torch_model`

```python
compile_torch_model(
    torch_model: Module,
    torch_inputset: Union[Tensor, ndarray, Tuple[Union[Tensor, ndarray], ]],
    import_qat: bool = False,
    configuration: Optional[Configuration] = None,
    artifacts: Optional[DebugArtifacts] = None,
    show_mlir: bool = False,
    n_bits: Union[int, Dict[str, int]] = 8,
    rounding_threshold_bits: Union[NoneType, int, Dict[str, Union[str, int]]] = None,
    p_error: Optional[float] = None,
    global_p_error: Optional[float] = None,
    verbose: bool = False,
    inputs_encryption_status: Optional[Sequence[str]] = None,
    reduce_sum_copy: bool = False
) → QuantizedModule
```

Compile a torch module into an FHE equivalent.

Take a model in torch, turn it to numpy, quantize its inputs / weights / outputs and finally compile it with Concrete

**Args:**

- <b>`torch_model`</b> (torch.nn.Module):  the model to quantize
- <b>`torch_inputset`</b> (Dataset):  the calibration input-set, can contain either torch  tensors or numpy.ndarray.
- <b>`import_qat`</b> (bool):  Set to True to import a network that contains quantizers and was  trained using quantization aware training
- <b>`configuration`</b> (Configuration):  Configuration object to use  during compilation
- <b>`artifacts`</b> (DebugArtifacts):  Artifacts object to fill  during compilation
- <b>`show_mlir`</b> (bool):  if set, the MLIR produced by the converter and which is going  to be sent to the compiler backend is shown on the screen, e.g., for debugging or demo
- <b>`n_bits`</b> (Union\[int, Dict\[str, int\]\]):  number of bits for quantization, can be a single value  or a dictionary with the following keys :
  \- "op_inputs" and "op_weights" (mandatory)
  \- "model_inputs" and "model_outputs" (optional, default to 5 bits).  When using a single integer for n_bits, its value is assigned to "op_inputs" and  "op_weights" bits. Default is 8 bits.
- <b>`rounding_threshold_bits`</b> (Union\[None, int, Dict\[str, Union\[str, int\]\]\]):  Defines precision  rounding for model accumulators. Accepts None, an int, or a dict.  The dict can specify 'method' (fhe.Exactness.EXACT or fhe.Exactness.APPROXIMATE)  and 'n_bits' ('auto' or int)
- <b>`p_error`</b> (Optional\[float\]):  probability of error of a single PBS
- <b>`global_p_error`</b> (Optional\[float\]):  probability of error of the full circuit. In FHE  simulation `global_p_error` is set to 0
- <b>`verbose`</b> (bool):  whether to show compilation information
- <b>`inputs_encryption_status`</b> (Optional\[Sequence\[str\]\]):  encryption status ('clear', 'encrypted')  for each input. By default all arguments will be encrypted.
- <b>`reduce_sum_copy`</b> (bool):  if the inputs of QuantizedReduceSum should be copied to avoid  bit-width propagation

**Returns:**

- <b>`QuantizedModule`</b>:  The resulting compiled QuantizedModule.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/compile.py#L321"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compile_onnx_model`

```python
compile_onnx_model(
    onnx_model: ModelProto,
    torch_inputset: Union[Tensor, ndarray, Tuple[Union[Tensor, ndarray], ]],
    import_qat: bool = False,
    configuration: Optional[Configuration] = None,
    artifacts: Optional[DebugArtifacts] = None,
    show_mlir: bool = False,
    n_bits: Union[int, Dict[str, int]] = 8,
    rounding_threshold_bits: Union[NoneType, int, Dict[str, Union[str, int]]] = None,
    p_error: Optional[float] = None,
    global_p_error: Optional[float] = None,
    verbose: bool = False,
    inputs_encryption_status: Optional[Sequence[str]] = None,
    reduce_sum_copy: bool = False
) → QuantizedModule
```

Compile a torch module into an FHE equivalent.

Take a model in torch, turn it to numpy, quantize its inputs / weights / outputs and finally compile it with Concrete-Python

**Args:**

- <b>`onnx_model`</b> (onnx.ModelProto):  the model to quantize
- <b>`torch_inputset`</b> (Dataset):  the calibration input-set, can contain either torch  tensors or numpy.ndarray.
- <b>`import_qat`</b> (bool):  Flag to signal that the network being imported contains quantizers in  in its computation graph and that Concrete ML should not re-quantize it.
- <b>`configuration`</b> (Configuration):  Configuration object to use  during compilation
- <b>`artifacts`</b> (DebugArtifacts):  Artifacts object to fill  during compilation
- <b>`show_mlir`</b> (bool):  if set, the MLIR produced by the converter and which is going  to be sent to the compiler backend is shown on the screen, e.g., for debugging or demo
- <b>`n_bits`</b> (Union\[int, Dict\[str, int\]\]):  number of bits for quantization, can be a single value  or a dictionary with the following keys :
  \- "op_inputs" and "op_weights" (mandatory)
  \- "model_inputs" and "model_outputs" (optional, default to 5 bits).  When using a single integer for n_bits, its value is assigned to "op_inputs" and  "op_weights" bits. Default is 8 bits.
- <b>`rounding_threshold_bits`</b> (Union\[None, int, Dict\[str, Union\[str, int\]\]\]):  Defines precision  rounding for model accumulators. Accepts None, an int, or a dict.  The dict can specify 'method' (fhe.Exactness.EXACT or fhe.Exactness.APPROXIMATE)  and 'n_bits' ('auto' or int)
- <b>`p_error`</b> (Optional\[float\]):  probability of error of a single PBS
- <b>`global_p_error`</b> (Optional\[float\]):  probability of error of the full circuit. In FHE  simulation `global_p_error` is set to 0
- <b>`verbose`</b> (bool):  whether to show compilation information
- <b>`inputs_encryption_status`</b> (Optional\[Sequence\[str\]\]):  encryption status ('clear', 'encrypted')  for each input. By default all arguments will be encrypted.
- <b>`reduce_sum_copy`</b> (bool):  if the inputs of QuantizedReduceSum should be copied to avoid  bit-width propagation

**Returns:**

- <b>`QuantizedModule`</b>:  The resulting compiled QuantizedModule.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/compile.py#L401"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compile_brevitas_qat_model`

```python
compile_brevitas_qat_model(
    torch_model: Module,
    torch_inputset: Union[Tensor, ndarray, Tuple[Union[Tensor, ndarray], ]],
    n_bits: Optional[int, Dict[str, int]] = None,
    configuration: Optional[Configuration] = None,
    artifacts: Optional[DebugArtifacts] = None,
    show_mlir: bool = False,
    rounding_threshold_bits: Union[NoneType, int, Dict[str, Union[str, int]]] = None,
    p_error: Optional[float] = None,
    global_p_error: Optional[float] = None,
    output_onnx_file: Union[NoneType, Path, str] = None,
    verbose: bool = False,
    inputs_encryption_status: Optional[Sequence[str]] = None,
    reduce_sum_copy: bool = False
) → QuantizedModule
```

Compile a Brevitas Quantization Aware Training model.

The torch_model parameter is a subclass of torch.nn.Module that uses quantized operations from brevitas.qnn. The model is trained before calling this function. This function compiles the trained model to FHE.

**Args:**

- <b>`torch_model`</b> (torch.nn.Module):  the model to quantize
- <b>`torch_inputset`</b> (Dataset):  the calibration input-set, can contain either torch  tensors or numpy.ndarray.
- <b>`n_bits`</b> (Optional\[Union\[int, dict\]):  the number of bits for the quantization. By default,  for most models, a value of None should be given, which instructs Concrete ML to use the  bit-widths configured using Brevitas quantization options. For some networks, that  perform a non-linear operation on an input on an output, if None is given, a default  value of 8 bits is used for the input/output quantization. For such models the user can  also specify a dictionary with model_inputs/model_outputs keys to override  the 8-bit default or a single integer for both values.
- <b>`configuration`</b> (Configuration):  Configuration object to use  during compilation
- <b>`artifacts`</b> (DebugArtifacts):  Artifacts object to fill  during compilation
- <b>`show_mlir`</b> (bool):  if set, the MLIR produced by the converter and which is going  to be sent to the compiler backend is shown on the screen, e.g., for debugging or demo
- <b>`rounding_threshold_bits`</b> (Union\[None, int, Dict\[str, Union\[str, int\]\]\]):  Defines precision  rounding for model accumulators. Accepts None, an int, or a dict.  The dict can specify 'method' (fhe.Exactness.EXACT or fhe.Exactness.APPROXIMATE)  and 'n_bits' ('auto' or int)
- <b>`p_error`</b> (Optional\[float\]):  probability of error of a single PBS
- <b>`global_p_error`</b> (Optional\[float\]):  probability of error of the full circuit. In FHE  simulation `global_p_error` is set to 0
- <b>`output_onnx_file`</b> (str):  temporary file to store ONNX model. If None a temporary file  is generated
- <b>`verbose`</b> (bool):  whether to show compilation information
- <b>`inputs_encryption_status`</b> (Optional\[Sequence\[str\]\]):  encryption status ('clear', 'encrypted')  for each input. By default all arguments will be encrypted.
- <b>`reduce_sum_copy`</b> (bool):  if the inputs of QuantizedReduceSum should be copied to avoid  bit-width propagation

**Returns:**

- <b>`QuantizedModule`</b>:  The resulting compiled QuantizedModule.
