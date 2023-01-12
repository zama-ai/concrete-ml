<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/torch/compile.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.torch.compile`

torch compilation function.

## **Global Variables**

- **MAX_BITWIDTH_BACKWARD_COMPATIBLE**
- **OPSET_VERSION_FOR_ONNX_EXPORT**

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/torch/compile.py#L30"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/torch/compile.py#L147"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compile_torch_model`

```python
compile_torch_model(
    torch_model: Module,
    torch_inputset: Union[Tensor, ndarray, Tuple[Union[Tensor, ndarray], ]],
    import_qat: bool = False,
    configuration: Optional[Configuration] = None,
    compilation_artifacts: Optional[DebugArtifacts] = None,
    show_mlir: bool = False,
    n_bits=8,
    use_virtual_lib: bool = False,
    p_error: Optional[float] = None,
    global_p_error: Optional[float] = None,
    verbose_compilation: bool = False
) → QuantizedModule
```

Compile a torch module into a FHE equivalent.

Take a model in torch, turn it to numpy, quantize its inputs / weights / outputs and finally compile it with Concrete-Numpy

**Args:**

- <b>`torch_model`</b> (torch.nn.Module):  the model to quantize
- <b>`torch_inputset`</b> (Dataset):  the calibration inputset, can contain either torch  tensors or numpy.ndarray.
- <b>`import_qat`</b> (bool):  Set to True to import a network that contains quantizers and was  trained using quantization aware training
- <b>`configuration`</b> (Configuration):  Configuration object to use  during compilation
- <b>`compilation_artifacts`</b> (DebugArtifacts):  Artifacts object to fill  during compilation
- <b>`show_mlir`</b> (bool):  if set, the MLIR produced by the converter and which is going  to be sent to the compiler backend is shown on the screen, e.g., for debugging or demo
- <b>`n_bits`</b>:  the number of bits for the quantization
- <b>`use_virtual_lib`</b> (bool):  set to use the so called virtual lib simulating FHE computation.  Defaults to False
- <b>`p_error`</b> (Optional\[float\]):  probability of error of a single PBS
- <b>`global_p_error`</b> (Optional\[float\]):  probability of error of the full circuit. Not simulated  by the VL, i.e., taken as 0
- <b>`verbose_compilation`</b> (bool):  whether to show compilation information

**Returns:**

- <b>`QuantizedModule`</b>:  The resulting compiled QuantizedModule.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/torch/compile.py#L204"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compile_onnx_model`

```python
compile_onnx_model(
    onnx_model: ModelProto,
    torch_inputset: Union[Tensor, ndarray, Tuple[Union[Tensor, ndarray], ]],
    import_qat: bool = False,
    configuration: Optional[Configuration] = None,
    compilation_artifacts: Optional[DebugArtifacts] = None,
    show_mlir: bool = False,
    n_bits=8,
    use_virtual_lib: bool = False,
    p_error: Optional[float] = None,
    global_p_error: Optional[float] = None,
    verbose_compilation: bool = False
) → QuantizedModule
```

Compile a torch module into a FHE equivalent.

Take a model in torch, turn it to numpy, quantize its inputs / weights / outputs and finally compile it with Concrete-Numpy

**Args:**

- <b>`onnx_model`</b> (onnx.ModelProto):  the model to quantize
- <b>`torch_inputset`</b> (Dataset):  the calibration inputset, can contain either torch  tensors or numpy.ndarray.
- <b>`import_qat`</b> (bool):  Flag to signal that the network being imported contains quantizers in  in its computation graph and that Concrete ML should not requantize it.
- <b>`configuration`</b> (Configuration):  Configuration object to use  during compilation
- <b>`compilation_artifacts`</b> (DebugArtifacts):  Artifacts object to fill  during compilation
- <b>`show_mlir`</b> (bool):  if set, the MLIR produced by the converter and which is going  to be sent to the compiler backend is shown on the screen, e.g., for debugging or demo
- <b>`n_bits`</b>:  the number of bits for the quantization
- <b>`use_virtual_lib`</b> (bool):  set to use the so called virtual lib simulating FHE computation.  Defaults to False.
- <b>`p_error`</b> (Optional\[float\]):  probability of error of a single PBS
- <b>`global_p_error`</b> (Optional\[float\]):  probability of error of the full circuit. Not simulated  by the VL, i.e., taken as 0
- <b>`verbose_compilation`</b> (bool):  whether to show compilation information

**Returns:**

- <b>`QuantizedModule`</b>:  The resulting compiled QuantizedModule.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/torch/compile.py#L269"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `compile_brevitas_qat_model`

```python
compile_brevitas_qat_model(
    torch_model: Module,
    torch_inputset: Union[Tensor, ndarray, Tuple[Union[Tensor, ndarray], ]],
    n_bits: Union[int, dict],
    configuration: Optional[Configuration] = None,
    compilation_artifacts: Optional[DebugArtifacts] = None,
    show_mlir: bool = False,
    use_virtual_lib: bool = False,
    p_error: Optional[float] = None,
    global_p_error: Optional[float] = None,
    output_onnx_file: Union[Path, str] = None,
    verbose_compilation: bool = False
) → QuantizedModule
```

Compile a Brevitas Quantization Aware Training model.

The torch_model parameter is a subclass of torch.nn.Module that uses quantized operations from brevitas.qnn. The model is trained before calling this function. This function compiles the trained model to FHE.

**Args:**

- <b>`torch_model`</b> (torch.nn.Module):  the model to quantize
- <b>`torch_inputset`</b> (Dataset):  the calibration inputset, can contain either torch  tensors or numpy.ndarray.
- <b>`n_bits`</b> (Union\[int,dict\]):  the number of bits for the quantization
- <b>`configuration`</b> (Configuration):  Configuration object to use  during compilation
- <b>`compilation_artifacts`</b> (DebugArtifacts):  Artifacts object to fill  during compilation
- <b>`show_mlir`</b> (bool):  if set, the MLIR produced by the converter and which is going  to be sent to the compiler backend is shown on the screen, e.g., for debugging or demo
- <b>`use_virtual_lib`</b> (bool):  set to use the so called virtual lib simulating FHE computation,  defaults to False.
- <b>`p_error`</b> (Optional\[float\]):  probability of error of a single PBS
- <b>`global_p_error`</b> (Optional\[float\]):  probability of error of the full circuit. Not simulated  by the VL, i.e., taken as 0
- <b>`output_onnx_file`</b> (str):  temporary file to store ONNX model. If None a temporary file  is generated
- <b>`verbose_compilation`</b> (bool):  whether to show compilation information

**Returns:**

- <b>`QuantizedModule`</b>:  The resulting compiled QuantizedModule.
