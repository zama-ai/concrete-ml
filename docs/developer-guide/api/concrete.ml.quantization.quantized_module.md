<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantized_module.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.quantization.quantized_module`

QuantizedModule API.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantized_module.py#L71"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedModule`

Inference for a quantized model.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantized_module.py#L81"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    ordered_module_input_names: Iterable[str] = None,
    ordered_module_output_names: Iterable[str] = None,
    quant_layers_dict: Dict[str, Tuple[Tuple[str, ], QuantizedOp]] = None
)
```

______________________________________________________________________

#### <kbd>property</kbd> is_compiled

Indicate if the model is compiled.

**Returns:**

- <b>`bool`</b>:  If the model is compiled.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

.. # noqa: DAR201

**Returns:**

- <b>`_onnx_model`</b> (onnx.ModelProto):  the ONNX model

______________________________________________________________________

#### <kbd>property</kbd> post_processing_params

Get the post-processing parameters.

**Returns:**

- <b>`Dict[str, Any]`</b>:  the post-processing parameters

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantized_module.py#L521"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `bitwidth_and_range_report`

```python
bitwidth_and_range_report() → Union[Dict[str, Dict[str, Union[Tuple[int, ], int]]], NoneType]
```

Report the ranges and bitwidths for layers that mix encrypted integer values.

**Returns:**

- <b>`op_names_to_report`</b> (Dict):  a dictionary with operation names as keys. For each  operation, (e.g. conv/gemm/add/avgpool ops), a range and a bitwidth are returned.  The range contains the min/max values encountered when computing the operation and  the bitwidth gives the number of bits needed to represent this range.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantized_module.py#L128"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_compiled`

```python
check_model_is_compiled()
```

Check if the quantized module is compiled.

**Raises:**

- <b>`AttributeError`</b>:  If the quantized module is not compiled.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantized_module.py#L436"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    inputs: Union[Tuple[ndarray, ], ndarray],
    configuration: Optional[Configuration] = None,
    artifacts: Optional[DebugArtifacts] = None,
    show_mlir: bool = False,
    p_error: Optional[float] = None,
    global_p_error: Optional[float] = None,
    verbose: bool = False
) → Circuit
```

Compile the module's forward function.

**Args:**

- <b>`inputs`</b> (numpy.ndarray):  A representative set of input values used for building  cryptographic parameters.
- <b>`configuration`</b> (Optional\[Configuration\]):  Options to use for compilation. Default  to None.
- <b>`artifacts`</b> (Optional\[DebugArtifacts\]):  Artifacts information about the  compilation process to store for debugging.
- <b>`show_mlir`</b> (bool):  Indicate if the MLIR graph should be printed during compilation.
- <b>`p_error`</b> (Optional\[float\]):  Probability of error of a single PBS. A p_error value cannot  be given if a global_p_error value is already set. Default to None, which sets this  error to a default value.
- <b>`global_p_error`</b> (Optional\[float\]):  Probability of error of the full circuit. A  global_p_error value cannot be given if a p_error value is already set. This feature  is not supported during simulation, meaning the probability is  currently set to 0. Default to None, which sets this  error to a default value.
- <b>`verbose`</b> (bool):  Indicate if compilation information should be printed  during compilation. Default to False.

**Returns:**

- <b>`Circuit`</b>:  The compiled Circuit.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantized_module.py#L402"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(q_values: ndarray) → ndarray
```

Take the last layer q_out and use its dequant function.

**Args:**

- <b>`q_values`</b> (numpy.ndarray):  Quantized values of the last layer.

**Returns:**

- <b>`numpy.ndarray`</b>:  Dequantized values of the last layer.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantized_module.py#L213"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(
    *qvalues: ndarray,
    debug: bool = False
) → Union[ndarray, Tuple[ndarray, Union[Dict[Any, Any], NoneType]]]
```

Forward pass with numpy function only.

**Args:**

- <b>`*qvalues (numpy.ndarray)`</b>:  numpy.array containing the quantized values.
- <b>`debug`</b> (bool):  In debug mode, returns quantized intermediary values of the computation.  This is useful when a model's intermediary values in Concrete-ML need  to be compared with the intermediary values obtained in pytorch/onnx.  When set, the second return value is a dictionary containing ONNX  operation names as keys and, as values, their input QuantizedArray or  ndarray. The use can thus extract the quantized or float values of  quantized inputs.

**Returns:**

- <b>`(numpy.ndarray)`</b>:  Predictions of the quantized model

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantized_module.py#L361"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward_and_dequant`

```python
forward_and_dequant(*q_x: ndarray) → ndarray
```

Forward pass with numpy function only plus dequantization.

**Args:**

- <b>`*q_x (numpy.ndarray)`</b>:  numpy.ndarray containing the quantized input values. Requires the  input dtype to be int64.

**Returns:**

- <b>`(numpy.ndarray)`</b>:  Predictions of the quantized model

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantized_module.py#L324"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward_in_fhe`

```python
forward_in_fhe(*qvalues: ndarray, simulate=True) → ndarray
```

Forward function running in FHE or simulated mode.

**Args:**

- <b>`*qvalues (numpy.ndarray)`</b>:  numpy.array containing the quantized values.
- <b>`simulate`</b> (bool):  whether the function should be run in FHE or in simulation mode.

**Returns:**

- <b>`(numpy.ndarray)`</b>:  Predictions of the quantized model

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantized_module.py#L159"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(values: ndarray) → ndarray
```

Apply post-processing to the dequantized values.

For quantized modules, there is no post-processing step but the method is kept to make the API consistent for the client-server API.

**Args:**

- <b>`values`</b> (numpy.ndarray):  The dequantized values to post-process.

**Returns:**

- <b>`numpy.ndarray`</b>:  The post-processed values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantized_module.py#L374"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(*values: ndarray) → Union[ndarray, Tuple[ndarray, ]]
```

Take the inputs in fp32 and quantize it using the learned quantization parameters.

**Args:**

- <b>`values`</b> (numpy.ndarray):  Floating point values.

**Returns:**

- <b>`Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]`</b>:  Quantized (numpy.int64) values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/quantization/quantized_module.py#L419"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_inputs_quantization_parameters`

```python
set_inputs_quantization_parameters(*input_q_params: UniformQuantizer)
```

Set the quantization parameters for the module's inputs.

**Args:**

- <b>`*input_q_params (UniformQuantizer)`</b>:  The quantizer(s) for the module.
