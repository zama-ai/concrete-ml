<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/quantization/quantized_module.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.quantization.quantized_module`

QuantizedModule API.

## **Global Variables**

- **SUPPORTED_FLOAT_TYPES**
- **SUPPORTED_INT_TYPES**
- **USE_OLD_VL**

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_module.py#L82"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedModule`

Inference for a quantized model.

<a href="../../../src/concrete/ml/quantization/quantized_module.py#L92"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    ordered_module_input_names: Optional[Iterable[str]] = None,
    ordered_module_output_names: Optional[Iterable[str]] = None,
    quant_layers_dict: Optional[Dict[str, Tuple[Tuple[str, ], QuantizedOp]]] = None,
    onnx_model: Optional[ModelProto] = None
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

<a href="../../../src/concrete/ml/quantization/quantized_module.py#L754"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `bitwidth_and_range_report`

```python
bitwidth_and_range_report() → Union[Dict[str, Dict[str, Union[Tuple[int, ], int]]], NoneType]
```

Report the ranges and bit-widths for layers that mix encrypted integer values.

**Returns:**

- <b>`op_names_to_report`</b> (Dict):  a dictionary with operation names as keys. For each  operation, (e.g., conv/gemm/add/avgpool ops), a range and a bit-width are returned.  The range contains the min/max values encountered when computing the operation and  the bit-width gives the number of bits needed to represent this range.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_module.py#L222"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_compiled`

```python
check_model_is_compiled()
```

Check if the quantized module is compiled.

**Raises:**

- <b>`AttributeError`</b>:  If the quantized module is not compiled.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_module.py#L625"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    inputs: Union[Tuple[ndarray, ], ndarray],
    configuration: Optional[Configuration] = None,
    artifacts: Optional[DebugArtifacts] = None,
    show_mlir: bool = False,
    p_error: Optional[float] = None,
    global_p_error: Optional[float] = None,
    verbose: bool = False,
    inputs_encryption_status: Optional[Sequence[str]] = None
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
- <b>`inputs_encryption_status`</b> (Optional\[Sequence\[str\]\]):  encryption status ('clear',  'encrypted') for each input.

**Returns:**

- <b>`Circuit`</b>:  The compiled Circuit.

**Raises:**

- <b>`ValueError`</b>:  if inputs_encryption_status does not match with the  parameters of the quantized module

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_module.py#L581"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(*q_y_preds: ndarray) → Union[ndarray, Tuple[ndarray, ]]
```

Take the last layer q_out and use its de-quant function.

**Args:**

- <b>`q_y_preds`</b> (numpy.ndarray):  Quantized outputs values.

**Returns:**

- <b>`Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]`</b>:  De-quantized output values of  the last layer.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_module.py#L205"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump`

```python
dump(file: <class 'TextIO'>) → None
```

Dump itself to a file.

**Args:**

- <b>`file`</b> (TextIO):  The file to dump the serialized object into.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_module.py#L152"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict
```

Dump itself to a dict.

**Returns:**

- <b>`metadata`</b> (Dict):  Dict of serialized objects.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_module.py#L197"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dumps`

```python
dumps() → str
```

Dump itself to a string.

**Returns:**

- <b>`metadata`</b> (str):  String of the serialized object.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_module.py#L313"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(
    *x: ndarray,
    fhe: Union[FheMode, str] = <FheMode.DISABLE: 'disable'>,
    debug: bool = False
) → Union[ndarray, Tuple[ndarray, ], Tuple[Union[Tuple[ndarray, ], ndarray], Dict[str, Dict[Union[int, str], Union[ndarray, QuantizedArray, NoneType, bool, int, float]]]]]
```

Forward pass with numpy function only on floating points.

This method executes the forward pass in the clear, with simulation or in FHE. Input values are expected to be floating points, as the method handles the quantization step. The returned values are floating points as well.

**Args:**

- <b>`*x (numpy.ndarray)`</b>:  Input float values to consider.
- <b>`fhe`</b> (Union\[FheMode, str\]):  The mode to use for prediction. Can be FheMode.DISABLE for  Concrete ML Python inference, FheMode.SIMULATE for FHE simulation and  FheMode.EXECUTE for actual FHE execution. Can also be the string representation of  any of these values. Default to FheMode.DISABLE.
- <b>`debug`</b> (bool):  In debug mode, returns quantized intermediary values of the computation.  This is useful when a model's intermediary values in Concrete ML need to be  compared with the intermediary values obtained in pytorch/onnx. When set, the  second return value is a dictionary containing ONNX operation names as keys and,  as values, their input QuantizedArray or ndarray. The use can thus extract the  quantized or float values of quantized inputs. This feature is only available in  FheMode.DISABLE mode. Default to False.

**Returns:**

- <b>`numpy.ndarray`</b>:  Predictions of the quantized model, in floating points.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_module.py#L171"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_dict`

```python
load_dict(metadata: Dict)
```

Load itself from a string.

**Args:**

- <b>`metadata`</b> (Dict):  Dict of serialized objects.

**Returns:**

- <b>`QuantizedModule`</b>:  The loaded object.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_module.py#L253"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(values: ndarray) → ndarray
```

Apply post-processing to the de-quantized values.

For quantized modules, there is no post-processing step but the method is kept to make the API consistent for the client-server API.

**Args:**

- <b>`values`</b> (numpy.ndarray):  The de-quantized values to post-process.

**Returns:**

- <b>`numpy.ndarray`</b>:  The post-processed values.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_module.py#L556"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(*x: ndarray) → Union[ndarray, Tuple[ndarray, ]]
```

Take the inputs in fp32 and quantize it using the learned quantization parameters.

**Args:**

- <b>`x`</b> (numpy.ndarray):  Floating point x.

**Returns:**

- <b>`Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]`</b>:  Quantized (numpy.int64) x.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_module.py#L387"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantized_forward`

```python
quantized_forward(
    *q_x: ndarray,
    fhe: Union[FheMode, str] = <FheMode.DISABLE: 'disable'>
) → Union[Tuple[ndarray, ], ndarray]
```

Forward function for the FHE circuit.

**Args:**

- <b>`*q_x (numpy.ndarray)`</b>:  Input integer values to consider.
- <b>`fhe`</b> (Union\[FheMode, str\]):  The mode to use for prediction. Can be FheMode.DISABLE for  Concrete ML Python inference, FheMode.SIMULATE for FHE simulation and  FheMode.EXECUTE for actual FHE execution. Can also be the string representation of  any of these values. Default to FheMode.DISABLE.

**Returns:**

- <b>`(Union[numpy.ndarray, Tuple[numpy.ndarray, ...]])`</b>:  Predictions of the quantized model, with integer values.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_module.py#L607"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_inputs_quantization_parameters`

```python
set_inputs_quantization_parameters(*input_q_params: UniformQuantizer)
```

Set the quantization parameters for the module's inputs.

**Args:**

- <b>`*input_q_params (UniformQuantizer)`</b>:  The quantizer(s) for the module.

______________________________________________________________________

<a href="../../../src/concrete/ml/quantization/quantized_module.py#L141"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_reduce_sum_copy`

```python
set_reduce_sum_copy()
```

Set reduce sum to copy or not the inputs.

Due to bit-width propagation in the compilation we need, in some situations, to copy the inputs with a PBS to avoid it.
