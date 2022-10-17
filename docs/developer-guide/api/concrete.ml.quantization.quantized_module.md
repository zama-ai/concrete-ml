<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_module.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.quantization.quantized_module`

QuantizedModule API.

## **Global Variables**

- **DEFAULT_P_ERROR_PBS**

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_module.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedModule`

Inference for a quantized model.

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_module.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    ordered_module_input_names: Iterable[str] = None,
    ordered_module_output_names: Iterable[str] = None,
    quant_layers_dict: Dict[str, Tuple[Tuple[str, ], QuantizedOp]] = None
)
```

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

**Returns:**

- <b>`Circuit`</b>:  the FHE circuit

______________________________________________________________________

#### <kbd>property</kbd> is_compiled

Return the compiled status of the module.

**Returns:**

- <b>`bool`</b>:  the compiled status of the module.

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

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_module.py#L301"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    q_inputs: Union[Tuple[ndarray, ], ndarray],
    configuration: Optional[Configuration] = None,
    compilation_artifacts: Optional[DebugArtifacts] = None,
    show_mlir: bool = False,
    use_virtual_lib: bool = False,
    p_error: Optional[float] = 6.3342483999973e-05
) → Circuit
```

Compile the forward function of the module.

**Args:**

- <b>`q_inputs`</b> (Union\[Tuple\[numpy.ndarray, ...\], numpy.ndarray\]):  Needed for tracing and  building the boundaries.
- <b>`configuration`</b> (Optional\[Configuration\]):  Configuration object to use during compilation
- <b>`compilation_artifacts`</b> (Optional\[DebugArtifacts\]):  Artifacts object to fill during
- <b>`show_mlir`</b> (bool):  if set, the MLIR produced by the converter and which is  going to be sent to the compiler backend is shown on the screen, e.g., for debugging  or demo. Defaults to False.
- <b>`use_virtual_lib`</b> (bool):  set to use the so called virtual lib simulating FHE computation.  Defaults to False.
- <b>`p_error`</b> (Optional\[float\]):  probability of error of a PBS.

**Returns:**

- <b>`Circuit`</b>:  the compiled Circuit.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_module.py#L269"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(qvalues: ndarray) → ndarray
```

Take the last layer q_out and use its dequant function.

**Args:**

- <b>`qvalues`</b> (numpy.ndarray):  Quantized values of the last layer.

**Returns:**

- <b>`numpy.ndarray`</b>:  Dequantized values of the last layer.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_module.py#L156"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(*qvalues: ndarray) → ndarray
```

Forward pass with numpy function only.

**Args:**

- <b>`*qvalues (numpy.ndarray)`</b>:  numpy.array containing the quantized values.

**Returns:**

- <b>`(numpy.ndarray)`</b>:  Predictions of the quantized model

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_module.py#L226"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward_and_dequant`

```python
forward_and_dequant(*q_x: ndarray) → ndarray
```

Forward pass with numpy function only plus dequantization.

**Args:**

- <b>`*q_x (numpy.ndarray)`</b>:  numpy.ndarray containing the quantized input values. Requires the  input dtype to be uint8.

**Returns:**

- <b>`(numpy.ndarray)`</b>:  Predictions of the quantized model

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_module.py#L105"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(qvalues: ndarray) → ndarray
```

Post-processing of the quantized output.

**Args:**

- <b>`qvalues`</b> (numpy.ndarray):  numpy.ndarray containing the quantized input values.

**Returns:**

- <b>`(numpy.ndarray)`</b>:  Predictions of the quantized model

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_module.py#L239"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(*values: ndarray) → Union[ndarray, Tuple[ndarray, ]]
```

Take the inputs in fp32 and quantize it using the learned quantization parameters.

**Args:**

- <b>`*values (numpy.ndarray)`</b>:  Floating point values.

**Returns:**

- <b>`Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]`</b>:  Quantized (numpy.uint32) values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/blob/release/0.4.x/src/concrete/ml/quantization/quantized_module.py#L286"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_inputs_quantization_parameters`

```python
set_inputs_quantization_parameters(*input_q_params: UniformQuantizer)
```

Set the quantization parameters for the module's inputs.

**Args:**

- <b>`*input_q_params (UniformQuantizer)`</b>:  The quantizer(s) for the module.
