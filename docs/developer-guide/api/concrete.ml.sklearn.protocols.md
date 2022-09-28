<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/protocols.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.sklearn.protocols`

Protocols.

Protocols are used to mix type hinting with duck-typing. Indeed we don't always want to have an abstract parent class between all objects. We are more interested in the behavior of such objects. Implementing a Protocol is a way to specify the behavior of objects.

To read more about Protocol please read: https://peps.python.org/pep-0544

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/protocols.py#L26"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `Quantizer`

Quantizer Protocol.

To use to type hint a quantizer.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/protocols.py#L44"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequant`

```python
dequant(X: 'ndarray') → ndarray
```

Dequantize some values.

**Args:**

- <b>`X`</b> (numpy.ndarray):  Values to dequantize

.. # noqa: DAR202

**Returns:**

- <b>`numpy.ndarray`</b>:  Dequantized values

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/protocols.py#L32"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quant`

```python
quant(values: 'ndarray') → ndarray
```

Quantize some values.

**Args:**

- <b>`values`</b> (numpy.ndarray):  Values to quantize

.. # noqa: DAR202

**Returns:**

- <b>`numpy.ndarray`</b>:  The quantized values

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/protocols.py#L58"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ConcreteBaseEstimatorProtocol`

A Concrete Estimator Protocol.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

onnx_model.

.. # noqa: DAR202

Results:  onnx.ModelProto

______________________________________________________________________

#### <kbd>property</kbd> quantize_input

Quantize input function.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/protocols.py#L97"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    X: 'ndarray',
    configuration: 'Optional[Configuration]',
    compilation_artifacts: 'Optional[DebugArtifacts]',
    show_mlir: 'bool',
    use_virtual_lib: 'bool',
    p_error: 'float'
) → Circuit
```

Compiles a model to a FHE Circuit.

**Args:**

- <b>`X`</b> (numpy.ndarray):  the dequantized dataset
- <b>`configuration`</b> (Optional\[Configuration\]):  the options for  compilation
- <b>`compilation_artifacts`</b> (Optional\[DebugArtifacts\]):  artifacts object to fill  during compilation
- <b>`show_mlir`</b> (bool):  whether or not to show MLIR during the compilation
- <b>`use_virtual_lib`</b> (bool):  whether to compile using the virtual library that allows higher  bitwidths
- <b>`p_error`</b> (float):  probability of error of a PBS

.. # noqa: DAR202

**Returns:**

- <b>`Circuit`</b>:  the compiled Circuit.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/protocols.py#L128"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X: 'ndarray', y: 'ndarray', **fit_params) → ConcreteBaseEstimatorProtocol
```

Initialize and fit the module.

**Args:**

- <b>`X `</b>:  training data  By default, you should be able to pass:  * numpy arrays  * torch tensors  * pandas DataFrame or Series
- <b>`y`</b> (numpy.ndarray):  labels associated with training data
- <b>`**fit_params`</b>:  additional parameters that can be used during training

.. # noqa: DAR202

**Returns:**

- <b>`ConcreteBaseEstimatorProtocol`</b>:  the trained estimator

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/protocols.py#L148"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(
    X: 'ndarray',
    y: 'ndarray',
    *args,
    **kwargs
) → Tuple[ConcreteBaseEstimatorProtocol, BaseEstimator]
```

Fit the quantized estimator and return reference estimator.

This function returns both the quantized estimator (itself), but also a wrapper around the non-quantized trained NN. This is useful in order to compare performance between the quantized and fp32 versions of the classifier

**Args:**

- <b>`X `</b>:  training data  By default, you should be able to pass:  * numpy arrays  * torch tensors  * pandas DataFrame or Series
- <b>`y`</b> (numpy.ndarray):  labels associated with training data
- <b>`*args`</b>:  The arguments to pass to the underlying model.
- <b>`**kwargs`</b>:  The keyword arguments to pass to the underlying model.

.. # noqa: DAR202

**Returns:**

- <b>`self`</b>:  self fitted
- <b>`model`</b>:  underlying estimator

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/protocols.py#L85"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: 'ndarray') → ndarray
```

Post-process models predictions.

**Args:**

- <b>`y_preds`</b> (numpy.ndarray):  predicted values by model (clear-quantized)

.. # noqa: DAR202

**Returns:**

- <b>`numpy.ndarray`</b>:  the post-processed predictions

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/protocols.py#L175"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ConcreteBaseClassifierProtocol`

Concrete classifier protocol.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

onnx_model.

.. # noqa: DAR202

Results:  onnx.ModelProto

______________________________________________________________________

#### <kbd>property</kbd> quantize_input

Quantize input function.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/protocols.py#L97"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    X: 'ndarray',
    configuration: 'Optional[Configuration]',
    compilation_artifacts: 'Optional[DebugArtifacts]',
    show_mlir: 'bool',
    use_virtual_lib: 'bool',
    p_error: 'float'
) → Circuit
```

Compiles a model to a FHE Circuit.

**Args:**

- <b>`X`</b> (numpy.ndarray):  the dequantized dataset
- <b>`configuration`</b> (Optional\[Configuration\]):  the options for  compilation
- <b>`compilation_artifacts`</b> (Optional\[DebugArtifacts\]):  artifacts object to fill  during compilation
- <b>`show_mlir`</b> (bool):  whether or not to show MLIR during the compilation
- <b>`use_virtual_lib`</b> (bool):  whether to compile using the virtual library that allows higher  bitwidths
- <b>`p_error`</b> (float):  probability of error of a PBS

.. # noqa: DAR202

**Returns:**

- <b>`Circuit`</b>:  the compiled Circuit.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/protocols.py#L128"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X: 'ndarray', y: 'ndarray', **fit_params) → ConcreteBaseEstimatorProtocol
```

Initialize and fit the module.

**Args:**

- <b>`X `</b>:  training data  By default, you should be able to pass:  * numpy arrays  * torch tensors  * pandas DataFrame or Series
- <b>`y`</b> (numpy.ndarray):  labels associated with training data
- <b>`**fit_params`</b>:  additional parameters that can be used during training

.. # noqa: DAR202

**Returns:**

- <b>`ConcreteBaseEstimatorProtocol`</b>:  the trained estimator

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/protocols.py#L148"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(
    X: 'ndarray',
    y: 'ndarray',
    *args,
    **kwargs
) → Tuple[ConcreteBaseEstimatorProtocol, BaseEstimator]
```

Fit the quantized estimator and return reference estimator.

This function returns both the quantized estimator (itself), but also a wrapper around the non-quantized trained NN. This is useful in order to compare performance between the quantized and fp32 versions of the classifier

**Args:**

- <b>`X `</b>:  training data  By default, you should be able to pass:  * numpy arrays  * torch tensors  * pandas DataFrame or Series
- <b>`y`</b> (numpy.ndarray):  labels associated with training data
- <b>`*args`</b>:  The arguments to pass to the underlying model.
- <b>`**kwargs`</b>:  The keyword arguments to pass to the underlying model.

.. # noqa: DAR202

**Returns:**

- <b>`self`</b>:  self fitted
- <b>`model`</b>:  underlying estimator

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/protocols.py#L85"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: 'ndarray') → ndarray
```

Post-process models predictions.

**Args:**

- <b>`y_preds`</b> (numpy.ndarray):  predicted values by model (clear-quantized)

.. # noqa: DAR202

**Returns:**

- <b>`numpy.ndarray`</b>:  the post-processed predictions

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/protocols.py#L178"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(X: 'ndarray', execute_in_fhe: 'bool') → ndarray
```

Predicts for each sample the class with highest probability.

**Args:**

- <b>`X`</b> (numpy.ndarray):  Features
- <b>`execute_in_fhe`</b> (bool):  Whether the inference should be done in fhe or not.

.. # noqa: DAR202

**Returns:**
numpy.ndarray

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/protocols.py#L192"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict_proba`

```python
predict_proba(X: 'ndarray', execute_in_fhe: 'bool') → ndarray
```

Predicts for each sample the probability of each class.

**Args:**

- <b>`X`</b> (numpy.ndarray):  Features
- <b>`execute_in_fhe`</b> (bool):  Whether the inference should be done in fhe or not.

.. # noqa: DAR202

**Returns:**
numpy.ndarray

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/protocols.py#L207"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ConcreteBaseRegressorProtocol`

Concrete regressor protocol.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

onnx_model.

.. # noqa: DAR202

Results:  onnx.ModelProto

______________________________________________________________________

#### <kbd>property</kbd> quantize_input

Quantize input function.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/protocols.py#L97"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    X: 'ndarray',
    configuration: 'Optional[Configuration]',
    compilation_artifacts: 'Optional[DebugArtifacts]',
    show_mlir: 'bool',
    use_virtual_lib: 'bool',
    p_error: 'float'
) → Circuit
```

Compiles a model to a FHE Circuit.

**Args:**

- <b>`X`</b> (numpy.ndarray):  the dequantized dataset
- <b>`configuration`</b> (Optional\[Configuration\]):  the options for  compilation
- <b>`compilation_artifacts`</b> (Optional\[DebugArtifacts\]):  artifacts object to fill  during compilation
- <b>`show_mlir`</b> (bool):  whether or not to show MLIR during the compilation
- <b>`use_virtual_lib`</b> (bool):  whether to compile using the virtual library that allows higher  bitwidths
- <b>`p_error`</b> (float):  probability of error of a PBS

.. # noqa: DAR202

**Returns:**

- <b>`Circuit`</b>:  the compiled Circuit.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/protocols.py#L128"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X: 'ndarray', y: 'ndarray', **fit_params) → ConcreteBaseEstimatorProtocol
```

Initialize and fit the module.

**Args:**

- <b>`X `</b>:  training data  By default, you should be able to pass:  * numpy arrays  * torch tensors  * pandas DataFrame or Series
- <b>`y`</b> (numpy.ndarray):  labels associated with training data
- <b>`**fit_params`</b>:  additional parameters that can be used during training

.. # noqa: DAR202

**Returns:**

- <b>`ConcreteBaseEstimatorProtocol`</b>:  the trained estimator

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/protocols.py#L148"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(
    X: 'ndarray',
    y: 'ndarray',
    *args,
    **kwargs
) → Tuple[ConcreteBaseEstimatorProtocol, BaseEstimator]
```

Fit the quantized estimator and return reference estimator.

This function returns both the quantized estimator (itself), but also a wrapper around the non-quantized trained NN. This is useful in order to compare performance between the quantized and fp32 versions of the classifier

**Args:**

- <b>`X `</b>:  training data  By default, you should be able to pass:  * numpy arrays  * torch tensors  * pandas DataFrame or Series
- <b>`y`</b> (numpy.ndarray):  labels associated with training data
- <b>`*args`</b>:  The arguments to pass to the underlying model.
- <b>`**kwargs`</b>:  The keyword arguments to pass to the underlying model.

.. # noqa: DAR202

**Returns:**

- <b>`self`</b>:  self fitted
- <b>`model`</b>:  underlying estimator

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/protocols.py#L85"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: 'ndarray') → ndarray
```

Post-process models predictions.

**Args:**

- <b>`y_preds`</b> (numpy.ndarray):  predicted values by model (clear-quantized)

.. # noqa: DAR202

**Returns:**

- <b>`numpy.ndarray`</b>:  the post-processed predictions

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/protocols.py#L210"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(X: 'ndarray', execute_in_fhe: 'bool') → ndarray
```

Predicts for each sample the expected value.

**Args:**

- <b>`X`</b> (numpy.ndarray):  Features
- <b>`execute_in_fhe`</b> (bool):  Whether the inference should be done in fhe or not.

.. # noqa: DAR202

**Returns:**
numpy.ndarray
