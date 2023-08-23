<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/sklearn/base.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.sklearn.base`

Base classes for all estimators.

## **Global Variables**

- **USE_OLD_VL**
- **OPSET_VERSION_FOR_ONNX_EXPORT**
- **QNN_AUTO_KWARGS**

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L91"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BaseEstimator`

Base class for all estimators in Concrete ML.

This class does not inherit from sklearn.base.BaseEstimator as it creates some conflicts with skorch in QuantizedTorchEstimatorMixin's subclasses (more specifically, the `get_params` method is not properly inherited).

**Attributes:**

- <b>`_is_a_public_cml_model`</b> (bool):  Private attribute indicating if the class is a public model  (as opposed to base or mixin classes).

<a href="../../../src/concrete/ml/sklearn/base.py#L108"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```

Initialize the base class with common attributes used in all estimators.

An underscore "\_" is appended to attributes that were created while fitting the model. This is done in order to follow scikit-Learn's standard format. More information available in their documentation: https://scikit-learn.org/stable/developers/develop.html#:~:text=Estimated%20Attributes%C2%B6

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete documentation (https://docs.zama.ai/concrete/getting-started/terminology_and_structure) Is None if the model is not fitted.

**Returns:**

- <b>`Circuit`</b>:  The FHE circuit.

______________________________________________________________________

#### <kbd>property</kbd> is_compiled

Indicate if the model is compiled.

**Returns:**

- <b>`bool`</b>:  If the model is compiled.

______________________________________________________________________

#### <kbd>property</kbd> is_fitted

Indicate if the model is fitted.

**Returns:**

- <b>`bool`</b>:  If the model is fitted.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

Is None if the model is not fitted.

**Returns:**

- <b>`onnx.ModelProto`</b>:  The ONNX model.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L309"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_compiled`

```python
check_model_is_compiled() → None
```

Check if the model is compiled.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not compiled.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L285"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_fitted`

```python
check_model_is_fitted() → None
```

Check if the model is fitted.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not fitted.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L484"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    X: 'Data',
    configuration: 'Optional[Configuration]' = None,
    artifacts: 'Optional[DebugArtifacts]' = None,
    show_mlir: 'bool' = False,
    p_error: 'Optional[float]' = None,
    global_p_error: 'Optional[float]' = None,
    verbose: 'bool' = False
) → Circuit
```

Compile the model.

**Args:**

- <b>`X`</b> (Data):  A representative set of input values used for building cryptographic  parameters, as a Numpy array, Torch tensor, Pandas DataFrame or List. This is  usually the training data-set or s sub-set of it.
- <b>`configuration`</b> (Optional\[Configuration\]):  Options to use for compilation. Default  to None.
- <b>`artifacts`</b> (Optional\[DebugArtifacts\]):  Artifacts information about the compilation  process to store for debugging. Default to None.
- <b>`show_mlir`</b> (bool):  Indicate if the MLIR graph should be printed during compilation.  Default to False.
- <b>`p_error`</b> (Optional\[float\]):  Probability of error of a single PBS. A p_error value cannot  be given if a global_p_error value is already set. Default to None, which sets this  error to a default value.
- <b>`global_p_error`</b> (Optional\[float\]):  Probability of error of the full circuit. A  global_p_error value cannot be given if a p_error value is already set. This feature  is not supported during the FHE simulation mode, meaning the probability is  currently set to 0. Default to None, which sets this error to a default value.
- <b>`verbose`</b> (bool):  Indicate if compilation information should be printed  during compilation. Default to False.

**Returns:**

- <b>`Circuit`</b>:  The compiled Circuit.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L463"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(q_y_preds: 'ndarray') → ndarray
```

De-quantize the output.

This step ensures that the fit method has been called.

**Args:**

- <b>`q_y_preds`</b> (numpy.ndarray):  The quantized output values to de-quantize.

**Returns:**

- <b>`numpy.ndarray`</b>:  The de-quantized output values.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L220"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump`

```python
dump(file: 'TextIO') → None
```

Dump itself to a file.

**Args:**

- <b>`file`</b> (TextIO):  The file to dump the serialized object into.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L192"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict[str, Any]
```

Dump the object as a dict.

**Returns:**

- <b>`Dict[str, Any]`</b>:  Dict of serialized objects.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L212"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dumps`

```python
dumps() → str
```

Dump itself to a string.

**Returns:**

- <b>`metadata`</b> (str):  String of the serialized object.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L376"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X: 'Data', y: 'Target', **fit_parameters)
```

Fit the estimator.

This method trains a scikit-learn estimator, computes its ONNX graph and defines the quantization parameters needed for proper FHE inference.

**Args:**

- <b>`X`</b> (Data):  The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
- <b>`y`</b> (Target):  The target data, as a Numpy array, Torch tensor, Pandas DataFrame, Pandas  Series or List.
- <b>`**fit_parameters`</b>:  Keyword arguments to pass to the float estimator's fit method.

**Returns:**
The fitted estimator.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L397"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(
    X: 'Data',
    y: 'Target',
    random_state: 'Optional[int]' = None,
    **fit_parameters
)
```

Fit both the Concrete ML and its equivalent float estimators.

**Args:**

- <b>`X`</b> (Data):  The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
- <b>`y`</b> (Target):  The target data, as a Numpy array, Torch tensor, Pandas DataFrame, Pandas  Series or List.
- <b>`random_state`</b> (Optional\[int\]):  The random state to use when fitting. Defaults to None.
- <b>`**fit_parameters`</b>:  Keyword arguments to pass to the float estimator's fit method.

**Returns:**
The Concrete ML and float equivalent fitted estimators.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L318"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep: 'bool' = True) → dict
```

Get parameters for this estimator.

This method is used to instantiate a scikit-learn model using the Concrete ML model's parameters. It does not override scikit-learn's existing `get_params` method in order to not break its implementation of `set_params`.

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and contained  subobjects that are estimators. Default to True.

**Returns:**

- <b>`params`</b> (dict):  Parameter names mapped to their values.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: 'Dict[str, Any]') → BaseEstimator
```

Load itself from a dict.

**Args:**

- <b>`metadata`</b> (Dict\[str, Any\]):  Dict of serialized objects.

**Returns:**

- <b>`BaseEstimator`</b>:  The loaded object.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L670"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: 'ndarray') → ndarray
```

Apply post-processing to the de-quantized predictions.

This post-processing step can include operations such as applying the sigmoid or softmax function for classifiers, or summing an ensemble's outputs. These steps are done in the clear because of current technical constraints. They most likely will be integrated in the FHE computations in the future.

For some simple models such a linear regression, there is no post-processing step but the method is kept to make the API consistent for the client-server API. Other models might need to use attributes stored in `post_processing_params`.

**Args:**

- <b>`y_preds`</b> (numpy.ndarray):  The de-quantized predictions to post-process.

**Returns:**

- <b>`numpy.ndarray`</b>:  The post-processed predictions.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L586"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(
    X: 'Data',
    fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>
) → ndarray
```

Predict values for X, in FHE or in the clear.

**Args:**

- <b>`X`</b> (Data):  The input values to predict, as a Numpy array, Torch tensor, Pandas DataFrame  or List.
- <b>`fhe`</b> (Union\[FheMode, str\]):  The mode to use for prediction.  Can be FheMode.DISABLE for Concrete ML Python inference,  FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.  Can also be the string representation of any of these values.  Default to FheMode.DISABLE.

**Returns:**

- <b>`np.ndarray`</b>:  The predicted values for X.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L450"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(X: 'ndarray') → ndarray
```

Quantize the input.

This step ensures that the fit method has been called.

**Args:**

- <b>`X`</b> (numpy.ndarray):  The input values to quantize.

**Returns:**

- <b>`numpy.ndarray`</b>:  The quantized input values.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L694"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BaseClassifier`

Base class for linear and tree-based classifiers in Concrete ML.

This class inherits from BaseEstimator and modifies some of its methods in order to align them with classifier behaviors. This notably include applying a sigmoid/softmax post-processing to the predicted values as well as handling a mapping of classes in case they are not ordered.

<a href="../../../src/concrete/ml/sklearn/base.py#L702"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*args, **kwargs)
```

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete documentation (https://docs.zama.ai/concrete/getting-started/terminology_and_structure) Is None if the model is not fitted.

**Returns:**

- <b>`Circuit`</b>:  The FHE circuit.

______________________________________________________________________

#### <kbd>property</kbd> is_compiled

Indicate if the model is compiled.

**Returns:**

- <b>`bool`</b>:  If the model is compiled.

______________________________________________________________________

#### <kbd>property</kbd> is_fitted

Indicate if the model is fitted.

**Returns:**

- <b>`bool`</b>:  If the model is fitted.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

Is None if the model is not fitted.

**Returns:**

- <b>`onnx.ModelProto`</b>:  The ONNX model.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L309"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_compiled`

```python
check_model_is_compiled() → None
```

Check if the model is compiled.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not compiled.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L285"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_fitted`

```python
check_model_is_fitted() → None
```

Check if the model is fitted.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not fitted.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L484"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    X: 'Data',
    configuration: 'Optional[Configuration]' = None,
    artifacts: 'Optional[DebugArtifacts]' = None,
    show_mlir: 'bool' = False,
    p_error: 'Optional[float]' = None,
    global_p_error: 'Optional[float]' = None,
    verbose: 'bool' = False
) → Circuit
```

Compile the model.

**Args:**

- <b>`X`</b> (Data):  A representative set of input values used for building cryptographic  parameters, as a Numpy array, Torch tensor, Pandas DataFrame or List. This is  usually the training data-set or s sub-set of it.
- <b>`configuration`</b> (Optional\[Configuration\]):  Options to use for compilation. Default  to None.
- <b>`artifacts`</b> (Optional\[DebugArtifacts\]):  Artifacts information about the compilation  process to store for debugging. Default to None.
- <b>`show_mlir`</b> (bool):  Indicate if the MLIR graph should be printed during compilation.  Default to False.
- <b>`p_error`</b> (Optional\[float\]):  Probability of error of a single PBS. A p_error value cannot  be given if a global_p_error value is already set. Default to None, which sets this  error to a default value.
- <b>`global_p_error`</b> (Optional\[float\]):  Probability of error of the full circuit. A  global_p_error value cannot be given if a p_error value is already set. This feature  is not supported during the FHE simulation mode, meaning the probability is  currently set to 0. Default to None, which sets this error to a default value.
- <b>`verbose`</b> (bool):  Indicate if compilation information should be printed  during compilation. Default to False.

**Returns:**

- <b>`Circuit`</b>:  The compiled Circuit.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L463"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(q_y_preds: 'ndarray') → ndarray
```

De-quantize the output.

This step ensures that the fit method has been called.

**Args:**

- <b>`q_y_preds`</b> (numpy.ndarray):  The quantized output values to de-quantize.

**Returns:**

- <b>`numpy.ndarray`</b>:  The de-quantized output values.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L220"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump`

```python
dump(file: 'TextIO') → None
```

Dump itself to a file.

**Args:**

- <b>`file`</b> (TextIO):  The file to dump the serialized object into.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L192"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict[str, Any]
```

Dump the object as a dict.

**Returns:**

- <b>`Dict[str, Any]`</b>:  Dict of serialized objects.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L212"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dumps`

```python
dumps() → str
```

Dump itself to a string.

**Returns:**

- <b>`metadata`</b> (str):  String of the serialized object.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L715"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X: 'Data', y: 'Target', **fit_parameters)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L397"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(
    X: 'Data',
    y: 'Target',
    random_state: 'Optional[int]' = None,
    **fit_parameters
)
```

Fit both the Concrete ML and its equivalent float estimators.

**Args:**

- <b>`X`</b> (Data):  The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
- <b>`y`</b> (Target):  The target data, as a Numpy array, Torch tensor, Pandas DataFrame, Pandas  Series or List.
- <b>`random_state`</b> (Optional\[int\]):  The random state to use when fitting. Defaults to None.
- <b>`**fit_parameters`</b>:  Keyword arguments to pass to the float estimator's fit method.

**Returns:**
The Concrete ML and float equivalent fitted estimators.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L318"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep: 'bool' = True) → dict
```

Get parameters for this estimator.

This method is used to instantiate a scikit-learn model using the Concrete ML model's parameters. It does not override scikit-learn's existing `get_params` method in order to not break its implementation of `set_params`.

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and contained  subobjects that are estimators. Default to True.

**Returns:**

- <b>`params`</b> (dict):  Parameter names mapped to their values.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: 'Dict[str, Any]') → BaseEstimator
```

Load itself from a dict.

**Args:**

- <b>`metadata`</b> (Dict\[str, Any\]):  Dict of serialized objects.

**Returns:**

- <b>`BaseEstimator`</b>:  The loaded object.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L760"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L749"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(
    X: 'Data',
    fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>
) → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L732"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict_proba`

```python
predict_proba(
    X: 'Data',
    fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>
) → ndarray
```

Predict class probabilities.

**Args:**

- <b>`X`</b> (Data):  The input values to predict, as a Numpy array, Torch tensor, Pandas DataFrame  or List.
- <b>`fhe`</b> (Union\[FheMode, str\]):  The mode to use for prediction.  Can be FheMode.DISABLE for Concrete ML Python inference,  FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.  Can also be the string representation of any of these values.  Default to FheMode.DISABLE.

**Returns:**

- <b>`numpy.ndarray`</b>:  The predicted class probabilities.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L450"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(X: 'ndarray') → ndarray
```

Quantize the input.

This step ensures that the fit method has been called.

**Args:**

- <b>`X`</b> (numpy.ndarray):  The input values to quantize.

**Returns:**

- <b>`numpy.ndarray`</b>:  The quantized input values.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L789"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedTorchEstimatorMixin`

Mixin that provides quantization for a torch module and follows the Estimator API.

<a href="../../../src/concrete/ml/sklearn/base.py#L799"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```

______________________________________________________________________

#### <kbd>property</kbd> base_module

Get the Torch module.

**Returns:**

- <b>`SparseQuantNeuralNetwork`</b>:  The fitted underlying module.

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

______________________________________________________________________

#### <kbd>property</kbd> input_quantizers

Get the input quantizers.

**Returns:**

- <b>`List[UniformQuantizer]`</b>:  The input quantizers.

______________________________________________________________________

#### <kbd>property</kbd> is_compiled

Indicate if the model is compiled.

**Returns:**

- <b>`bool`</b>:  If the model is compiled.

______________________________________________________________________

#### <kbd>property</kbd> is_fitted

Indicate if the model is fitted.

**Returns:**

- <b>`bool`</b>:  If the model is fitted.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

Is None if the model is not fitted.

**Returns:**

- <b>`onnx.ModelProto`</b>:  The ONNX model.

______________________________________________________________________

#### <kbd>property</kbd> output_quantizers

Get the output quantizers.

**Returns:**

- <b>`List[UniformQuantizer]`</b>:  The output quantizers.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L309"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_compiled`

```python
check_model_is_compiled() → None
```

Check if the model is compiled.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not compiled.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L285"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_fitted`

```python
check_model_is_fitted() → None
```

Check if the model is fitted.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not fitted.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1110"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    X: 'Data',
    configuration: 'Optional[Configuration]' = None,
    artifacts: 'Optional[DebugArtifacts]' = None,
    show_mlir: 'bool' = False,
    p_error: 'Optional[float]' = None,
    global_p_error: 'Optional[float]' = None,
    verbose: 'bool' = False
) → Circuit
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1103"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(q_y_preds: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L220"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump`

```python
dump(file: 'TextIO') → None
```

Dump itself to a file.

**Args:**

- <b>`file`</b> (TextIO):  The file to dump the serialized object into.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L192"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict[str, Any]
```

Dump the object as a dict.

**Returns:**

- <b>`Dict[str, Any]`</b>:  Dict of serialized objects.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L212"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dumps`

```python
dumps() → str
```

Dump itself to a string.

**Returns:**

- <b>`metadata`</b> (str):  String of the serialized object.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L908"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X: 'Data', y: 'Target', **fit_parameters)
```

Fit he estimator.

If the module was already initialized, the module will be re-initialized unless `warm_start` is set to True. In addition to the torch training step, this method performs quantization of the trained Torch model using Quantization Aware Training (QAT).

Values of dtype float64 are not supported and will be casted to float32.

**Args:**

- <b>`X`</b> (Data):  The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
- <b>`y`</b> (Target):  The target data,  as a Numpy array, Torch tensor, Pandas DataFrame, Pandas  Series or List.
- <b>`**fit_parameters`</b>:  Keyword arguments to pass to skorch's fit method.

**Returns:**
The fitted estimator.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1054"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(
    X: 'Data',
    y: 'Target',
    random_state: 'Optional[int]' = None,
    **fit_parameters
)
```

Fit the quantized estimator as well as its equivalent float estimator.

This function returns both the quantized estimator (itself) as well as its non-quantized (float) equivalent, which are both trained separately. This method differs from the BaseEstimator's `fit_benchmark` method as QNNs use QAT instead of PTQ. Hence, here, the float model is topologically equivalent as we have less control over the influence of QAT over the weights.

Values of dtype float64 are not supported and will be casted to float32.

**Args:**

- <b>`X`</b> (Data):  The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
- <b>`y`</b> (Target):  The target data,  as a Numpy array, Torch tensor, Pandas DataFrame Pandas  Series or List.
- <b>`random_state`</b> (Optional\[int\]):  The random state to use when fitting. However, skorch  does not handle such a parameter and setting it will have no effect. Defaults  to None.
- <b>`**fit_parameters`</b>:  Keyword arguments to pass to skorch's fit method.

**Returns:**
The Concrete ML and equivalent skorch fitted estimators.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L857"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_params`

```python
get_params(deep: 'bool' = True) → dict
```

Get parameters for this estimator.

This method is overloaded in order to make sure that auto-computed parameters are not considered when cloning the model (e.g during a GridSearchCV call).

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and  contained subobjects that are estimators.

**Returns:**

- <b>`params`</b> (dict):  Parameter names mapped to their values.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L888"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep: 'bool' = True) → Dict
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: 'Dict[str, Any]') → BaseEstimator
```

Load itself from a dict.

**Args:**

- <b>`metadata`</b> (Dict\[str, Any\]):  Dict of serialized objects.

**Returns:**

- <b>`BaseEstimator`</b>:  The loaded object.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1160"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L586"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(
    X: 'Data',
    fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>
) → ndarray
```

Predict values for X, in FHE or in the clear.

**Args:**

- <b>`X`</b> (Data):  The input values to predict, as a Numpy array, Torch tensor, Pandas DataFrame  or List.
- <b>`fhe`</b> (Union\[FheMode, str\]):  The mode to use for prediction.  Can be FheMode.DISABLE for Concrete ML Python inference,  FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.  Can also be the string representation of any of these values.  Default to FheMode.DISABLE.

**Returns:**

- <b>`np.ndarray`</b>:  The predicted values for X.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1164"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `prune`

```python
prune(X: 'Data', y: 'Target', n_prune_neurons_percentage: 'float', **fit_params)
```

Prune a copy of this Neural Network model.

This can be used when the number of neurons on the hidden layers is too high. For example, when creating a Neural Network model with `n_hidden_neurons_multiplier` high (3-4), it can be used to speed up the model inference in FHE. Many times, up to 50% of neurons can be pruned without losing accuracy, when using this function to fine-tune an already trained model with good accuracy. This method should be used once good accuracy is obtained.

**Args:**

- <b>`X`</b> (Data):  The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
- <b>`y`</b> (Target):  The target data,  as a Numpy array, Torch tensor, Pandas DataFrame Pandas  Series or List.
- <b>`n_prune_neurons_percentage`</b> (float):  The percentage of neurons to remove. A value of  0 (resp. 1.0) means no (resp. all) neurons will be removed.
- <b>`fit_params`</b>:  Additional parameters to pass to the underlying nn.Module's forward method.

**Returns:**
A new pruned copy of the Neural Network model.

**Raises:**

- <b>`ValueError`</b>:  If the model has not been trained or has already been pruned.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1095"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(X: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1237"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BaseTreeEstimatorMixin`

Mixin class for tree-based estimators.

This class inherits from sklearn.base.BaseEstimator in order to have access to scikit-learn's `get_params` and `set_params` methods.

<a href="../../../src/concrete/ml/sklearn/base.py#L1254"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(n_bits: 'int')
```

Initialize the TreeBasedEstimatorMixin.

**Args:**

- <b>`n_bits`</b> (int):  The number of bits used for quantization.

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete documentation (https://docs.zama.ai/concrete/getting-started/terminology_and_structure) Is None if the model is not fitted.

**Returns:**

- <b>`Circuit`</b>:  The FHE circuit.

______________________________________________________________________

#### <kbd>property</kbd> is_compiled

Indicate if the model is compiled.

**Returns:**

- <b>`bool`</b>:  If the model is compiled.

______________________________________________________________________

#### <kbd>property</kbd> is_fitted

Indicate if the model is fitted.

**Returns:**

- <b>`bool`</b>:  If the model is fitted.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

Is None if the model is not fitted.

**Returns:**

- <b>`onnx.ModelProto`</b>:  The ONNX model.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L309"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_compiled`

```python
check_model_is_compiled() → None
```

Check if the model is compiled.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not compiled.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L285"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_fitted`

```python
check_model_is_fitted() → None
```

Check if the model is fitted.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not fitted.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1338"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(*args, **kwargs) → Circuit
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1316"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(q_y_preds: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L220"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump`

```python
dump(file: 'TextIO') → None
```

Dump itself to a file.

**Args:**

- <b>`file`</b> (TextIO):  The file to dump the serialized object into.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L192"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict[str, Any]
```

Dump the object as a dict.

**Returns:**

- <b>`Dict[str, Any]`</b>:  Dict of serialized objects.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L212"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dumps`

```python
dumps() → str
```

Dump itself to a string.

**Returns:**

- <b>`metadata`</b> (str):  String of the serialized object.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1267"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X: 'Data', y: 'Target', **fit_parameters)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L397"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(
    X: 'Data',
    y: 'Target',
    random_state: 'Optional[int]' = None,
    **fit_parameters
)
```

Fit both the Concrete ML and its equivalent float estimators.

**Args:**

- <b>`X`</b> (Data):  The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
- <b>`y`</b> (Target):  The target data, as a Numpy array, Torch tensor, Pandas DataFrame, Pandas  Series or List.
- <b>`random_state`</b> (Optional\[int\]):  The random state to use when fitting. Defaults to None.
- <b>`**fit_parameters`</b>:  Keyword arguments to pass to the float estimator's fit method.

**Returns:**
The Concrete ML and float equivalent fitted estimators.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L318"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep: 'bool' = True) → dict
```

Get parameters for this estimator.

This method is used to instantiate a scikit-learn model using the Concrete ML model's parameters. It does not override scikit-learn's existing `get_params` method in order to not break its implementation of `set_params`.

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and contained  subobjects that are estimators. Default to True.

**Returns:**

- <b>`params`</b> (dict):  Parameter names mapped to their values.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: 'Dict[str, Any]') → BaseEstimator
```

Load itself from a dict.

**Args:**

- <b>`metadata`</b> (Dict\[str, Any\]):  Dict of serialized objects.

**Returns:**

- <b>`BaseEstimator`</b>:  The loaded object.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1389"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1384"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(
    X: 'Data',
    fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>
) → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1304"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(X: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1399"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BaseTreeRegressorMixin`

Mixin class for tree-based regressors.

This class is used to create a tree-based regressor class that inherits from sklearn.base.RegressorMixin, which essentially gives access to scikit-learn's `score` method for regressors.

<a href="../../../src/concrete/ml/sklearn/base.py#L1254"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(n_bits: 'int')
```

Initialize the TreeBasedEstimatorMixin.

**Args:**

- <b>`n_bits`</b> (int):  The number of bits used for quantization.

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete documentation (https://docs.zama.ai/concrete/getting-started/terminology_and_structure) Is None if the model is not fitted.

**Returns:**

- <b>`Circuit`</b>:  The FHE circuit.

______________________________________________________________________

#### <kbd>property</kbd> is_compiled

Indicate if the model is compiled.

**Returns:**

- <b>`bool`</b>:  If the model is compiled.

______________________________________________________________________

#### <kbd>property</kbd> is_fitted

Indicate if the model is fitted.

**Returns:**

- <b>`bool`</b>:  If the model is fitted.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

Is None if the model is not fitted.

**Returns:**

- <b>`onnx.ModelProto`</b>:  The ONNX model.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L309"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_compiled`

```python
check_model_is_compiled() → None
```

Check if the model is compiled.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not compiled.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L285"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_fitted`

```python
check_model_is_fitted() → None
```

Check if the model is fitted.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not fitted.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1338"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(*args, **kwargs) → Circuit
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1316"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(q_y_preds: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L220"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump`

```python
dump(file: 'TextIO') → None
```

Dump itself to a file.

**Args:**

- <b>`file`</b> (TextIO):  The file to dump the serialized object into.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L192"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict[str, Any]
```

Dump the object as a dict.

**Returns:**

- <b>`Dict[str, Any]`</b>:  Dict of serialized objects.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L212"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dumps`

```python
dumps() → str
```

Dump itself to a string.

**Returns:**

- <b>`metadata`</b> (str):  String of the serialized object.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1267"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X: 'Data', y: 'Target', **fit_parameters)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L397"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(
    X: 'Data',
    y: 'Target',
    random_state: 'Optional[int]' = None,
    **fit_parameters
)
```

Fit both the Concrete ML and its equivalent float estimators.

**Args:**

- <b>`X`</b> (Data):  The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
- <b>`y`</b> (Target):  The target data, as a Numpy array, Torch tensor, Pandas DataFrame, Pandas  Series or List.
- <b>`random_state`</b> (Optional\[int\]):  The random state to use when fitting. Defaults to None.
- <b>`**fit_parameters`</b>:  Keyword arguments to pass to the float estimator's fit method.

**Returns:**
The Concrete ML and float equivalent fitted estimators.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L318"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep: 'bool' = True) → dict
```

Get parameters for this estimator.

This method is used to instantiate a scikit-learn model using the Concrete ML model's parameters. It does not override scikit-learn's existing `get_params` method in order to not break its implementation of `set_params`.

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and contained  subobjects that are estimators. Default to True.

**Returns:**

- <b>`params`</b> (dict):  Parameter names mapped to their values.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: 'Dict[str, Any]') → BaseEstimator
```

Load itself from a dict.

**Args:**

- <b>`metadata`</b> (Dict\[str, Any\]):  Dict of serialized objects.

**Returns:**

- <b>`BaseEstimator`</b>:  The loaded object.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1389"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1384"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(
    X: 'Data',
    fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>
) → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1304"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(X: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1408"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BaseTreeClassifierMixin`

Mixin class for tree-based classifiers.

This class is used to create a tree-based classifier class that inherits from sklearn.base.ClassifierMixin, which essentially gives access to scikit-learn's `score` method for classifiers.

Additionally, this class adjusts some of the tree-based base class's methods in order to make them compliant with classification workflows.

<a href="../../../src/concrete/ml/sklearn/base.py#L702"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*args, **kwargs)
```

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete documentation (https://docs.zama.ai/concrete/getting-started/terminology_and_structure) Is None if the model is not fitted.

**Returns:**

- <b>`Circuit`</b>:  The FHE circuit.

______________________________________________________________________

#### <kbd>property</kbd> is_compiled

Indicate if the model is compiled.

**Returns:**

- <b>`bool`</b>:  If the model is compiled.

______________________________________________________________________

#### <kbd>property</kbd> is_fitted

Indicate if the model is fitted.

**Returns:**

- <b>`bool`</b>:  If the model is fitted.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

Is None if the model is not fitted.

**Returns:**

- <b>`onnx.ModelProto`</b>:  The ONNX model.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L309"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_compiled`

```python
check_model_is_compiled() → None
```

Check if the model is compiled.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not compiled.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L285"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_fitted`

```python
check_model_is_fitted() → None
```

Check if the model is fitted.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not fitted.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1338"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(*args, **kwargs) → Circuit
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1316"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(q_y_preds: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L220"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump`

```python
dump(file: 'TextIO') → None
```

Dump itself to a file.

**Args:**

- <b>`file`</b> (TextIO):  The file to dump the serialized object into.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L192"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict[str, Any]
```

Dump the object as a dict.

**Returns:**

- <b>`Dict[str, Any]`</b>:  Dict of serialized objects.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L212"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dumps`

```python
dumps() → str
```

Dump itself to a string.

**Returns:**

- <b>`metadata`</b> (str):  String of the serialized object.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L715"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X: 'Data', y: 'Target', **fit_parameters)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L397"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(
    X: 'Data',
    y: 'Target',
    random_state: 'Optional[int]' = None,
    **fit_parameters
)
```

Fit both the Concrete ML and its equivalent float estimators.

**Args:**

- <b>`X`</b> (Data):  The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
- <b>`y`</b> (Target):  The target data, as a Numpy array, Torch tensor, Pandas DataFrame, Pandas  Series or List.
- <b>`random_state`</b> (Optional\[int\]):  The random state to use when fitting. Defaults to None.
- <b>`**fit_parameters`</b>:  Keyword arguments to pass to the float estimator's fit method.

**Returns:**
The Concrete ML and float equivalent fitted estimators.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L318"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep: 'bool' = True) → dict
```

Get parameters for this estimator.

This method is used to instantiate a scikit-learn model using the Concrete ML model's parameters. It does not override scikit-learn's existing `get_params` method in order to not break its implementation of `set_params`.

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and contained  subobjects that are estimators. Default to True.

**Returns:**

- <b>`params`</b> (dict):  Parameter names mapped to their values.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: 'Dict[str, Any]') → BaseEstimator
```

Load itself from a dict.

**Args:**

- <b>`metadata`</b> (Dict\[str, Any\]):  Dict of serialized objects.

**Returns:**

- <b>`BaseEstimator`</b>:  The loaded object.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L760"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L749"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(
    X: 'Data',
    fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>
) → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L732"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict_proba`

```python
predict_proba(
    X: 'Data',
    fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>
) → ndarray
```

Predict class probabilities.

**Args:**

- <b>`X`</b> (Data):  The input values to predict, as a Numpy array, Torch tensor, Pandas DataFrame  or List.
- <b>`fhe`</b> (Union\[FheMode, str\]):  The mode to use for prediction.  Can be FheMode.DISABLE for Concrete ML Python inference,  FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.  Can also be the string representation of any of these values.  Default to FheMode.DISABLE.

**Returns:**

- <b>`numpy.ndarray`</b>:  The predicted class probabilities.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1304"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(X: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1423"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SklearnLinearModelMixin`

A Mixin class for sklearn linear models with FHE.

This class inherits from sklearn.base.BaseEstimator in order to have access to scikit-learn's `get_params` and `set_params` methods.

<a href="../../../src/concrete/ml/sklearn/base.py#L1437"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(n_bits: 'Union[int, Dict[str, int]]' = 8)
```

Initialize the FHE linear model.

**Args:**

- <b>`n_bits`</b> (int, Dict\[str, int\]):  Number of bits to quantize the model. If an int is passed  for n_bits, the value will be used for quantizing inputs and weights. If a dict is  passed, then it should contain "op_inputs" and "op_weights" as keys with  corresponding number of quantization bits so that:
  \- op_inputs : number of bits to quantize the input values
  \- op_weights: number of bits to quantize the learned parameters  Default to 8.

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete documentation (https://docs.zama.ai/concrete/getting-started/terminology_and_structure) Is None if the model is not fitted.

**Returns:**

- <b>`Circuit`</b>:  The FHE circuit.

______________________________________________________________________

#### <kbd>property</kbd> is_compiled

Indicate if the model is compiled.

**Returns:**

- <b>`bool`</b>:  If the model is compiled.

______________________________________________________________________

#### <kbd>property</kbd> is_fitted

Indicate if the model is fitted.

**Returns:**

- <b>`bool`</b>:  If the model is fitted.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

Is None if the model is not fitted.

**Returns:**

- <b>`onnx.ModelProto`</b>:  The ONNX model.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L309"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_compiled`

```python
check_model_is_compiled() → None
```

Check if the model is compiled.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not compiled.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L285"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_fitted`

```python
check_model_is_fitted() → None
```

Check if the model is fitted.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not fitted.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1617"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(*args, **kwargs) → Circuit
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1576"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(q_y_preds: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L220"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump`

```python
dump(file: 'TextIO') → None
```

Dump itself to a file.

**Args:**

- <b>`file`</b> (TextIO):  The file to dump the serialized object into.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L192"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict[str, Any]
```

Dump the object as a dict.

**Returns:**

- <b>`Dict[str, Any]`</b>:  Dict of serialized objects.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L212"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dumps`

```python
dumps() → str
```

Dump itself to a string.

**Returns:**

- <b>`metadata`</b> (str):  String of the serialized object.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1487"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X: 'Data', y: 'Target', **fit_parameters)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L397"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(
    X: 'Data',
    y: 'Target',
    random_state: 'Optional[int]' = None,
    **fit_parameters
)
```

Fit both the Concrete ML and its equivalent float estimators.

**Args:**

- <b>`X`</b> (Data):  The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
- <b>`y`</b> (Target):  The target data, as a Numpy array, Torch tensor, Pandas DataFrame, Pandas  Series or List.
- <b>`random_state`</b> (Optional\[int\]):  The random state to use when fitting. Defaults to None.
- <b>`**fit_parameters`</b>:  Keyword arguments to pass to the float estimator's fit method.

**Returns:**
The Concrete ML and float equivalent fitted estimators.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L318"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep: 'bool' = True) → dict
```

Get parameters for this estimator.

This method is used to instantiate a scikit-learn model using the Concrete ML model's parameters. It does not override scikit-learn's existing `get_params` method in order to not break its implementation of `set_params`.

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and contained  subobjects that are estimators. Default to True.

**Returns:**

- <b>`params`</b> (dict):  Parameter names mapped to their values.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: 'Dict[str, Any]') → BaseEstimator
```

Load itself from a dict.

**Args:**

- <b>`metadata`</b> (Dict\[str, Any\]):  Dict of serialized objects.

**Returns:**

- <b>`BaseEstimator`</b>:  The loaded object.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L670"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: 'ndarray') → ndarray
```

Apply post-processing to the de-quantized predictions.

This post-processing step can include operations such as applying the sigmoid or softmax function for classifiers, or summing an ensemble's outputs. These steps are done in the clear because of current technical constraints. They most likely will be integrated in the FHE computations in the future.

For some simple models such a linear regression, there is no post-processing step but the method is kept to make the API consistent for the client-server API. Other models might need to use attributes stored in `post_processing_params`.

**Args:**

- <b>`y_preds`</b> (numpy.ndarray):  The de-quantized predictions to post-process.

**Returns:**

- <b>`numpy.ndarray`</b>:  The post-processed predictions.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L586"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(
    X: 'Data',
    fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>
) → ndarray
```

Predict values for X, in FHE or in the clear.

**Args:**

- <b>`X`</b> (Data):  The input values to predict, as a Numpy array, Torch tensor, Pandas DataFrame  or List.
- <b>`fhe`</b> (Union\[FheMode, str\]):  The mode to use for prediction.  Can be FheMode.DISABLE for Concrete ML Python inference,  FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.  Can also be the string representation of any of these values.  Default to FheMode.DISABLE.

**Returns:**

- <b>`np.ndarray`</b>:  The predicted values for X.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1569"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(X: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1635"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SklearnLinearRegressorMixin`

A Mixin class for sklearn linear regressors with FHE.

This class is used to create a linear regressor class that inherits from sklearn.base.RegressorMixin, which essentially gives access to scikit-learn's `score` method for regressors.

<a href="../../../src/concrete/ml/sklearn/base.py#L1437"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(n_bits: 'Union[int, Dict[str, int]]' = 8)
```

Initialize the FHE linear model.

**Args:**

- <b>`n_bits`</b> (int, Dict\[str, int\]):  Number of bits to quantize the model. If an int is passed  for n_bits, the value will be used for quantizing inputs and weights. If a dict is  passed, then it should contain "op_inputs" and "op_weights" as keys with  corresponding number of quantization bits so that:
  \- op_inputs : number of bits to quantize the input values
  \- op_weights: number of bits to quantize the learned parameters  Default to 8.

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete documentation (https://docs.zama.ai/concrete/getting-started/terminology_and_structure) Is None if the model is not fitted.

**Returns:**

- <b>`Circuit`</b>:  The FHE circuit.

______________________________________________________________________

#### <kbd>property</kbd> is_compiled

Indicate if the model is compiled.

**Returns:**

- <b>`bool`</b>:  If the model is compiled.

______________________________________________________________________

#### <kbd>property</kbd> is_fitted

Indicate if the model is fitted.

**Returns:**

- <b>`bool`</b>:  If the model is fitted.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

Is None if the model is not fitted.

**Returns:**

- <b>`onnx.ModelProto`</b>:  The ONNX model.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L309"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_compiled`

```python
check_model_is_compiled() → None
```

Check if the model is compiled.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not compiled.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L285"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_fitted`

```python
check_model_is_fitted() → None
```

Check if the model is fitted.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not fitted.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1617"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(*args, **kwargs) → Circuit
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1576"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(q_y_preds: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L220"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump`

```python
dump(file: 'TextIO') → None
```

Dump itself to a file.

**Args:**

- <b>`file`</b> (TextIO):  The file to dump the serialized object into.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L192"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict[str, Any]
```

Dump the object as a dict.

**Returns:**

- <b>`Dict[str, Any]`</b>:  Dict of serialized objects.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L212"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dumps`

```python
dumps() → str
```

Dump itself to a string.

**Returns:**

- <b>`metadata`</b> (str):  String of the serialized object.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1487"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X: 'Data', y: 'Target', **fit_parameters)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L397"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(
    X: 'Data',
    y: 'Target',
    random_state: 'Optional[int]' = None,
    **fit_parameters
)
```

Fit both the Concrete ML and its equivalent float estimators.

**Args:**

- <b>`X`</b> (Data):  The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
- <b>`y`</b> (Target):  The target data, as a Numpy array, Torch tensor, Pandas DataFrame, Pandas  Series or List.
- <b>`random_state`</b> (Optional\[int\]):  The random state to use when fitting. Defaults to None.
- <b>`**fit_parameters`</b>:  Keyword arguments to pass to the float estimator's fit method.

**Returns:**
The Concrete ML and float equivalent fitted estimators.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L318"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep: 'bool' = True) → dict
```

Get parameters for this estimator.

This method is used to instantiate a scikit-learn model using the Concrete ML model's parameters. It does not override scikit-learn's existing `get_params` method in order to not break its implementation of `set_params`.

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and contained  subobjects that are estimators. Default to True.

**Returns:**

- <b>`params`</b> (dict):  Parameter names mapped to their values.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: 'Dict[str, Any]') → BaseEstimator
```

Load itself from a dict.

**Args:**

- <b>`metadata`</b> (Dict\[str, Any\]):  Dict of serialized objects.

**Returns:**

- <b>`BaseEstimator`</b>:  The loaded object.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L670"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: 'ndarray') → ndarray
```

Apply post-processing to the de-quantized predictions.

This post-processing step can include operations such as applying the sigmoid or softmax function for classifiers, or summing an ensemble's outputs. These steps are done in the clear because of current technical constraints. They most likely will be integrated in the FHE computations in the future.

For some simple models such a linear regression, there is no post-processing step but the method is kept to make the API consistent for the client-server API. Other models might need to use attributes stored in `post_processing_params`.

**Args:**

- <b>`y_preds`</b> (numpy.ndarray):  The de-quantized predictions to post-process.

**Returns:**

- <b>`numpy.ndarray`</b>:  The post-processed predictions.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L586"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(
    X: 'Data',
    fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>
) → ndarray
```

Predict values for X, in FHE or in the clear.

**Args:**

- <b>`X`</b> (Data):  The input values to predict, as a Numpy array, Torch tensor, Pandas DataFrame  or List.
- <b>`fhe`</b> (Union\[FheMode, str\]):  The mode to use for prediction.  Can be FheMode.DISABLE for Concrete ML Python inference,  FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.  Can also be the string representation of any of these values.  Default to FheMode.DISABLE.

**Returns:**

- <b>`np.ndarray`</b>:  The predicted values for X.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1569"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(X: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1644"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SklearnLinearClassifierMixin`

A Mixin class for sklearn linear classifiers with FHE.

This class is used to create a linear classifier class that inherits from sklearn.base.ClassifierMixin, which essentially gives access to scikit-learn's `score` method for classifiers.

Additionally, this class adjusts some of the tree-based base class's methods in order to make them compliant with classification workflows.

<a href="../../../src/concrete/ml/sklearn/base.py#L702"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*args, **kwargs)
```

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete documentation (https://docs.zama.ai/concrete/getting-started/terminology_and_structure) Is None if the model is not fitted.

**Returns:**

- <b>`Circuit`</b>:  The FHE circuit.

______________________________________________________________________

#### <kbd>property</kbd> is_compiled

Indicate if the model is compiled.

**Returns:**

- <b>`bool`</b>:  If the model is compiled.

______________________________________________________________________

#### <kbd>property</kbd> is_fitted

Indicate if the model is fitted.

**Returns:**

- <b>`bool`</b>:  If the model is fitted.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

Is None if the model is not fitted.

**Returns:**

- <b>`onnx.ModelProto`</b>:  The ONNX model.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L309"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_compiled`

```python
check_model_is_compiled() → None
```

Check if the model is compiled.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not compiled.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L285"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_fitted`

```python
check_model_is_fitted() → None
```

Check if the model is fitted.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not fitted.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1617"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(*args, **kwargs) → Circuit
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1665"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `decision_function`

```python
decision_function(
    X: 'Data',
    fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>
) → ndarray
```

Predict confidence scores.

**Args:**

- <b>`X`</b> (Data):  The input values to predict, as a Numpy array, Torch tensor, Pandas DataFrame  or List.
- <b>`fhe`</b> (Union\[FheMode, str\]):  The mode to use for prediction.  Can be FheMode.DISABLE for Concrete ML Python inference,  FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.  Can also be the string representation of any of these values.  Default to FheMode.DISABLE.

**Returns:**

- <b>`numpy.ndarray`</b>:  The predicted confidence scores.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1576"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(q_y_preds: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L220"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump`

```python
dump(file: 'TextIO') → None
```

Dump itself to a file.

**Args:**

- <b>`file`</b> (TextIO):  The file to dump the serialized object into.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L192"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict[str, Any]
```

Dump the object as a dict.

**Returns:**

- <b>`Dict[str, Any]`</b>:  Dict of serialized objects.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L212"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dumps`

```python
dumps() → str
```

Dump itself to a string.

**Returns:**

- <b>`metadata`</b> (str):  String of the serialized object.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L715"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X: 'Data', y: 'Target', **fit_parameters)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L397"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(
    X: 'Data',
    y: 'Target',
    random_state: 'Optional[int]' = None,
    **fit_parameters
)
```

Fit both the Concrete ML and its equivalent float estimators.

**Args:**

- <b>`X`</b> (Data):  The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
- <b>`y`</b> (Target):  The target data, as a Numpy array, Torch tensor, Pandas DataFrame, Pandas  Series or List.
- <b>`random_state`</b> (Optional\[int\]):  The random state to use when fitting. Defaults to None.
- <b>`**fit_parameters`</b>:  Keyword arguments to pass to the float estimator's fit method.

**Returns:**
The Concrete ML and float equivalent fitted estimators.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L318"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep: 'bool' = True) → dict
```

Get parameters for this estimator.

This method is used to instantiate a scikit-learn model using the Concrete ML model's parameters. It does not override scikit-learn's existing `get_params` method in order to not break its implementation of `set_params`.

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and contained  subobjects that are estimators. Default to True.

**Returns:**

- <b>`params`</b> (dict):  Parameter names mapped to their values.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: 'Dict[str, Any]') → BaseEstimator
```

Load itself from a dict.

**Args:**

- <b>`metadata`</b> (Dict\[str, Any\]):  Dict of serialized objects.

**Returns:**

- <b>`BaseEstimator`</b>:  The loaded object.

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L760"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L749"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(
    X: 'Data',
    fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>
) → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1687"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict_proba`

```python
predict_proba(
    X: 'Data',
    fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>
) → ndarray
```

______________________________________________________________________

<a href="../../../src/concrete/ml/sklearn/base.py#L1569"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(X: 'ndarray') → ndarray
```
