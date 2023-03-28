<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.sklearn.base`

Module that contains base classes for our libraries estimators.

## **Global Variables**

- **OPSET_VERSION_FOR_ONNX_EXPORT**

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BaseEstimator`

Base class for all estimators in Concrete-ML.

This class does not inherit from sklearn.base.BaseEstimator as it creates some conflicts with Skorch in QuantizedTorchEstimatorMixin's subclasses (more specifically, the `get_params` method is not properly inherited).

**Attributes:**

- <b>`_is_a_public_cml_model`</b> (bool):  Private attribute indicating if the class is a public model  (as opposed to base or mixin classes).

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L86"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```

Initialize the base class with common attributes used in all estimators.

An underscore "\_" is appended to attributes that were created while fitting the model. This is done in order to follow Scikit-Learn's standard format. More information available in their documentation: https://scikit-learn.org/stable/developers/develop.html#:~:text=Estimated%20Attributes%C2%B6

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete-Numpy documentation: https://docs.zama.ai/concrete-numpy/developer/terminology_and_structure#terminology Is None if the model is not fitted.

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L191"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_compiled`

```python
check_model_is_compiled()
```

Check if the model is compiled.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not compiled.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_fitted`

```python
check_model_is_fitted()
```

Check if the model is fitted.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not fitted.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L361"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    X,
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

- <b>`X`</b>:  A representative set of input values used for building cryptographic parameters.
- <b>`configuration`</b> (Optional\[Configuration\]):  Options to use for compilation. Default  to None.
- <b>`artifacts`</b> (Optional\[DebugArtifacts\]):  Artifacts information about the  compilation process to store for debugging.
- <b>`show_mlir`</b> (bool):  Indicate if the MLIR graph should be printed during compilation.
- <b>`p_error`</b> (Optional\[float\]):  Probability of error of a single PBS. A p_error value cannot  be given if a global_p_error value is already set. Default to None, which sets this  error to a default value.
- <b>`global_p_error`</b> (Optional\[float\]):  Probability of error of the full circuit. A  global_p_error value cannot be given if a p_error value is already set. This feature  is not supported during Virtual Library simulation, meaning the probability is  currently set to 0. Default to None, which sets this  error to a default value.
- <b>`verbose`</b> (bool):  Indicate if compilation information should be printed  during compilation. Default to False.

**Returns:**

- <b>`Circuit`</b>:  The compiled Circuit.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L340"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(q_y_preds: 'ndarray') → ndarray
```

Dequantize the output.

This step ensures that the fit method has been called.

**Args:**

- <b>`q_y_preds`</b> (numpy.ndarray):  The quantized output values to dequantize.

**Returns:**

- <b>`numpy.ndarray`</b>:  The dequantized output values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L538"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump`

```python
dump(file: 'IO[str]')
```

Dump itelf to a file.

**Args:**

- <b>`file`</b> (IO\[str\]):  file of where to dump.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L521"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict[str, Any]
```

Dump the object as a dict.

**Returns:**

- <b>`Dict[str, Any]`</b>:  a dict representing the object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L529"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dumps`

```python
dumps() → str
```

Dump itelf to a string.

**Returns:**

- <b>`metadata`</b> (str):  string of serialized object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L253"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X, y, **fit_parameters) → Any
```

Fit the estimator.

This method trains a Scikit-Learn estimator, computes its ONNX graph and defines the quantization parameters needed for proper FHE inference.

**Args:**

- <b>`X`</b>:  The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
- <b>`y`</b>:  The target data,  as a Numpy array, Torch tensor, Pandas DataFrame, Pandas Series  or List.
- <b>`**fit_parameters`</b>:  Keyword arguments to pass to the float estimator's fit method.

**Returns:**

- <b>`Any`</b>:  The fitted estimator.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L274"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(
    X,
    y,
    random_state: 'Optional[int]' = None,
    **fit_parameters
) → Tuple[Any, Any]
```

Fit both the Concrete-ML and its equivalent float estimators.

**Args:**

- <b>`X`</b>:  The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
- <b>`y`</b>:  The target data,  as a Numpy array, Torch tensor, Pandas DataFrame, Pandas Series  or List.
- <b>`random_state`</b> (Optional\[int\]):  The random state to use when fitting. Defaults to None.
- <b>`**fit_parameters`</b>:  Keyword arguments to pass to the float estimator's fit method.

**Returns:**

- <b>`Tuple[Any, Any]`</b>:  The Concrete-ML and float equivalent fitted estimators.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep: 'bool' = True) → dict
```

Get parameters for this estimator.

This method is used to instantiate a Scikit-Learn model using the Concrete-ML model's parameters. It does not override Scikit-Learn's existing `get_params` method in order to not break its implementation of `set_params`.

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and contained  subobjects that are estimators. Default to True.

**Returns:**

- <b>`params`</b> (dict):  Parameter names mapped to their values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L559"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load`

```python
load(file: 'IO[str]') → BaseEstimator
```

Load itelf from a file.

**Args:**

- <b>`file`</b> (IO\[str\]):  file of serialized object

**Returns:**

- <b>`BaseEstimator`</b>:  the loaded object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L547"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: 'Dict[str, Any]') → BaseEstimator
```

Load itelf from a dict.

**Args:**

- <b>`metadata`</b> (Dict\[str, Any\]):  dict of metadata of the object

**Returns:**

- <b>`BaseEstimator`</b>:  the loaded object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L572"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `loads`

```python
loads(metadata: 'str') → BaseEstimator
```

Load itelf from a string.

**Args:**

- <b>`metadata`</b> (str):  serialized object

**Returns:**

- <b>`BaseEstimator`</b>:  the loaded object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L501"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: 'ndarray') → ndarray
```

Apply post-processing to the dequantized predictions.

This post-processing step can include operations such as applying the sigmoid or softmax function for classifiers, or summing an ensemble's outputs. These steps are done in the clear because of current technical constraints. They most likely will be integrated in the FHE computations in the future.

For some simple models such a linear regression, there is no post-processing step but the method is kept to make the API consistent for the client-server API. Other models might need to use attributes stored in `post_processing_params`.

**Args:**

- <b>`y_preds`</b> (numpy.ndarray):  The dequantized predictions to post-process.

**Returns:**

- <b>`numpy.ndarray`</b>:  The post-processed predictions.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L443"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(
    X: 'ndarray',
    fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>
) → ndarray
```

Predict values for X, in FHE or in the clear.

**Args:**

- <b>`X`</b> (numpy.ndarray):  The input values to predict.
- <b>`fhe`</b> (Union\[FheMode, str\]):  The mode to use for prediction.  Can be FheMode.DISABLE for Concrete-ML python inference,  FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.  Can also be the string representation of any of these values.  Default to FheMode.DISABLE.

**Returns:**

- <b>`np.ndarray`</b>:  The predicted values for X.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L327"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L589"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BaseClassifier`

Base class for linear and tree-based classifiers in Concrete-ML.

This class inherits from BaseEstimator and modifies some of its methods in order to align them with classifier behaviors. This notably include applying a sigmoid/softmax post-processing to the predicted values as well as handling a mapping of classes in case they are not ordered.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L597"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*args, **kwargs)
```

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete-Numpy documentation: https://docs.zama.ai/concrete-numpy/developer/terminology_and_structure#terminology Is None if the model is not fitted.

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L191"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_compiled`

```python
check_model_is_compiled()
```

Check if the model is compiled.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not compiled.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_fitted`

```python
check_model_is_fitted()
```

Check if the model is fitted.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not fitted.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L361"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    X,
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

- <b>`X`</b>:  A representative set of input values used for building cryptographic parameters.
- <b>`configuration`</b> (Optional\[Configuration\]):  Options to use for compilation. Default  to None.
- <b>`artifacts`</b> (Optional\[DebugArtifacts\]):  Artifacts information about the  compilation process to store for debugging.
- <b>`show_mlir`</b> (bool):  Indicate if the MLIR graph should be printed during compilation.
- <b>`p_error`</b> (Optional\[float\]):  Probability of error of a single PBS. A p_error value cannot  be given if a global_p_error value is already set. Default to None, which sets this  error to a default value.
- <b>`global_p_error`</b> (Optional\[float\]):  Probability of error of the full circuit. A  global_p_error value cannot be given if a p_error value is already set. This feature  is not supported during Virtual Library simulation, meaning the probability is  currently set to 0. Default to None, which sets this  error to a default value.
- <b>`verbose`</b> (bool):  Indicate if compilation information should be printed  during compilation. Default to False.

**Returns:**

- <b>`Circuit`</b>:  The compiled Circuit.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L340"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(q_y_preds: 'ndarray') → ndarray
```

Dequantize the output.

This step ensures that the fit method has been called.

**Args:**

- <b>`q_y_preds`</b> (numpy.ndarray):  The quantized output values to dequantize.

**Returns:**

- <b>`numpy.ndarray`</b>:  The dequantized output values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L538"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump`

```python
dump(file: 'IO[str]')
```

Dump itelf to a file.

**Args:**

- <b>`file`</b> (IO\[str\]):  file of where to dump.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L521"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict[str, Any]
```

Dump the object as a dict.

**Returns:**

- <b>`Dict[str, Any]`</b>:  a dict representing the object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L529"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dumps`

```python
dumps() → str
```

Dump itelf to a string.

**Returns:**

- <b>`metadata`</b> (str):  string of serialized object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L610"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X, y, **fit_parameters) → Any
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L274"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(
    X,
    y,
    random_state: 'Optional[int]' = None,
    **fit_parameters
) → Tuple[Any, Any]
```

Fit both the Concrete-ML and its equivalent float estimators.

**Args:**

- <b>`X`</b>:  The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
- <b>`y`</b>:  The target data,  as a Numpy array, Torch tensor, Pandas DataFrame, Pandas Series  or List.
- <b>`random_state`</b> (Optional\[int\]):  The random state to use when fitting. Defaults to None.
- <b>`**fit_parameters`</b>:  Keyword arguments to pass to the float estimator's fit method.

**Returns:**

- <b>`Tuple[Any, Any]`</b>:  The Concrete-ML and float equivalent fitted estimators.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep: 'bool' = True) → dict
```

Get parameters for this estimator.

This method is used to instantiate a Scikit-Learn model using the Concrete-ML model's parameters. It does not override Scikit-Learn's existing `get_params` method in order to not break its implementation of `set_params`.

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and contained  subobjects that are estimators. Default to True.

**Returns:**

- <b>`params`</b> (dict):  Parameter names mapped to their values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L559"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load`

```python
load(file: 'IO[str]') → BaseEstimator
```

Load itelf from a file.

**Args:**

- <b>`file`</b> (IO\[str\]):  file of serialized object

**Returns:**

- <b>`BaseEstimator`</b>:  the loaded object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L547"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: 'Dict[str, Any]') → BaseEstimator
```

Load itelf from a dict.

**Args:**

- <b>`metadata`</b> (Dict\[str, Any\]):  dict of metadata of the object

**Returns:**

- <b>`BaseEstimator`</b>:  the loaded object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L572"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `loads`

```python
loads(metadata: 'str') → BaseEstimator
```

Load itelf from a string.

**Args:**

- <b>`metadata`</b> (str):  serialized object

**Returns:**

- <b>`BaseEstimator`</b>:  the loaded object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L652"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: 'ndarray')
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L641"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(X, fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>) → ndarray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L625"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict_proba`

```python
predict_proba(
    X,
    fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>
) → ndarray
```

Predict class probabilities.

**Args:**

- <b>`X`</b>:  The input values to predict.
- <b>`fhe`</b> (Union\[FheMode, str\]):  The mode to use for prediction.  Can be FheMode.DISABLE for Concrete-ML python inference,  FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.  Can also be the string representation of any of these values.  Default to FheMode.DISABLE.

**Returns:**

- <b>`numpy.ndarray`</b>:  The predicted class probabilities.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L327"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L674"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedTorchEstimatorMixin`

Mixin that provides quantization for a torch module and follows the Estimator API.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L684"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```

______________________________________________________________________

#### <kbd>property</kbd> base_module_to_compile

Get the Torch module that should be compiled to FHE.

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L191"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_compiled`

```python
check_model_is_compiled()
```

Check if the model is compiled.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not compiled.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_fitted`

```python
check_model_is_fitted()
```

Check if the model is fitted.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not fitted.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L942"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(X, *args, **kwargs) → Circuit
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L938"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(q_y_preds: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L538"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump`

```python
dump(file: 'IO[str]')
```

Dump itelf to a file.

**Args:**

- <b>`file`</b> (IO\[str\]):  file of where to dump.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L521"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict[str, Any]
```

Dump the object as a dict.

**Returns:**

- <b>`Dict[str, Any]`</b>:  a dict representing the object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L529"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dumps`

```python
dumps() → str
```

Dump itelf to a string.

**Returns:**

- <b>`metadata`</b> (str):  string of serialized object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L752"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X, y, **fit_parameters) → Any
```

Fit he estimator.

If the module was already initialized, the module will be re-initialized unless `warm_start` is set to True. In addition to the torch training step, this method performs quantization of the trained Torch model.

Values of dtype float64 are not supported and will be casted to float32.

**Args:**

- <b>`X`</b>:  The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
- <b>`y`</b>:  The target data,  as a Numpy array, Torch tensor, Pandas DataFrame, Pandas Series  or List.
- <b>`**fit_parameters`</b>:  Keyword arguments to pass to Skorch's fit method.

**Returns:**

- <b>`Any`</b>:  The fitted estimator.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L883"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(
    X,
    y,
    random_state: 'Optional[int]' = None,
    **fit_parameters
) → Tuple[Any, Any]
```

Fit the quantized estimator as well as its equivalent float estimator.

This function returns both the quantized estimator (itself) as well as its non-quantized (float) equivalent, which are both trained separately. This method differs from the BaseEstimator's `fit_benchmark` method as QNNs use QAT instead of PTQ. Hence, here, the float model is topologically equivalent as we have less control over the influence of QAT over the weights.

Values of dtype float64 are not supported and will be casted to float32.

**Args:**

- <b>`X`</b>:  The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
- <b>`y`</b>:  The target data,  as a Numpy array, Torch tensor, Pandas DataFrame Pandas Series  or List.
- <b>`random_state`</b> (Optional\[int\]):  The random state to use when fitting. However, Skorch  does not handle such a parameter and setting it will have no effect. Defaults  to None.
- <b>`**fit_parameters`</b>:  Keyword arguments to pass to Skorch's fit method.

**Returns:**

- <b>`Tuple[Any, Any]`</b>:  The Concrete-ML and equivalent Skorch fitted estimators.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep: 'bool' = True) → dict
```

Get parameters for this estimator.

This method is used to instantiate a Scikit-Learn model using the Concrete-ML model's parameters. It does not override Scikit-Learn's existing `get_params` method in order to not break its implementation of `set_params`.

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and contained  subobjects that are estimators. Default to True.

**Returns:**

- <b>`params`</b> (dict):  Parameter names mapped to their values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L559"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load`

```python
load(file: 'IO[str]') → BaseEstimator
```

Load itelf from a file.

**Args:**

- <b>`file`</b> (IO\[str\]):  file of serialized object

**Returns:**

- <b>`BaseEstimator`</b>:  the loaded object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L547"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: 'Dict[str, Any]') → BaseEstimator
```

Load itelf from a dict.

**Args:**

- <b>`metadata`</b> (Dict\[str, Any\]):  dict of metadata of the object

**Returns:**

- <b>`BaseEstimator`</b>:  the loaded object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L572"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `loads`

```python
loads(metadata: 'str') → BaseEstimator
```

Load itelf from a string.

**Args:**

- <b>`metadata`</b> (str):  serialized object

**Returns:**

- <b>`BaseEstimator`</b>:  the loaded object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1005"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L970"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(X, fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L973"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict_proba`

```python
predict_proba(X, fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>)
```

Predict values for a regressor and class probabilities for a classifier.

In Scikit-Learn, predict_proba is not defined for regressors. However, Skorch seems to use it as `predict`, and defines what to return using a post-processing method. We therefore decided to follow the same structure and therefore include `predict_proba` in the QNN's base class.

Values of dtype float64 are not supported and will be casted to float32.

**Args:**

- <b>`X`</b>:  The input values to predict.
- <b>`fhe`</b> (Union\[FheMode, str\]):  The mode to use for prediction.  Can be FheMode.DISABLE for Concrete-ML python inference,  FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.  Can also be the string representation of any of these values.  Default to FheMode.DISABLE.

**Returns:**

- <b>`numpy.ndarray`</b>:  The predicted values or class probabilities.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L931"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(X: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1030"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BaseTreeEstimatorMixin`

Mixin class for tree-based estimators.

This class inherits from sklearn.base.BaseEstimator in order to have access to Scikit-Learn's `get_params` and `set_params` methods.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1047"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete-Numpy documentation: https://docs.zama.ai/concrete-numpy/developer/terminology_and_structure#terminology Is None if the model is not fitted.

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L191"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_compiled`

```python
check_model_is_compiled()
```

Check if the model is compiled.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not compiled.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_fitted`

```python
check_model_is_fitted()
```

Check if the model is fitted.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not fitted.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1138"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(*args, **kwargs) → Circuit
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1116"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(q_y_preds: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L538"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump`

```python
dump(file: 'IO[str]')
```

Dump itelf to a file.

**Args:**

- <b>`file`</b> (IO\[str\]):  file of where to dump.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L521"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict[str, Any]
```

Dump the object as a dict.

**Returns:**

- <b>`Dict[str, Any]`</b>:  a dict representing the object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L529"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dumps`

```python
dumps() → str
```

Dump itelf to a string.

**Returns:**

- <b>`metadata`</b> (str):  string of serialized object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1069"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X, y, **fit_parameters) → Any
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L274"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(
    X,
    y,
    random_state: 'Optional[int]' = None,
    **fit_parameters
) → Tuple[Any, Any]
```

Fit both the Concrete-ML and its equivalent float estimators.

**Args:**

- <b>`X`</b>:  The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
- <b>`y`</b>:  The target data,  as a Numpy array, Torch tensor, Pandas DataFrame, Pandas Series  or List.
- <b>`random_state`</b> (Optional\[int\]):  The random state to use when fitting. Defaults to None.
- <b>`**fit_parameters`</b>:  Keyword arguments to pass to the float estimator's fit method.

**Returns:**

- <b>`Tuple[Any, Any]`</b>:  The Concrete-ML and float equivalent fitted estimators.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep: 'bool' = True) → dict
```

Get parameters for this estimator.

This method is used to instantiate a Scikit-Learn model using the Concrete-ML model's parameters. It does not override Scikit-Learn's existing `get_params` method in order to not break its implementation of `set_params`.

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and contained  subobjects that are estimators. Default to True.

**Returns:**

- <b>`params`</b> (dict):  Parameter names mapped to their values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L559"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load`

```python
load(file: 'IO[str]') → BaseEstimator
```

Load itelf from a file.

**Args:**

- <b>`file`</b> (IO\[str\]):  file of serialized object

**Returns:**

- <b>`BaseEstimator`</b>:  the loaded object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L547"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: 'Dict[str, Any]') → BaseEstimator
```

Load itelf from a dict.

**Args:**

- <b>`metadata`</b> (Dict\[str, Any\]):  dict of metadata of the object

**Returns:**

- <b>`BaseEstimator`</b>:  the loaded object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L572"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `loads`

```python
loads(metadata: 'str') → BaseEstimator
```

Load itelf from a string.

**Args:**

- <b>`metadata`</b> (str):  serialized object

**Returns:**

- <b>`BaseEstimator`</b>:  the loaded object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1169"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1162"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(
    X: 'ndarray',
    fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>
) → ndarray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1104"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(X: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1179"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BaseTreeRegressorMixin`

Mixin class for tree-based regressors.

This class is used to create a tree-based regressor class that inherits from sklearn.base.RegressorMixin, which essentially gives access to Scikit-Learn's `score` method for regressors.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1047"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete-Numpy documentation: https://docs.zama.ai/concrete-numpy/developer/terminology_and_structure#terminology Is None if the model is not fitted.

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L191"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_compiled`

```python
check_model_is_compiled()
```

Check if the model is compiled.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not compiled.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_fitted`

```python
check_model_is_fitted()
```

Check if the model is fitted.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not fitted.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1138"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(*args, **kwargs) → Circuit
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1116"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(q_y_preds: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L538"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump`

```python
dump(file: 'IO[str]')
```

Dump itelf to a file.

**Args:**

- <b>`file`</b> (IO\[str\]):  file of where to dump.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L521"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict[str, Any]
```

Dump the object as a dict.

**Returns:**

- <b>`Dict[str, Any]`</b>:  a dict representing the object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L529"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dumps`

```python
dumps() → str
```

Dump itelf to a string.

**Returns:**

- <b>`metadata`</b> (str):  string of serialized object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1069"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X, y, **fit_parameters) → Any
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L274"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(
    X,
    y,
    random_state: 'Optional[int]' = None,
    **fit_parameters
) → Tuple[Any, Any]
```

Fit both the Concrete-ML and its equivalent float estimators.

**Args:**

- <b>`X`</b>:  The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
- <b>`y`</b>:  The target data,  as a Numpy array, Torch tensor, Pandas DataFrame, Pandas Series  or List.
- <b>`random_state`</b> (Optional\[int\]):  The random state to use when fitting. Defaults to None.
- <b>`**fit_parameters`</b>:  Keyword arguments to pass to the float estimator's fit method.

**Returns:**

- <b>`Tuple[Any, Any]`</b>:  The Concrete-ML and float equivalent fitted estimators.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep: 'bool' = True) → dict
```

Get parameters for this estimator.

This method is used to instantiate a Scikit-Learn model using the Concrete-ML model's parameters. It does not override Scikit-Learn's existing `get_params` method in order to not break its implementation of `set_params`.

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and contained  subobjects that are estimators. Default to True.

**Returns:**

- <b>`params`</b> (dict):  Parameter names mapped to their values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L559"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load`

```python
load(file: 'IO[str]') → BaseEstimator
```

Load itelf from a file.

**Args:**

- <b>`file`</b> (IO\[str\]):  file of serialized object

**Returns:**

- <b>`BaseEstimator`</b>:  the loaded object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L547"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: 'Dict[str, Any]') → BaseEstimator
```

Load itelf from a dict.

**Args:**

- <b>`metadata`</b> (Dict\[str, Any\]):  dict of metadata of the object

**Returns:**

- <b>`BaseEstimator`</b>:  the loaded object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L572"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `loads`

```python
loads(metadata: 'str') → BaseEstimator
```

Load itelf from a string.

**Args:**

- <b>`metadata`</b> (str):  serialized object

**Returns:**

- <b>`BaseEstimator`</b>:  the loaded object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1169"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1162"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(
    X: 'ndarray',
    fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>
) → ndarray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1104"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(X: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1188"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BaseTreeClassifierMixin`

Mixin class for tree-based classifiers.

This class is used to create a tree-based classifier class that inherits from sklearn.base.ClassifierMixin, which essentially gives access to Scikit-Learn's `score` method for classifiers.

Additionally, this class adjusts some of the tree-based base class's methods in order to make them compliant with classification workflows.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L597"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*args, **kwargs)
```

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete-Numpy documentation: https://docs.zama.ai/concrete-numpy/developer/terminology_and_structure#terminology Is None if the model is not fitted.

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L191"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_compiled`

```python
check_model_is_compiled()
```

Check if the model is compiled.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not compiled.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_fitted`

```python
check_model_is_fitted()
```

Check if the model is fitted.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not fitted.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1138"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(*args, **kwargs) → Circuit
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1116"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(q_y_preds: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L538"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump`

```python
dump(file: 'IO[str]')
```

Dump itelf to a file.

**Args:**

- <b>`file`</b> (IO\[str\]):  file of where to dump.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L521"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict[str, Any]
```

Dump the object as a dict.

**Returns:**

- <b>`Dict[str, Any]`</b>:  a dict representing the object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L529"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dumps`

```python
dumps() → str
```

Dump itelf to a string.

**Returns:**

- <b>`metadata`</b> (str):  string of serialized object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L610"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X, y, **fit_parameters) → Any
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L274"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(
    X,
    y,
    random_state: 'Optional[int]' = None,
    **fit_parameters
) → Tuple[Any, Any]
```

Fit both the Concrete-ML and its equivalent float estimators.

**Args:**

- <b>`X`</b>:  The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
- <b>`y`</b>:  The target data,  as a Numpy array, Torch tensor, Pandas DataFrame, Pandas Series  or List.
- <b>`random_state`</b> (Optional\[int\]):  The random state to use when fitting. Defaults to None.
- <b>`**fit_parameters`</b>:  Keyword arguments to pass to the float estimator's fit method.

**Returns:**

- <b>`Tuple[Any, Any]`</b>:  The Concrete-ML and float equivalent fitted estimators.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep: 'bool' = True) → dict
```

Get parameters for this estimator.

This method is used to instantiate a Scikit-Learn model using the Concrete-ML model's parameters. It does not override Scikit-Learn's existing `get_params` method in order to not break its implementation of `set_params`.

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and contained  subobjects that are estimators. Default to True.

**Returns:**

- <b>`params`</b> (dict):  Parameter names mapped to their values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L559"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load`

```python
load(file: 'IO[str]') → BaseEstimator
```

Load itelf from a file.

**Args:**

- <b>`file`</b> (IO\[str\]):  file of serialized object

**Returns:**

- <b>`BaseEstimator`</b>:  the loaded object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L547"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: 'Dict[str, Any]') → BaseEstimator
```

Load itelf from a dict.

**Args:**

- <b>`metadata`</b> (Dict\[str, Any\]):  dict of metadata of the object

**Returns:**

- <b>`BaseEstimator`</b>:  the loaded object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L572"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `loads`

```python
loads(metadata: 'str') → BaseEstimator
```

Load itelf from a string.

**Args:**

- <b>`metadata`</b> (str):  serialized object

**Returns:**

- <b>`BaseEstimator`</b>:  the loaded object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L652"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: 'ndarray')
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L641"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(X, fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>) → ndarray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L625"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict_proba`

```python
predict_proba(
    X,
    fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>
) → ndarray
```

Predict class probabilities.

**Args:**

- <b>`X`</b>:  The input values to predict.
- <b>`fhe`</b> (Union\[FheMode, str\]):  The mode to use for prediction.  Can be FheMode.DISABLE for Concrete-ML python inference,  FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.  Can also be the string representation of any of these values.  Default to FheMode.DISABLE.

**Returns:**

- <b>`numpy.ndarray`</b>:  The predicted class probabilities.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1104"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(X: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1203"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SklearnLinearModelMixin`

A Mixin class for sklearn linear models with FHE.

This class inherits from sklearn.base.BaseEstimator in order to have access to Scikit-Learn's `get_params` and `set_params` methods.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1217"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete-Numpy documentation: https://docs.zama.ai/concrete-numpy/developer/terminology_and_structure#terminology Is None if the model is not fitted.

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L191"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_compiled`

```python
check_model_is_compiled()
```

Check if the model is compiled.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not compiled.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_fitted`

```python
check_model_is_fitted()
```

Check if the model is fitted.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not fitted.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L361"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    X,
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

- <b>`X`</b>:  A representative set of input values used for building cryptographic parameters.
- <b>`configuration`</b> (Optional\[Configuration\]):  Options to use for compilation. Default  to None.
- <b>`artifacts`</b> (Optional\[DebugArtifacts\]):  Artifacts information about the  compilation process to store for debugging.
- <b>`show_mlir`</b> (bool):  Indicate if the MLIR graph should be printed during compilation.
- <b>`p_error`</b> (Optional\[float\]):  Probability of error of a single PBS. A p_error value cannot  be given if a global_p_error value is already set. Default to None, which sets this  error to a default value.
- <b>`global_p_error`</b> (Optional\[float\]):  Probability of error of the full circuit. A  global_p_error value cannot be given if a p_error value is already set. This feature  is not supported during Virtual Library simulation, meaning the probability is  currently set to 0. Default to None, which sets this  error to a default value.
- <b>`verbose`</b> (bool):  Indicate if compilation information should be printed  during compilation. Default to False.

**Returns:**

- <b>`Circuit`</b>:  The compiled Circuit.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1360"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(q_y_preds: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L538"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump`

```python
dump(file: 'IO[str]')
```

Dump itelf to a file.

**Args:**

- <b>`file`</b> (IO\[str\]):  file of where to dump.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L521"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict[str, Any]
```

Dump the object as a dict.

**Returns:**

- <b>`Dict[str, Any]`</b>:  a dict representing the object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L529"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dumps`

```python
dumps() → str
```

Dump itelf to a string.

**Returns:**

- <b>`metadata`</b> (str):  string of serialized object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1275"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X, y, **fit_parameters) → Any
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L274"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(
    X,
    y,
    random_state: 'Optional[int]' = None,
    **fit_parameters
) → Tuple[Any, Any]
```

Fit both the Concrete-ML and its equivalent float estimators.

**Args:**

- <b>`X`</b>:  The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
- <b>`y`</b>:  The target data,  as a Numpy array, Torch tensor, Pandas DataFrame, Pandas Series  or List.
- <b>`random_state`</b> (Optional\[int\]):  The random state to use when fitting. Defaults to None.
- <b>`**fit_parameters`</b>:  Keyword arguments to pass to the float estimator's fit method.

**Returns:**

- <b>`Tuple[Any, Any]`</b>:  The Concrete-ML and float equivalent fitted estimators.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep: 'bool' = True) → dict
```

Get parameters for this estimator.

This method is used to instantiate a Scikit-Learn model using the Concrete-ML model's parameters. It does not override Scikit-Learn's existing `get_params` method in order to not break its implementation of `set_params`.

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and contained  subobjects that are estimators. Default to True.

**Returns:**

- <b>`params`</b> (dict):  Parameter names mapped to their values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L559"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load`

```python
load(file: 'IO[str]') → BaseEstimator
```

Load itelf from a file.

**Args:**

- <b>`file`</b> (IO\[str\]):  file of serialized object

**Returns:**

- <b>`BaseEstimator`</b>:  the loaded object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L547"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: 'Dict[str, Any]') → BaseEstimator
```

Load itelf from a dict.

**Args:**

- <b>`metadata`</b> (Dict\[str, Any\]):  dict of metadata of the object

**Returns:**

- <b>`BaseEstimator`</b>:  the loaded object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L572"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `loads`

```python
loads(metadata: 'str') → BaseEstimator
```

Load itelf from a string.

**Args:**

- <b>`metadata`</b> (str):  serialized object

**Returns:**

- <b>`BaseEstimator`</b>:  the loaded object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L501"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: 'ndarray') → ndarray
```

Apply post-processing to the dequantized predictions.

This post-processing step can include operations such as applying the sigmoid or softmax function for classifiers, or summing an ensemble's outputs. These steps are done in the clear because of current technical constraints. They most likely will be integrated in the FHE computations in the future.

For some simple models such a linear regression, there is no post-processing step but the method is kept to make the API consistent for the client-server API. Other models might need to use attributes stored in `post_processing_params`.

**Args:**

- <b>`y_preds`</b> (numpy.ndarray):  The dequantized predictions to post-process.

**Returns:**

- <b>`numpy.ndarray`</b>:  The post-processed predictions.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L443"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(
    X: 'ndarray',
    fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>
) → ndarray
```

Predict values for X, in FHE or in the clear.

**Args:**

- <b>`X`</b> (numpy.ndarray):  The input values to predict.
- <b>`fhe`</b> (Union\[FheMode, str\]):  The mode to use for prediction.  Can be FheMode.DISABLE for Concrete-ML python inference,  FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.  Can also be the string representation of any of these values.  Default to FheMode.DISABLE.

**Returns:**

- <b>`np.ndarray`</b>:  The predicted values for X.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1353"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(X: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1399"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SklearnLinearRegressorMixin`

A Mixin class for sklearn linear regressors with FHE.

This class is used to create a linear regressor class that inherits from sklearn.base.RegressorMixin, which essentially gives access to Scikit-Learn's `score` method for regressors.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1217"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete-Numpy documentation: https://docs.zama.ai/concrete-numpy/developer/terminology_and_structure#terminology Is None if the model is not fitted.

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L191"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_compiled`

```python
check_model_is_compiled()
```

Check if the model is compiled.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not compiled.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_fitted`

```python
check_model_is_fitted()
```

Check if the model is fitted.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not fitted.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L361"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    X,
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

- <b>`X`</b>:  A representative set of input values used for building cryptographic parameters.
- <b>`configuration`</b> (Optional\[Configuration\]):  Options to use for compilation. Default  to None.
- <b>`artifacts`</b> (Optional\[DebugArtifacts\]):  Artifacts information about the  compilation process to store for debugging.
- <b>`show_mlir`</b> (bool):  Indicate if the MLIR graph should be printed during compilation.
- <b>`p_error`</b> (Optional\[float\]):  Probability of error of a single PBS. A p_error value cannot  be given if a global_p_error value is already set. Default to None, which sets this  error to a default value.
- <b>`global_p_error`</b> (Optional\[float\]):  Probability of error of the full circuit. A  global_p_error value cannot be given if a p_error value is already set. This feature  is not supported during Virtual Library simulation, meaning the probability is  currently set to 0. Default to None, which sets this  error to a default value.
- <b>`verbose`</b> (bool):  Indicate if compilation information should be printed  during compilation. Default to False.

**Returns:**

- <b>`Circuit`</b>:  The compiled Circuit.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1360"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(q_y_preds: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L538"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump`

```python
dump(file: 'IO[str]')
```

Dump itelf to a file.

**Args:**

- <b>`file`</b> (IO\[str\]):  file of where to dump.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L521"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict[str, Any]
```

Dump the object as a dict.

**Returns:**

- <b>`Dict[str, Any]`</b>:  a dict representing the object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L529"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dumps`

```python
dumps() → str
```

Dump itelf to a string.

**Returns:**

- <b>`metadata`</b> (str):  string of serialized object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1275"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X, y, **fit_parameters) → Any
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L274"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(
    X,
    y,
    random_state: 'Optional[int]' = None,
    **fit_parameters
) → Tuple[Any, Any]
```

Fit both the Concrete-ML and its equivalent float estimators.

**Args:**

- <b>`X`</b>:  The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
- <b>`y`</b>:  The target data,  as a Numpy array, Torch tensor, Pandas DataFrame, Pandas Series  or List.
- <b>`random_state`</b> (Optional\[int\]):  The random state to use when fitting. Defaults to None.
- <b>`**fit_parameters`</b>:  Keyword arguments to pass to the float estimator's fit method.

**Returns:**

- <b>`Tuple[Any, Any]`</b>:  The Concrete-ML and float equivalent fitted estimators.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep: 'bool' = True) → dict
```

Get parameters for this estimator.

This method is used to instantiate a Scikit-Learn model using the Concrete-ML model's parameters. It does not override Scikit-Learn's existing `get_params` method in order to not break its implementation of `set_params`.

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and contained  subobjects that are estimators. Default to True.

**Returns:**

- <b>`params`</b> (dict):  Parameter names mapped to their values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L559"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load`

```python
load(file: 'IO[str]') → BaseEstimator
```

Load itelf from a file.

**Args:**

- <b>`file`</b> (IO\[str\]):  file of serialized object

**Returns:**

- <b>`BaseEstimator`</b>:  the loaded object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L547"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: 'Dict[str, Any]') → BaseEstimator
```

Load itelf from a dict.

**Args:**

- <b>`metadata`</b> (Dict\[str, Any\]):  dict of metadata of the object

**Returns:**

- <b>`BaseEstimator`</b>:  the loaded object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L572"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `loads`

```python
loads(metadata: 'str') → BaseEstimator
```

Load itelf from a string.

**Args:**

- <b>`metadata`</b> (str):  serialized object

**Returns:**

- <b>`BaseEstimator`</b>:  the loaded object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L501"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: 'ndarray') → ndarray
```

Apply post-processing to the dequantized predictions.

This post-processing step can include operations such as applying the sigmoid or softmax function for classifiers, or summing an ensemble's outputs. These steps are done in the clear because of current technical constraints. They most likely will be integrated in the FHE computations in the future.

For some simple models such a linear regression, there is no post-processing step but the method is kept to make the API consistent for the client-server API. Other models might need to use attributes stored in `post_processing_params`.

**Args:**

- <b>`y_preds`</b> (numpy.ndarray):  The dequantized predictions to post-process.

**Returns:**

- <b>`numpy.ndarray`</b>:  The post-processed predictions.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L443"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(
    X: 'ndarray',
    fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>
) → ndarray
```

Predict values for X, in FHE or in the clear.

**Args:**

- <b>`X`</b> (numpy.ndarray):  The input values to predict.
- <b>`fhe`</b> (Union\[FheMode, str\]):  The mode to use for prediction.  Can be FheMode.DISABLE for Concrete-ML python inference,  FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.  Can also be the string representation of any of these values.  Default to FheMode.DISABLE.

**Returns:**

- <b>`np.ndarray`</b>:  The predicted values for X.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1353"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(X: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1408"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SklearnLinearClassifierMixin`

A Mixin class for sklearn linear classifiers with FHE.

This class is used to create a linear classifier class that inherits from sklearn.base.ClassifierMixin, which essentially gives access to Scikit-Learn's `score` method for classifiers.

Additionally, this class adjusts some of the tree-based base class's methods in order to make them compliant with classification workflows.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L597"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*args, **kwargs)
```

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete-Numpy documentation: https://docs.zama.ai/concrete-numpy/developer/terminology_and_structure#terminology Is None if the model is not fitted.

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L191"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_compiled`

```python
check_model_is_compiled()
```

Check if the model is compiled.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not compiled.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L167"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_fitted`

```python
check_model_is_fitted()
```

Check if the model is fitted.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not fitted.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L361"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    X,
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

- <b>`X`</b>:  A representative set of input values used for building cryptographic parameters.
- <b>`configuration`</b> (Optional\[Configuration\]):  Options to use for compilation. Default  to None.
- <b>`artifacts`</b> (Optional\[DebugArtifacts\]):  Artifacts information about the  compilation process to store for debugging.
- <b>`show_mlir`</b> (bool):  Indicate if the MLIR graph should be printed during compilation.
- <b>`p_error`</b> (Optional\[float\]):  Probability of error of a single PBS. A p_error value cannot  be given if a global_p_error value is already set. Default to None, which sets this  error to a default value.
- <b>`global_p_error`</b> (Optional\[float\]):  Probability of error of the full circuit. A  global_p_error value cannot be given if a p_error value is already set. This feature  is not supported during Virtual Library simulation, meaning the probability is  currently set to 0. Default to None, which sets this  error to a default value.
- <b>`verbose`</b> (bool):  Indicate if compilation information should be printed  during compilation. Default to False.

**Returns:**

- <b>`Circuit`</b>:  The compiled Circuit.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1428"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `decision_function`

```python
decision_function(
    X,
    fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>
) → ndarray
```

Predict confidence scores.

**Args:**

- <b>`X`</b>:  The input values to predict.
- <b>`fhe`</b> (Union\[FheMode, str\]):  The mode to use for prediction.  Can be FheMode.DISABLE for Concrete-ML python inference,  FheMode.SIMULATE for FHE simulation and FheMode.EXECUTE for actual FHE execution.  Can also be the string representation of any of these values.  Default to FheMode.DISABLE.

**Returns:**

- <b>`numpy.ndarray`</b>:  The predicted confidence scores.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1360"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(q_y_preds: 'ndarray') → ndarray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L538"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump`

```python
dump(file: 'IO[str]')
```

Dump itelf to a file.

**Args:**

- <b>`file`</b> (IO\[str\]):  file of where to dump.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L521"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dump_dict`

```python
dump_dict() → Dict[str, Any]
```

Dump the object as a dict.

**Returns:**

- <b>`Dict[str, Any]`</b>:  a dict representing the object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L529"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dumps`

```python
dumps() → str
```

Dump itelf to a string.

**Returns:**

- <b>`metadata`</b> (str):  string of serialized object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L610"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X, y, **fit_parameters) → Any
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L274"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(
    X,
    y,
    random_state: 'Optional[int]' = None,
    **fit_parameters
) → Tuple[Any, Any]
```

Fit both the Concrete-ML and its equivalent float estimators.

**Args:**

- <b>`X`</b>:  The training data, as a Numpy array, Torch tensor, Pandas DataFrame or List.
- <b>`y`</b>:  The target data,  as a Numpy array, Torch tensor, Pandas DataFrame, Pandas Series  or List.
- <b>`random_state`</b> (Optional\[int\]):  The random state to use when fitting. Defaults to None.
- <b>`**fit_parameters`</b>:  Keyword arguments to pass to the float estimator's fit method.

**Returns:**

- <b>`Tuple[Any, Any]`</b>:  The Concrete-ML and float equivalent fitted estimators.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L200"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep: 'bool' = True) → dict
```

Get parameters for this estimator.

This method is used to instantiate a Scikit-Learn model using the Concrete-ML model's parameters. It does not override Scikit-Learn's existing `get_params` method in order to not break its implementation of `set_params`.

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and contained  subobjects that are estimators. Default to True.

**Returns:**

- <b>`params`</b> (dict):  Parameter names mapped to their values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L559"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load`

```python
load(file: 'IO[str]') → BaseEstimator
```

Load itelf from a file.

**Args:**

- <b>`file`</b> (IO\[str\]):  file of serialized object

**Returns:**

- <b>`BaseEstimator`</b>:  the loaded object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L547"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `load_dict`

```python
load_dict(metadata: 'Dict[str, Any]') → BaseEstimator
```

Load itelf from a dict.

**Args:**

- <b>`metadata`</b> (Dict\[str, Any\]):  dict of metadata of the object

**Returns:**

- <b>`BaseEstimator`</b>:  the loaded object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L572"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>classmethod</kbd> `loads`

```python
loads(metadata: 'str') → BaseEstimator
```

Load itelf from a string.

**Args:**

- <b>`metadata`</b> (str):  serialized object

**Returns:**

- <b>`BaseEstimator`</b>:  the loaded object

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L652"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: 'ndarray')
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L641"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(X, fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>) → ndarray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1448"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict_proba`

```python
predict_proba(
    X,
    fhe: 'Union[FheMode, str]' = <FheMode.DISABLE: 'disable'>
) → ndarray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1353"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(X: 'ndarray') → ndarray
```
