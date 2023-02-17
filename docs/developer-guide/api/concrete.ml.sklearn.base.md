<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.sklearn.base`

Module that contains base classes for our libraries estimators.

## **Global Variables**

- **OPSET_VERSION_FOR_ONNX_EXPORT**

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1558"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_sklearn_models`

```python
get_sklearn_models()
```

Return the list of available models in Concrete-ML.

**Returns:**
the lists of models in Concrete-ML

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1612"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_sklearn_linear_models`

```python
get_sklearn_linear_models(
    classifier: bool = True,
    regressor: bool = True,
    str_in_class_name: str = None
)
```

Return the list of available linear models in Concrete-ML.

**Args:**

- <b>`classifier`</b> (bool):  whether you want classifiers or not
- <b>`regressor`</b> (bool):  whether you want regressors or not
- <b>`str_in_class_name`</b> (str):  if not None, only return models with this as a substring in the  class name

**Returns:**
the lists of linear models in Concrete-ML

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1630"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_sklearn_tree_models`

```python
get_sklearn_tree_models(
    classifier: bool = True,
    regressor: bool = True,
    str_in_class_name: str = None
)
```

Return the list of available tree models in Concrete-ML.

**Args:**

- <b>`classifier`</b> (bool):  whether you want classifiers or not
- <b>`regressor`</b> (bool):  whether you want regressors or not
- <b>`str_in_class_name`</b> (str):  if not None, only return models with this as a substring in the  class name

**Returns:**
the lists of tree models in Concrete-ML

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1648"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_sklearn_neural_net_models`

```python
get_sklearn_neural_net_models(
    classifier: bool = True,
    regressor: bool = True,
    str_in_class_name: str = None
)
```

Return the list of available neural net models in Concrete-ML.

**Args:**

- <b>`classifier`</b> (bool):  whether you want classifiers or not
- <b>`regressor`</b> (bool):  whether you want regressors or not
- <b>`str_in_class_name`</b> (str):  if not None, only return models with this as a substring in the  class name

**Returns:**
the lists of neural net models in Concrete-ML

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BaseEstimator`

Base class for all estimators in Concrete-ML.

This class does not inherit from sklearn.base.BaseEstimator as it creates some conflicts with Skorch in QuantizedTorchEstimatorMixin's subclasses (more specifically, the `get_params` method is not properly inherited).

**Attributes:**

- <b>`_is_a_public_cml_model`</b> (bool):  Private attribute indicating if the class is a public model  (as opposed to base or mixin classes).

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L80"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```

Initialize the base class with common attributes used in all estimators.

An underscore "\_" is appended to attributes that were created while fitting the model. This is done in order to follow Scikit-Learn's standard format. More information available in their documentation: https://scikit-learn.org/stable/developers/develop.html#:~:text=Estimated%20Attributes%C2%B6

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete-Numpy documentation: https://docs.zama.ai/concrete-numpy/developer/terminology_and_structure#terminology Is None if the model was not fitted.

**Returns:**

- <b>`Circuit`</b>:  The FHE circuit.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

Is None if the model was not fitted.

**Returns:**

- <b>`onnx.ModelProto`</b>:  The ONNX model.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_compiled`

```python
check_model_is_compiled()
```

Check if the model is compiled.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not compiled.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_fitted`

```python
check_model_is_fitted()
```

Check if the model is fitted.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not fitted.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L210"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(q_y_preds: ndarray) → ndarray
```

Dequantize the output.

This step ensures that the fit method has been called.

**Args:**

- <b>`q_y_preds`</b> (numpy.ndarray):  The quantized output values to dequantize.

**Returns:**

- <b>`numpy.ndarray`</b>:  The dequantized output values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L173"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep=True)
```

Get parameters for this estimator.

This method is used to instantiate a Scikit-Learn model using the Concrete-ML model's parameters. It does not override Scikit-Learn's existing `get_params` method in order to not break its implementation of `set_params`.

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and contained  subobjects that are estimators. Default to True.

**Returns:**

- <b>`params`</b> (dict):  Parameter names mapped to their values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L224"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: ndarray) → ndarray
```

Apply post-processing to the dequantized predictions.

This post-processing step can include operations such as applying the sigmoid or softmax function for classifiers, or summing an ensemble's outputs. These steps are done in the clear because of current technical constraints. They most likely will be integrated in the FHE computations in the future.

For some simple models such a linear regression, there is no post-processing step but the method is kept to make the API consistent for the client-server API. Other models might need to use attributes stored in `post_processing_params`.

**Args:**

- <b>`y_preds`</b> (numpy.ndarray):  The dequantized predictions to post-process.

**Returns:**

- <b>`numpy.ndarray`</b>:  The post-processed predictions.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L197"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(X: ndarray) → ndarray
```

Quantize the input.

This step ensures that the fit method has been called.

**Args:**

- <b>`X`</b> (numpy.ndarray):  The input values to quantize.

**Returns:**

- <b>`numpy.ndarray`</b>:  The quantized input values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L245"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedTorchEstimatorMixin`

Mixin that provides quantization for a torch module and follows the Estimator API.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L255"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__()
```

______________________________________________________________________

#### <kbd>property</kbd> base_estimator_type

Get the sklearn estimator that should be trained by the child class.

______________________________________________________________________

#### <kbd>property</kbd> base_module_to_compile

Get the Torch module that should be compiled to FHE.

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

______________________________________________________________________

#### <kbd>property</kbd> input_quantizers

Get the input quantizers if the model is fitted.

**Returns:**

- <b>`List[UniformQuantizer]`</b>:  The input quantizers, if the model is fitted.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

Is None if the model was not fitted.

**Returns:**

- <b>`onnx.ModelProto`</b>:  The ONNX model.

______________________________________________________________________

#### <kbd>property</kbd> output_quantizers

Get the output quantizers if the model is fitted.

**Returns:**

- <b>`List[UniformQuantizer]`</b>:  The output quantizers, if the model is fitted.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_compiled`

```python
check_model_is_compiled()
```

Check if the model is compiled.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not compiled.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_fitted`

```python
check_model_is_fitted()
```

Check if the model is fitted.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not fitted.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L363"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    X: ndarray,
    configuration: Optional[Configuration] = None,
    compilation_artifacts: Optional[DebugArtifacts] = None,
    show_mlir: bool = False,
    use_virtual_lib: bool = False,
    p_error: Optional[float] = None,
    global_p_error: Optional[float] = None,
    verbose_compilation: bool = False
) → Circuit
```

Compile the model.

**Args:**

- <b>`X`</b> (numpy.ndarray):  the dequantized dataset
- <b>`configuration`</b> (Optional\[Configuration\]):  the options for  compilation
- <b>`compilation_artifacts`</b> (Optional\[DebugArtifacts\]):  artifacts object to fill  during compilation
- <b>`show_mlir`</b> (bool):  whether or not to show MLIR during the compilation
- <b>`use_virtual_lib`</b> (bool):  whether to compile using the virtual library that allows higher  bitwidths
- <b>`p_error`</b> (Optional\[float\]):  probability of error of a single PBS
- <b>`global_p_error`</b> (Optional\[float\]):  probability of error of the full circuit. Not  simulated by the VL, i.e., taken as 0
- <b>`verbose_compilation`</b> (bool):  whether to show compilation information

**Returns:**

- <b>`Circuit`</b>:  the compiled Circuit.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L324"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(q_y_preds: ndarray)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L426"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X, y, **fit_params)
```

Initialize and fit the module.

If the module was already initialized, by calling fit, the module will be re-initialized (unless `warm_start` is True). In addition to the torch training step, this method performs quantization of the trained torch model.

**Args:**

- <b>`X `</b>:  training data  By default, you should be able to pass:  * numpy arrays  * torch tensors  * pandas DataFrame or Series
- <b>`y`</b> (numpy.ndarray):  labels associated with training data
- <b>`**fit_params`</b>:  additional parameters that can be used during training, these are passed  to the torch training interface

**Returns:**

- <b>`self`</b>:  the trained quantized estimator

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L597"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(X: ndarray, y: ndarray, *args, **kwargs) → Tuple[Any, Any]
```

Fit the quantized estimator as well as its equivalent float estimator.

This function returns both the quantized estimator (itself) as well as its non-quantized (float) equivalent, which are both trained separately. This is useful in order to compare performances between quantized and fp32 versions.

**Args:**

- <b>`X `</b>:  The training data  By default, you should be able to pass:  * numpy arrays  * torch tensors  * pandas DataFrame or Series
- <b>`y`</b> (numpy.ndarray):  The labels associated with the training data
- <b>`*args`</b>:  The arguments to pass to the sklearn linear model.
- <b>`**kwargs`</b>:  The keyword arguments to pass to the sklearn linear model.

**Returns:**

- <b>`self`</b>:  The trained quantized estimator
- <b>`fp32_model`</b>:  The trained float equivalent estimator

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L173"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep=True)
```

Get parameters for this estimator.

This method is used to instantiate a Scikit-Learn model using the Concrete-ML model's parameters. It does not override Scikit-Learn's existing `get_params` method in order to not break its implementation of `set_params`.

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and contained  subobjects that are estimators. Default to True.

**Returns:**

- <b>`params`</b> (dict):  Parameter names mapped to their values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L328"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: ndarray) → ndarray
```

Apply post-processing to the dequantized predictions.

**Args:**

- <b>`y_preds`</b> (numpy.ndarray):  The dequantized predictions to post-process.

**Raises:**

- <b>`ValueError`</b>:  If the post-processing function is unknown.

**Returns:**

- <b>`numpy.ndarray`</b>:  The post-processed dequantized predictions.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L531"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(X, execute_in_fhe=False)
```

Predict on user provided data.

Predicts using the quantized clear or FHE classifier

**Args:**

- <b>`X `</b>:  input data, a numpy array of raw values (non quantized)
- <b>`execute_in_fhe `</b>:  whether to execute the inference in FHE or in the clear

**Returns:**

- <b>`y_pred `</b>:  numpy ndarray with predictions

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L549"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict_proba`

```python
predict_proba(X, execute_in_fhe=False)
```

Predict on user provided data, returning probabilities.

Predicts using the quantized clear or FHE classifier

**Args:**

- <b>`X `</b>:  input data, a numpy array of raw values (non quantized)
- <b>`execute_in_fhe `</b>:  whether to execute the inference in FHE or in the clear

**Returns:**

- <b>`y_pred `</b>:  numpy ndarray with probabilities (if applicable)

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L320"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(X: ndarray)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L675"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BaseTreeEstimatorMixin`

Mixin class for tree-based estimators.

This class inherits from sklearn.base.BaseEstimator in order to have access to Scikit-Learn's `get_params` and `set_params` methods.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L696"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(n_bits: int)
```

Initialize the TreeBasedEstimatorMixin.

**Args:**

- <b>`n_bits`</b> (int):  Number of bits used for quantization.

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete-Numpy documentation: https://docs.zama.ai/concrete-numpy/developer/terminology_and_structure#terminology Is None if the model was not fitted.

**Returns:**

- <b>`Circuit`</b>:  The FHE circuit.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

Is None if the model was not fitted.

**Returns:**

- <b>`onnx.ModelProto`</b>:  The ONNX model.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_compiled`

```python
check_model_is_compiled()
```

Check if the model is compiled.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not compiled.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_fitted`

```python
check_model_is_fitted()
```

Check if the model is fitted.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not fitted.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L880"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    X: ndarray,
    configuration: Optional[Configuration] = None,
    compilation_artifacts: Optional[DebugArtifacts] = None,
    show_mlir: bool = False,
    use_virtual_lib: bool = False,
    p_error: Optional[float] = None,
    global_p_error: Optional[float] = None,
    verbose_compilation: bool = False
) → Circuit
```

Compile the model.

**Args:**

- <b>`X`</b> (numpy.ndarray):  the dequantized dataset
- <b>`configuration`</b> (Optional\[Configuration\]):  the options for  compilation
- <b>`compilation_artifacts`</b> (Optional\[DebugArtifacts\]):  artifacts object to fill  during compilation
- <b>`show_mlir`</b> (bool):  whether or not to show MLIR during the compilation
- <b>`use_virtual_lib`</b> (bool):  set to True to use the so called virtual lib  simulating FHE computation. Defaults to False
- <b>`p_error`</b> (Optional\[float\]):  probability of error of a single PBS
- <b>`global_p_error`</b> (Optional\[float\]):  probability of error of the full circuit. Not  simulated by the VL, i.e., taken as 0
- <b>`verbose_compilation`</b> (bool):  whether to show compilation information

**Returns:**

- <b>`Circuit`</b>:  the compiled Circuit.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L724"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(q_y_preds: ndarray)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L734"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X, y: ndarray, **kwargs) → Any
```

Fit the tree-based estimator.

**Args:**

- <b>`X `</b>:  training data  By default, you should be able to pass:  * numpy arrays  * torch tensors  * pandas DataFrame or Series
- <b>`y`</b> (numpy.ndarray):  The target data.
- <b>`**kwargs`</b>:  args for super().fit

**Returns:**

- <b>`Any`</b>:  The fitted model.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L783"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(
    X: ndarray,
    y: ndarray,
    *args,
    random_state: Optional[int] = None,
    **kwargs
) → Tuple[Any, Any]
```

Fit the sklearn tree-based model and the FHE tree-based model.

**Args:**

- <b>`X`</b> (numpy.ndarray):  The input data.
- <b>`y`</b> (numpy.ndarray):  The target data. random_state (Optional\[Union\[int, numpy.random.RandomState, None\]\]):  The random state. Defaults to None.
- <b>`*args`</b>:  args for super().fit
- <b>`**kwargs`</b>:  kwargs for super().fit

**Returns:**
Tuple\[ConcreteEstimators, SklearnEstimators\]:  The FHE and sklearn tree-based models.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L173"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep=True)
```

Get parameters for this estimator.

This method is used to instantiate a Scikit-Learn model using the Concrete-ML model's parameters. It does not override Scikit-Learn's existing `get_params` method in order to not break its implementation of `set_params`.

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and contained  subobjects that are estimators. Default to True.

**Returns:**

- <b>`params`</b> (dict):  Parameter names mapped to their values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L957"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: ndarray) → ndarray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L849"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(X: ndarray, execute_in_fhe: bool = False) → ndarray
```

Predict values for X.

**Args:**

- <b>`X`</b> (numpy.ndarray):  The input data.
- <b>`execute_in_fhe`</b> (bool):  Whether to execute in FHE or not. Defaults to False.

**Returns:**

- <b>`numpy.ndarray`</b>:  The predicted values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L713"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(X: ndarray)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L966"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BaseTreeRegressorMixin`

Mixin class for tree-based regressors.

This class is used to created a tree-based regressor class that inherits from sklearn.base.RegressorMixin, which essentially gives access to Scikit-Learn's `score` method for regressors.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L696"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(n_bits: int)
```

Initialize the TreeBasedEstimatorMixin.

**Args:**

- <b>`n_bits`</b> (int):  Number of bits used for quantization.

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete-Numpy documentation: https://docs.zama.ai/concrete-numpy/developer/terminology_and_structure#terminology Is None if the model was not fitted.

**Returns:**

- <b>`Circuit`</b>:  The FHE circuit.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

Is None if the model was not fitted.

**Returns:**

- <b>`onnx.ModelProto`</b>:  The ONNX model.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_compiled`

```python
check_model_is_compiled()
```

Check if the model is compiled.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not compiled.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_fitted`

```python
check_model_is_fitted()
```

Check if the model is fitted.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not fitted.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L880"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    X: ndarray,
    configuration: Optional[Configuration] = None,
    compilation_artifacts: Optional[DebugArtifacts] = None,
    show_mlir: bool = False,
    use_virtual_lib: bool = False,
    p_error: Optional[float] = None,
    global_p_error: Optional[float] = None,
    verbose_compilation: bool = False
) → Circuit
```

Compile the model.

**Args:**

- <b>`X`</b> (numpy.ndarray):  the dequantized dataset
- <b>`configuration`</b> (Optional\[Configuration\]):  the options for  compilation
- <b>`compilation_artifacts`</b> (Optional\[DebugArtifacts\]):  artifacts object to fill  during compilation
- <b>`show_mlir`</b> (bool):  whether or not to show MLIR during the compilation
- <b>`use_virtual_lib`</b> (bool):  set to True to use the so called virtual lib  simulating FHE computation. Defaults to False
- <b>`p_error`</b> (Optional\[float\]):  probability of error of a single PBS
- <b>`global_p_error`</b> (Optional\[float\]):  probability of error of the full circuit. Not  simulated by the VL, i.e., taken as 0
- <b>`verbose_compilation`</b> (bool):  whether to show compilation information

**Returns:**

- <b>`Circuit`</b>:  the compiled Circuit.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L724"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(q_y_preds: ndarray)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L734"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X, y: ndarray, **kwargs) → Any
```

Fit the tree-based estimator.

**Args:**

- <b>`X `</b>:  training data  By default, you should be able to pass:  * numpy arrays  * torch tensors  * pandas DataFrame or Series
- <b>`y`</b> (numpy.ndarray):  The target data.
- <b>`**kwargs`</b>:  args for super().fit

**Returns:**

- <b>`Any`</b>:  The fitted model.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L783"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(
    X: ndarray,
    y: ndarray,
    *args,
    random_state: Optional[int] = None,
    **kwargs
) → Tuple[Any, Any]
```

Fit the sklearn tree-based model and the FHE tree-based model.

**Args:**

- <b>`X`</b> (numpy.ndarray):  The input data.
- <b>`y`</b> (numpy.ndarray):  The target data. random_state (Optional\[Union\[int, numpy.random.RandomState, None\]\]):  The random state. Defaults to None.
- <b>`*args`</b>:  args for super().fit
- <b>`**kwargs`</b>:  kwargs for super().fit

**Returns:**
Tuple\[ConcreteEstimators, SklearnEstimators\]:  The FHE and sklearn tree-based models.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L173"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep=True)
```

Get parameters for this estimator.

This method is used to instantiate a Scikit-Learn model using the Concrete-ML model's parameters. It does not override Scikit-Learn's existing `get_params` method in order to not break its implementation of `set_params`.

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and contained  subobjects that are estimators. Default to True.

**Returns:**

- <b>`params`</b> (dict):  Parameter names mapped to their values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L957"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: ndarray) → ndarray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L849"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(X: ndarray, execute_in_fhe: bool = False) → ndarray
```

Predict values for X.

**Args:**

- <b>`X`</b> (numpy.ndarray):  The input data.
- <b>`execute_in_fhe`</b> (bool):  Whether to execute in FHE or not. Defaults to False.

**Returns:**

- <b>`numpy.ndarray`</b>:  The predicted values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L713"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(X: ndarray)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L975"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BaseTreeClassifierMixin`

Mixin class for tree-based classifiers.

This class is used to created a tree-based classifier class that inherits from sklearn.base.ClassifierMixin, which essentially gives access to Scikit-Learn's `score` method for classifiers.

Additionally, this class adjusts some of the tree-based base class's methods in order to make them compliant with classification workflows.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L696"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(n_bits: int)
```

Initialize the TreeBasedEstimatorMixin.

**Args:**

- <b>`n_bits`</b> (int):  Number of bits used for quantization.

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete-Numpy documentation: https://docs.zama.ai/concrete-numpy/developer/terminology_and_structure#terminology Is None if the model was not fitted.

**Returns:**

- <b>`Circuit`</b>:  The FHE circuit.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

Is None if the model was not fitted.

**Returns:**

- <b>`onnx.ModelProto`</b>:  The ONNX model.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_compiled`

```python
check_model_is_compiled()
```

Check if the model is compiled.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not compiled.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_fitted`

```python
check_model_is_fitted()
```

Check if the model is fitted.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not fitted.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L880"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    X: ndarray,
    configuration: Optional[Configuration] = None,
    compilation_artifacts: Optional[DebugArtifacts] = None,
    show_mlir: bool = False,
    use_virtual_lib: bool = False,
    p_error: Optional[float] = None,
    global_p_error: Optional[float] = None,
    verbose_compilation: bool = False
) → Circuit
```

Compile the model.

**Args:**

- <b>`X`</b> (numpy.ndarray):  the dequantized dataset
- <b>`configuration`</b> (Optional\[Configuration\]):  the options for  compilation
- <b>`compilation_artifacts`</b> (Optional\[DebugArtifacts\]):  artifacts object to fill  during compilation
- <b>`show_mlir`</b> (bool):  whether or not to show MLIR during the compilation
- <b>`use_virtual_lib`</b> (bool):  set to True to use the so called virtual lib  simulating FHE computation. Defaults to False
- <b>`p_error`</b> (Optional\[float\]):  probability of error of a single PBS
- <b>`global_p_error`</b> (Optional\[float\]):  probability of error of the full circuit. Not  simulated by the VL, i.e., taken as 0
- <b>`verbose_compilation`</b> (bool):  whether to show compilation information

**Returns:**

- <b>`Circuit`</b>:  the compiled Circuit.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L724"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(q_y_preds: ndarray)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L990"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X, y: ndarray, **kwargs) → Any
```

Fit the tree-based estimator.

**Args:**

- <b>`X `</b>:  training data  By default, you should be able to pass:  * numpy arrays  * torch tensors  * pandas DataFrame or Series
- <b>`y`</b> (numpy.ndarray):  The target data.
- <b>`**kwargs`</b>:  args for super().fit

**Returns:**

- <b>`Any`</b>:  The fitted model.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L783"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(
    X: ndarray,
    y: ndarray,
    *args,
    random_state: Optional[int] = None,
    **kwargs
) → Tuple[Any, Any]
```

Fit the sklearn tree-based model and the FHE tree-based model.

**Args:**

- <b>`X`</b> (numpy.ndarray):  The input data.
- <b>`y`</b> (numpy.ndarray):  The target data. random_state (Optional\[Union\[int, numpy.random.RandomState, None\]\]):  The random state. Defaults to None.
- <b>`*args`</b>:  args for super().fit
- <b>`**kwargs`</b>:  kwargs for super().fit

**Returns:**
Tuple\[ConcreteEstimators, SklearnEstimators\]:  The FHE and sklearn tree-based models.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L173"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep=True)
```

Get parameters for this estimator.

This method is used to instantiate a Scikit-Learn model using the Concrete-ML model's parameters. It does not override Scikit-Learn's existing `get_params` method in order to not break its implementation of `set_params`.

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and contained  subobjects that are estimators. Default to True.

**Returns:**

- <b>`params`</b> (dict):  Parameter names mapped to their values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L957"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: ndarray) → ndarray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1023"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(X: ndarray, execute_in_fhe: bool = False) → ndarray
```

Predict the class with highest probability.

**Args:**

- <b>`X`</b> (numpy.ndarray):  The input data.
- <b>`execute_in_fhe`</b> (bool):  Whether to execute in FHE. Defaults to False.

**Returns:**

- <b>`numpy.ndarray`</b>:  The predicted target values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1049"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict_proba`

```python
predict_proba(X: ndarray, execute_in_fhe: bool = False) → ndarray
```

Predict the probability.

**Args:**

- <b>`X`</b> (numpy.ndarray):  The input data.
- <b>`execute_in_fhe`</b> (bool):  Whether to execute in FHE. Defaults to False.

**Returns:**

- <b>`numpy.ndarray`</b>:  The predicted probabilities.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L713"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(X: ndarray)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1065"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SklearnLinearModelMixin`

A Mixin class for sklearn linear models with FHE.

This class inherits from sklearn.base.BaseEstimator in order to have access to Scikit-Learn's `get_params` and `set_params` methods.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1088"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(n_bits: Union[int, Dict[str, int]] = 8)
```

Initialize the FHE linear model.

**Args:**

- <b>`n_bits`</b> (int, Dict\[str, int\]):  Number of bits to quantize the model. If an int is passed  for n_bits, the value will be used for quantizing inputs and weights. If a dict is  passed, then it should contain "op_inputs" and "op_weights" as keys with  corresponding number of quantization bits so that:
  \- op_inputs : number of bits to quantize the input values
  \- op_weights: number of bits to quantize the learned parameters  Default to 8.

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete-Numpy documentation: https://docs.zama.ai/concrete-numpy/developer/terminology_and_structure#terminology Is None if the model was not fitted.

**Returns:**

- <b>`Circuit`</b>:  The FHE circuit.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

Is None if the model was not fitted.

**Returns:**

- <b>`onnx.ModelProto`</b>:  The ONNX model.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_compiled`

```python
check_model_is_compiled()
```

Check if the model is compiled.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not compiled.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_fitted`

```python
check_model_is_fitted()
```

Check if the model is fitted.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not fitted.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1429"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clean_graph`

```python
clean_graph()
```

Clean the graph of the onnx model.

This will remove the Cast node in the model's onnx.graph since they have no use in quantized or FHE models.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1268"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    X: ndarray,
    configuration: Optional[Configuration] = None,
    compilation_artifacts: Optional[DebugArtifacts] = None,
    show_mlir: bool = False,
    use_virtual_lib: bool = False,
    p_error: Optional[float] = None,
    global_p_error: Optional[float] = None,
    verbose_compilation: bool = False
) → Circuit
```

Compile the FHE linear model.

**Args:**

- <b>`X`</b> (numpy.ndarray):  The input data.
- <b>`configuration`</b> (Optional\[Configuration\]):  Configuration object  to use during compilation
- <b>`compilation_artifacts`</b> (Optional\[DebugArtifacts\]):  Artifacts object to fill during  compilation
- <b>`show_mlir`</b> (bool):  If set, the MLIR produced by the converter and which is  going to be sent to the compiler backend is shown on the screen, e.g., for debugging  or demo. Defaults to False.
- <b>`use_virtual_lib`</b> (bool):  Whether to compile using the virtual library that allows higher  bitwidths with simulated FHE computation. Defaults to False
- <b>`p_error`</b> (Optional\[float\]):  Probability of error of a single PBS
- <b>`global_p_error`</b> (Optional\[float\]):  probability of error of the full circuit. Not  simulated by the VL, i.e., taken as 0
- <b>`verbose_compilation`</b> (bool):  whether to show compilation information

**Returns:**

- <b>`Circuit`</b>:  The compiled Circuit.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1401"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(q_y_preds: ndarray) → ndarray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1112"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X, y: ndarray, *args, **kwargs) → Any
```

Fit the FHE linear model.

**Args:**

- <b>`X `</b>:  Training data  By default, you should be able to pass:  * numpy arrays  * torch tensors  * pandas DataFrame or Series
- <b>`y`</b> (numpy.ndarray):  The target data.
- <b>`*args`</b>:  The arguments to pass to the sklearn linear model.
- <b>`**kwargs`</b>:  The keyword arguments to pass to the sklearn linear model.

**Returns:**
Any

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1220"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(
    X: ndarray,
    y: ndarray,
    *args,
    random_state: Optional[int] = None,
    **kwargs
) → Tuple[Any, Any]
```

Fit the sklearn linear model and the FHE linear model.

**Args:**

- <b>`X`</b> (numpy.ndarray):  The input data.
- <b>`y`</b> (numpy.ndarray):  The target data. random_state (Optional\[Union\[int, numpy.random.RandomState, None\]\]):  The random state. Defaults to None.
- <b>`*args`</b>:  The arguments to pass to the sklearn linear model.  or not (False). Default to False.
- <b>`*args`</b>:  Arguments for super().fit
- <b>`**kwargs`</b>:  Keyword arguments for super().fit

**Returns:**
Tuple\[SklearnLinearModelMixin, sklearn.linear_model.LinearRegression\]:  The FHE and sklearn LinearRegression.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L173"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep=True)
```

Get parameters for this estimator.

This method is used to instantiate a Scikit-Learn model using the Concrete-ML model's parameters. It does not override Scikit-Learn's existing `get_params` method in order to not break its implementation of `set_params`.

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and contained  subobjects that are estimators. Default to True.

**Returns:**

- <b>`params`</b> (dict):  Parameter names mapped to their values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L224"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: ndarray) → ndarray
```

Apply post-processing to the dequantized predictions.

This post-processing step can include operations such as applying the sigmoid or softmax function for classifiers, or summing an ensemble's outputs. These steps are done in the clear because of current technical constraints. They most likely will be integrated in the FHE computations in the future.

For some simple models such a linear regression, there is no post-processing step but the method is kept to make the API consistent for the client-server API. Other models might need to use attributes stored in `post_processing_params`.

**Args:**

- <b>`y_preds`</b> (numpy.ndarray):  The dequantized predictions to post-process.

**Returns:**

- <b>`numpy.ndarray`</b>:  The post-processed predictions.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1352"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(X: ndarray, execute_in_fhe: bool = False) → ndarray
```

Predict on user data.

Predict on user data using either the quantized clear model, implemented with tensors, or, if execute_in_fhe is set, using the compiled FHE circuit

**Args:**

- <b>`X`</b> (numpy.ndarray):  The input data
- <b>`execute_in_fhe`</b> (bool):  Whether to execute the inference in FHE

**Returns:**

- <b>`numpy.ndarray`</b>:  The prediction as ordinals

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1397"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(X: ndarray)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1468"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SklearnLineaRegressorMixin`

A Mixin class for sklearn linear regressors with FHE.

This class is used to created a linear regressor class that inherits from sklearn.base.RegressorMixin, which essentially gives access to Scikit-Learn's `score` method for regressors.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1088"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(n_bits: Union[int, Dict[str, int]] = 8)
```

Initialize the FHE linear model.

**Args:**

- <b>`n_bits`</b> (int, Dict\[str, int\]):  Number of bits to quantize the model. If an int is passed  for n_bits, the value will be used for quantizing inputs and weights. If a dict is  passed, then it should contain "op_inputs" and "op_weights" as keys with  corresponding number of quantization bits so that:
  \- op_inputs : number of bits to quantize the input values
  \- op_weights: number of bits to quantize the learned parameters  Default to 8.

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete-Numpy documentation: https://docs.zama.ai/concrete-numpy/developer/terminology_and_structure#terminology Is None if the model was not fitted.

**Returns:**

- <b>`Circuit`</b>:  The FHE circuit.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

Is None if the model was not fitted.

**Returns:**

- <b>`onnx.ModelProto`</b>:  The ONNX model.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_compiled`

```python
check_model_is_compiled()
```

Check if the model is compiled.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not compiled.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_fitted`

```python
check_model_is_fitted()
```

Check if the model is fitted.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not fitted.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1429"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clean_graph`

```python
clean_graph()
```

Clean the graph of the onnx model.

This will remove the Cast node in the model's onnx.graph since they have no use in quantized or FHE models.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1268"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    X: ndarray,
    configuration: Optional[Configuration] = None,
    compilation_artifacts: Optional[DebugArtifacts] = None,
    show_mlir: bool = False,
    use_virtual_lib: bool = False,
    p_error: Optional[float] = None,
    global_p_error: Optional[float] = None,
    verbose_compilation: bool = False
) → Circuit
```

Compile the FHE linear model.

**Args:**

- <b>`X`</b> (numpy.ndarray):  The input data.
- <b>`configuration`</b> (Optional\[Configuration\]):  Configuration object  to use during compilation
- <b>`compilation_artifacts`</b> (Optional\[DebugArtifacts\]):  Artifacts object to fill during  compilation
- <b>`show_mlir`</b> (bool):  If set, the MLIR produced by the converter and which is  going to be sent to the compiler backend is shown on the screen, e.g., for debugging  or demo. Defaults to False.
- <b>`use_virtual_lib`</b> (bool):  Whether to compile using the virtual library that allows higher  bitwidths with simulated FHE computation. Defaults to False
- <b>`p_error`</b> (Optional\[float\]):  Probability of error of a single PBS
- <b>`global_p_error`</b> (Optional\[float\]):  probability of error of the full circuit. Not  simulated by the VL, i.e., taken as 0
- <b>`verbose_compilation`</b> (bool):  whether to show compilation information

**Returns:**

- <b>`Circuit`</b>:  The compiled Circuit.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1401"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(q_y_preds: ndarray) → ndarray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1112"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X, y: ndarray, *args, **kwargs) → Any
```

Fit the FHE linear model.

**Args:**

- <b>`X `</b>:  Training data  By default, you should be able to pass:  * numpy arrays  * torch tensors  * pandas DataFrame or Series
- <b>`y`</b> (numpy.ndarray):  The target data.
- <b>`*args`</b>:  The arguments to pass to the sklearn linear model.
- <b>`**kwargs`</b>:  The keyword arguments to pass to the sklearn linear model.

**Returns:**
Any

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1220"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(
    X: ndarray,
    y: ndarray,
    *args,
    random_state: Optional[int] = None,
    **kwargs
) → Tuple[Any, Any]
```

Fit the sklearn linear model and the FHE linear model.

**Args:**

- <b>`X`</b> (numpy.ndarray):  The input data.
- <b>`y`</b> (numpy.ndarray):  The target data. random_state (Optional\[Union\[int, numpy.random.RandomState, None\]\]):  The random state. Defaults to None.
- <b>`*args`</b>:  The arguments to pass to the sklearn linear model.  or not (False). Default to False.
- <b>`*args`</b>:  Arguments for super().fit
- <b>`**kwargs`</b>:  Keyword arguments for super().fit

**Returns:**
Tuple\[SklearnLinearModelMixin, sklearn.linear_model.LinearRegression\]:  The FHE and sklearn LinearRegression.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L173"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep=True)
```

Get parameters for this estimator.

This method is used to instantiate a Scikit-Learn model using the Concrete-ML model's parameters. It does not override Scikit-Learn's existing `get_params` method in order to not break its implementation of `set_params`.

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and contained  subobjects that are estimators. Default to True.

**Returns:**

- <b>`params`</b> (dict):  Parameter names mapped to their values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L224"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: ndarray) → ndarray
```

Apply post-processing to the dequantized predictions.

This post-processing step can include operations such as applying the sigmoid or softmax function for classifiers, or summing an ensemble's outputs. These steps are done in the clear because of current technical constraints. They most likely will be integrated in the FHE computations in the future.

For some simple models such a linear regression, there is no post-processing step but the method is kept to make the API consistent for the client-server API. Other models might need to use attributes stored in `post_processing_params`.

**Args:**

- <b>`y_preds`</b> (numpy.ndarray):  The dequantized predictions to post-process.

**Returns:**

- <b>`numpy.ndarray`</b>:  The post-processed predictions.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1352"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(X: ndarray, execute_in_fhe: bool = False) → ndarray
```

Predict on user data.

Predict on user data using either the quantized clear model, implemented with tensors, or, if execute_in_fhe is set, using the compiled FHE circuit

**Args:**

- <b>`X`</b> (numpy.ndarray):  The input data
- <b>`execute_in_fhe`</b> (bool):  Whether to execute the inference in FHE

**Returns:**

- <b>`numpy.ndarray`</b>:  The prediction as ordinals

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1397"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(X: ndarray)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1477"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SklearnLinearClassifierMixin`

A Mixin class for sklearn linear classifiers with FHE.

This class is used to created a linear classifier class that inherits from sklearn.base.ClassifierMixin, which essentially gives access to Scikit-Learn's `score` method for classifiers.

Additionally, this class adjusts some of the tree-based base class's methods in order to make them compliant with classification workflows.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1088"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(n_bits: Union[int, Dict[str, int]] = 8)
```

Initialize the FHE linear model.

**Args:**

- <b>`n_bits`</b> (int, Dict\[str, int\]):  Number of bits to quantize the model. If an int is passed  for n_bits, the value will be used for quantizing inputs and weights. If a dict is  passed, then it should contain "op_inputs" and "op_weights" as keys with  corresponding number of quantization bits so that:
  \- op_inputs : number of bits to quantize the input values
  \- op_weights: number of bits to quantize the learned parameters  Default to 8.

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

Get the FHE circuit.

The FHE circuit combines computational graph, mlir, client and server into a single object. More information available in Concrete-Numpy documentation: https://docs.zama.ai/concrete-numpy/developer/terminology_and_structure#terminology Is None if the model was not fitted.

**Returns:**

- <b>`Circuit`</b>:  The FHE circuit.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

Is None if the model was not fitted.

**Returns:**

- <b>`onnx.ModelProto`</b>:  The ONNX model.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L161"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_compiled`

```python
check_model_is_compiled()
```

Check if the model is compiled.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not compiled.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L149"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `check_model_is_fitted`

```python
check_model_is_fitted()
```

Check if the model is fitted.

**Raises:**

- <b>`AttributeError`</b>:  If the model is not fitted.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1548"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clean_graph`

```python
clean_graph()
```

Clean the graph of the onnx model.

Any operators following gemm, including the sigmoid, softmax and argmax operators, are removed from the graph. They will be executed in clear in the post-processing method.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1268"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(
    X: ndarray,
    configuration: Optional[Configuration] = None,
    compilation_artifacts: Optional[DebugArtifacts] = None,
    show_mlir: bool = False,
    use_virtual_lib: bool = False,
    p_error: Optional[float] = None,
    global_p_error: Optional[float] = None,
    verbose_compilation: bool = False
) → Circuit
```

Compile the FHE linear model.

**Args:**

- <b>`X`</b> (numpy.ndarray):  The input data.
- <b>`configuration`</b> (Optional\[Configuration\]):  Configuration object  to use during compilation
- <b>`compilation_artifacts`</b> (Optional\[DebugArtifacts\]):  Artifacts object to fill during  compilation
- <b>`show_mlir`</b> (bool):  If set, the MLIR produced by the converter and which is  going to be sent to the compiler backend is shown on the screen, e.g., for debugging  or demo. Defaults to False.
- <b>`use_virtual_lib`</b> (bool):  Whether to compile using the virtual library that allows higher  bitwidths with simulated FHE computation. Defaults to False
- <b>`p_error`</b> (Optional\[float\]):  Probability of error of a single PBS
- <b>`global_p_error`</b> (Optional\[float\]):  probability of error of the full circuit. Not  simulated by the VL, i.e., taken as 0
- <b>`verbose_compilation`</b> (bool):  whether to show compilation information

**Returns:**

- <b>`Circuit`</b>:  The compiled Circuit.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1504"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `decision_function`

```python
decision_function(X: ndarray, execute_in_fhe: bool = False) → ndarray
```

Predict confidence scores for samples.

**Args:**

- <b>`X`</b> (numpy.ndarray):  Samples to predict.
- <b>`execute_in_fhe`</b> (bool):  If True, the inference will be executed in FHE. Default to False.

**Returns:**

- <b>`numpy.ndarray`</b>:  Confidence scores for samples.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1401"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(q_y_preds: ndarray) → ndarray
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1112"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X, y: ndarray, *args, **kwargs) → Any
```

Fit the FHE linear model.

**Args:**

- <b>`X `</b>:  Training data  By default, you should be able to pass:  * numpy arrays  * torch tensors  * pandas DataFrame or Series
- <b>`y`</b> (numpy.ndarray):  The target data.
- <b>`*args`</b>:  The arguments to pass to the sklearn linear model.
- <b>`**kwargs`</b>:  The keyword arguments to pass to the sklearn linear model.

**Returns:**
Any

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1220"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(
    X: ndarray,
    y: ndarray,
    *args,
    random_state: Optional[int] = None,
    **kwargs
) → Tuple[Any, Any]
```

Fit the sklearn linear model and the FHE linear model.

**Args:**

- <b>`X`</b> (numpy.ndarray):  The input data.
- <b>`y`</b> (numpy.ndarray):  The target data. random_state (Optional\[Union\[int, numpy.random.RandomState, None\]\]):  The random state. Defaults to None.
- <b>`*args`</b>:  The arguments to pass to the sklearn linear model.  or not (False). Default to False.
- <b>`*args`</b>:  Arguments for super().fit
- <b>`**kwargs`</b>:  Keyword arguments for super().fit

**Returns:**
Tuple\[SklearnLinearModelMixin, sklearn.linear_model.LinearRegression\]:  The FHE and sklearn LinearRegression.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L173"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep=True)
```

Get parameters for this estimator.

This method is used to instantiate a Scikit-Learn model using the Concrete-ML model's parameters. It does not override Scikit-Learn's existing `get_params` method in order to not break its implementation of `set_params`.

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and contained  subobjects that are estimators. Default to True.

**Returns:**

- <b>`params`</b> (dict):  Parameter names mapped to their values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1488"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: ndarray)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1531"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(X: ndarray, execute_in_fhe: bool = False) → ndarray
```

Predict on user data.

Predict on user data using either the quantized clear model, implemented with tensors, or, if execute_in_fhe is set, using the compiled FHE circuit.

**Args:**

- <b>`X`</b> (numpy.ndarray):  Samples to predict.
- <b>`execute_in_fhe`</b> (bool):  If True, the inference will be executed in FHE. Default to False.

**Returns:**

- <b>`numpy.ndarray`</b>:  The prediction as ordinals.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1517"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict_proba`

```python
predict_proba(X: ndarray, execute_in_fhe: bool = False) → ndarray
```

Predict class probabilities for samples.

**Args:**

- <b>`X`</b> (numpy.ndarray):  Samples to predict.
- <b>`execute_in_fhe`</b> (bool):  If True, the inference will be executed in FHE. Default to False.

**Returns:**

- <b>`numpy.ndarray`</b>:  Class probabilities for samples.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/base.py#L1397"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(X: ndarray)
```
