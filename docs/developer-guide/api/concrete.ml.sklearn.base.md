<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.sklearn.base`

Module that contains base classes for our libraries estimators.

## **Global Variables**

- **OPSET_VERSION_FOR_ONNX_EXPORT**

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L1573"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_sklearn_models`

```python
get_sklearn_models()
```

Return the list of available models in Concrete-ML.

**Returns:**
the lists of models in Concrete-ML

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L1627"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L1645"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L1663"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L62"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedTorchEstimatorMixin`

Mixin that provides quantization for a torch module and follows the Estimator API.

This class should be mixed in with another that provides the full Estimator API. This class only provides modifiers for .fit() (with quantization) and .predict() (optionally in FHE)

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L80"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

Get the FHE circuit.

**Returns:**

- <b>`Circuit`</b>:  the FHE circuit

______________________________________________________________________

#### <kbd>property</kbd> input_quantizers

Get the input quantizers.

**Returns:**

- <b>`List[Quantizer]`</b>:  the input quantizers

______________________________________________________________________

#### <kbd>property</kbd> n_bits_quant

Get the number of quantization bits.

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

.. # noqa: DAR201

**Returns:**

- <b>`_onnx_model_`</b> (onnx.ModelProto):  the ONNX model

______________________________________________________________________

#### <kbd>property</kbd> output_quantizers

Get the input quantizers.

**Returns:**

- <b>`List[QuantizedArray]`</b>:  the input quantizers

______________________________________________________________________

#### <kbd>property</kbd> quantize_input

Get the input quantization function.

**Returns:**

- <b>`Callable `</b>:  function that quantizes the input

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L225"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

**Raises:**

- <b>`ValueError`</b>:  if called before the model is trained

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L292"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L451"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L94"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_params_for_benchmark`

```python
get_params_for_benchmark()
```

Get the parameters to instantiate the sklearn estimator trained by the child class.

**Returns:**

- <b>`params`</b> (dict):  dictionary with parameters that will initialize a new Estimator

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L188"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: ndarray) → ndarray
```

Post-processing the output.

**Args:**

- <b>`y_preds`</b> (numpy.ndarray):  the output to post-process

**Raises:**

- <b>`ValueError`</b>:  if unknown post-processing function

**Returns:**

- <b>`numpy.ndarray`</b>:  the post-processed output

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L379"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L397"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

**Raises:**

- <b>`ValueError`</b>:  if the estimator was not yet trained or compiled

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L546"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BaseTreeEstimatorMixin`

Mixin class for tree-based estimators.

A place to share methods that are used on all tree-based estimators.

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L572"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(n_bits: int)
```

Initialize the TreeBasedEstimatorMixin.

**Args:**

- <b>`n_bits`</b> (int):  number of bits used for quantization

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

.. # noqa: DAR201

**Returns:**

- <b>`onnx.ModelProto`</b>:  the ONNX model

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L701"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L611"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(y_preds: ndarray)
```

Dequantize the integer predictions.

**Args:**

- <b>`y_preds`</b> (numpy.ndarray):  the predictions

**Returns:**
the dequantized predictions

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L629"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L596"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(X: ndarray)
```

Quantize the input.

**Args:**

- <b>`X`</b> (numpy.ndarray):  the input

**Returns:**
the quantized input

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L782"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BaseTreeRegressorMixin`

Mixin class for tree-based regressors.

A place to share methods that are used on all tree-based regressors.

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L572"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(n_bits: int)
```

Initialize the TreeBasedEstimatorMixin.

**Args:**

- <b>`n_bits`</b> (int):  number of bits used for quantization

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

.. # noqa: DAR201

**Returns:**

- <b>`onnx.ModelProto`</b>:  the ONNX model

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L701"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L611"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(y_preds: ndarray)
```

Dequantize the integer predictions.

**Args:**

- <b>`y_preds`</b> (numpy.ndarray):  the predictions

**Returns:**
the dequantized predictions

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L797"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L629"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L845"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: ndarray) → ndarray
```

Apply post-processing to the predictions.

**Args:**

- <b>`y_preds`</b> (numpy.ndarray):  The predictions.

**Returns:**

- <b>`numpy.ndarray`</b>:  The post-processed predictions.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L867"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict`

```python
predict(X: ndarray, execute_in_fhe: bool = False) → ndarray
```

Predict the probability.

**Args:**

- <b>`X`</b> (numpy.ndarray):  The input data.
- <b>`execute_in_fhe`</b> (bool):  Whether to execute in FHE. Defaults to False.

**Returns:**

- <b>`numpy.ndarray`</b>:  The predicted probabilities.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L596"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(X: ndarray)
```

Quantize the input.

**Args:**

- <b>`X`</b> (numpy.ndarray):  the input

**Returns:**
the quantized input

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L891"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BaseTreeClassifierMixin`

Mixin class for tree-based classifiers.

A place to share methods that are used on all tree-based classifiers.

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L572"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(n_bits: int)
```

Initialize the TreeBasedEstimatorMixin.

**Args:**

- <b>`n_bits`</b> (int):  number of bits used for quantization

______________________________________________________________________

#### <kbd>property</kbd> onnx_model

Get the ONNX model.

.. # noqa: DAR201

**Returns:**

- <b>`onnx.ModelProto`</b>:  the ONNX model

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L701"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L611"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(y_preds: ndarray)
```

Dequantize the integer predictions.

**Args:**

- <b>`y_preds`</b> (numpy.ndarray):  the predictions

**Returns:**
the dequantized predictions

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L901"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L629"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L1001"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: ndarray) → ndarray
```

Apply post-processing to the predictions.

**Args:**

- <b>`y_preds`</b> (numpy.ndarray):  The predictions.

**Returns:**

- <b>`numpy.ndarray`</b>:  The post-processed predictions.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L961"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L979"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L596"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(X: ndarray)
```

Quantize the input.

**Args:**

- <b>`X`</b> (numpy.ndarray):  the input

**Returns:**
the quantized input

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L1025"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SklearnLinearModelMixin`

A Mixin class for sklearn linear models with FHE.

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L1040"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*args, n_bits: Union[int, Dict[str, int]] = 8, **kwargs)
```

Initialize the FHE linear model.

**Args:**

- <b>`n_bits`</b> (int, Dict\[str, int\]):  Number of bits to quantize the model. If an int is passed  for n_bits, the value will be used for quantizing inputs and weights. If a dict is  passed, then it should contain "op_inputs" and "op_weights" as keys with  corresponding number of quantization bits so that:
  \- op_inputs : number of bits to quantize the input values
  \- op_weights: number of bits to quantize the learned parameters  Default to 8.
- <b>`*args`</b>:  The arguments to pass to the sklearn linear model.
- <b>`**kwargs`</b>:  The keyword arguments to pass to the sklearn linear model.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L1436"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clean_graph`

```python
clean_graph()
```

Clean the graph of the onnx model.

This will remove the Cast node in the model's onnx.graph since they have no use in quantized or FHE models.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L1224"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L1379"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(q_y_preds: ndarray) → ndarray
```

Dequantize the output.

**Args:**

- <b>`q_y_preds`</b> (numpy.ndarray):  The quantized output to dequantize

**Returns:**

- <b>`numpy.ndarray`</b>:  The dequantized output

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L1067"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L1174"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L1402"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: ndarray) → ndarray
```

Post-processing the quantized output.

For linear models, post-processing only considers a dequantization step.

**Args:**

- <b>`y_preds`</b> (numpy.ndarray):  The quantized outputs to post-process

**Returns:**

- <b>`numpy.ndarray`</b>:  The post-processed output

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L1311"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L1368"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(X: ndarray)
```

Quantize the input.

**Args:**

- <b>`X`</b> (numpy.ndarray):  The input to quantize

**Returns:**

- <b>`numpy.ndarray`</b>:  The quantized input

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L1480"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SklearnLinearClassifierMixin`

A Mixin class for sklearn linear classifiers with FHE.

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L1040"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*args, n_bits: Union[int, Dict[str, int]] = 8, **kwargs)
```

Initialize the FHE linear model.

**Args:**

- <b>`n_bits`</b> (int, Dict\[str, int\]):  Number of bits to quantize the model. If an int is passed  for n_bits, the value will be used for quantizing inputs and weights. If a dict is  passed, then it should contain "op_inputs" and "op_weights" as keys with  corresponding number of quantization bits so that:
  \- op_inputs : number of bits to quantize the input values
  \- op_weights: number of bits to quantize the learned parameters  Default to 8.
- <b>`*args`</b>:  The arguments to pass to the sklearn linear model.
- <b>`**kwargs`</b>:  The keyword arguments to pass to the sklearn linear model.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L1563"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `clean_graph`

```python
clean_graph()
```

Clean the graph of the onnx model.

Any operators following gemm, including the sigmoid, softmax and argmax operators, are removed from the graph. They will be executed in clear in the post-processing method.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L1224"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L1519"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L1379"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `dequantize_output`

```python
dequantize_output(q_y_preds: ndarray) → ndarray
```

Dequantize the output.

**Args:**

- <b>`q_y_preds`</b> (numpy.ndarray):  The quantized output to dequantize

**Returns:**

- <b>`numpy.ndarray`</b>:  The dequantized output

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L1067"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L1174"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L1483"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `post_processing`

```python
post_processing(y_preds: ndarray, already_dequantized: bool = False)
```

Post-processing the predictions.

This step may include a dequantization of the inputs if not done previously, in particular within the client-server workflow.

**Args:**

- <b>`y_preds`</b> (numpy.ndarray):  The predictions to post-process.
- <b>`already_dequantized`</b> (bool):  Whether the inputs were already dequantized or not. Default  to False.

**Returns:**

- <b>`numpy.ndarray`</b>:  The post-processed predictions.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L1546"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L1532"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml/tree/release/0.6.x/src/concrete/ml/sklearn/base.py#L1368"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `quantize_input`

```python
quantize_input(X: ndarray)
```

Quantize the input.

**Args:**

- <b>`X`</b> (numpy.ndarray):  The input to quantize

**Returns:**

- <b>`numpy.ndarray`</b>:  The quantized input
