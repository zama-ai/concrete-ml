<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.sklearn.qnn`

Scikit-learn interface for concrete quantized neural networks.

## **Global Variables**

- **MAX_BITWIDTH_BACKWARD_COMPATIBLE**
- **QNN_AUTO_KWARGS**

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L47"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SparseQuantNeuralNetImpl`

Sparse Quantized Neural Network classifier.

This class implements an MLP that is compatible with FHE constraints. The weights and activations are quantized to low bitwidth and pruning is used to ensure accumulators do not surpass an user-provided accumulator bit-width. The number of classes and number of layers are specified by the user, as well as the breadth of the network

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L56"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    input_dim,
    n_layers,
    n_outputs,
    n_hidden_neurons_multiplier=4,
    n_w_bits=3,
    n_a_bits=3,
    n_accum_bits=8,
    n_prune_neurons_percentage=0.0,
    activation_function=<class 'torch.nn.modules.activation.ReLU'>,
    quant_narrow=False,
    quant_signed=True
)
```

Sparse Quantized Neural Network constructor.

**Args:**

- <b>`input_dim`</b>:  Number of dimensions of the input data
- <b>`n_layers`</b>:  Number of linear layers for this network
- <b>`n_outputs`</b>:  Number of output classes or regression targets
- <b>`n_w_bits`</b>:  Number of weight bits
- <b>`n_a_bits`</b>:  Number of activation and input bits
- <b>`n_accum_bits`</b>:  Maximal allowed bitwidth of intermediate accumulators
- <b>`n_hidden_neurons_multiplier`</b>:  The number of neurons on the hidden will be the number  of dimensions of the input multiplied by `n_hidden_neurons_multiplier`. Note that  pruning is used to adjust the accumulator size to attempt to  keep the maximum accumulator bitwidth to  `n_accum_bits`, meaning that not all hidden layer neurons will be active.  The default value for `n_hidden_neurons_multiplier` is chosen for small dimensions  of the input. Reducing this value decreases the FHE inference time considerably  but also decreases the robustness and accuracy of model training.
- <b>`n_prune_neurons_percentage`</b>:  How many neurons to prune on the hidden layers. This  should be used mostly through the dedicated `.prune()` mechanism. This can  be used in when setting `n_hidden_neurons_multiplier` high (3-4), once good accuracy  is obtained, to speed up the model in FHE.
- <b>`activation_function`</b>:  a torch class that is used to construct activation functions in  the network (e.g. torch.ReLU, torch.SELU, torch.Sigmoid, etc)
- <b>`quant_narrow `</b>:  whether this network should use narrow range quantized integer values
- <b>`quant_signed `</b>:  whether to use signed quantized integer values

**Raises:**

- <b>`ValueError`</b>:  if the parameters have invalid values or the computed accumulator bitwidth  is zero

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L256"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `enable_pruning`

```python
enable_pruning()
```

Enable pruning in the network. Pruning must be made permanent to recover pruned weights.

**Raises:**

- <b>`ValueError`</b>:  if the quantization parameters are invalid

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L302"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(x)
```

Forward pass.

**Args:**

- <b>`x`</b> (torch.Tensor):  network input

**Returns:**

- <b>`x`</b> (torch.Tensor):  network prediction

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L190"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `make_pruning_permanent`

```python
make_pruning_permanent()
```

Make the learned pruning permanent in the network.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L170"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `max_active_neurons`

```python
max_active_neurons()
```

Compute the maximum number of active (non-zero weight) neurons.

The computation is done using the quantization parameters passed to the constructor. Warning: With the current quantization algorithm (asymmetric) the value returned by this function is not guaranteed to ensure FHE compatibility. For some weight distributions, weights that are 0 (which are pruned weights) will not be quantized to 0. Therefore the total number of active quantized neurons will not be equal to max_active_neurons.

**Returns:**

- <b>`n`</b> (int):  maximum number of active neurons

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L313"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `on_train_end`

```python
on_train_end()
```

Call back when training is finished, can be useful to remove training hooks.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L318"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `QuantizedSkorchEstimatorMixin`

Mixin class that adds quantization features to Skorch NN estimators.

______________________________________________________________________

#### <kbd>property</kbd> base_estimator_type

Get the sklearn estimator that should be trained by the child class.

______________________________________________________________________

#### <kbd>property</kbd> base_module_to_compile

Get the module that should be compiled to FHE. In our case this is a torch nn.Module.

**Returns:**

- <b>`module`</b> (nn.Module):  the instantiated torch module

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

______________________________________________________________________

#### <kbd>property</kbd> input_quantizers

Get the input quantizers if the model is fitted.

**Returns:**

- <b>`List[UniformQuantizer]`</b>:  The input quantizers, if the model is fitted.

______________________________________________________________________

#### <kbd>property</kbd> n_bits_quant

Get the number of bits used for quantization.

This is stored by the torch.nn.module instance and thus cannot be retrieved until this instance is created.

**Returns:**

- <b>`int`</b>:  the number of bits to quantize the network

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L386"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep=True)
```

Get parameters for benchmark when cloning a skorch wrapped NN.

We must remove all parameters related to the module. Skorch takes either a class or a class instance for the `module` parameter. We want to pass our trained model, a class instance. But for this to work, we need to remove all module related constructor params. If not, skorch will instantiate a new class instance of the same type as the passed module see skorch net.py NeuralNet::initialize_instance

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and contained  subobjects that are estimators. Default to True.

**Returns:**

- <b>`params`</b> (dict):  parameters to create an equivalent fp32 sklearn estimator for benchmark

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L326"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `infer`

```python
infer(x, **fit_params)
```

Perform a single inference step on a batch of data.

This method is specific to Skorch estimators.

**Args:**

- <b>`x`</b> (torch.Tensor):  A batch of the input data, produced by a Dataset
- <b>`**fit_params (dict) `</b>:  Additional parameters passed to the `forward` method of  the module and to the `self.train_split` call.

**Returns:**
A torch tensor with the inference results for each item in the input

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L432"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `on_train_end`

```python
on_train_end(net, X=None, y=None, **kwargs)
```

Call back when training is finished by the skorch wrapper.

Check if the underlying neural net has a callback for this event and, if so, call it.

**Args:**

- <b>`net`</b>:  estimator for which training has ended (equal to self)
- <b>`X`</b>:  data
- <b>`y`</b>:  targets
- <b>`kwargs`</b>:  other arguments

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L448"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FixedTypeSkorchNeuralNet`

A mixin with a helpful modification to a skorch estimator that fixes the module type.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L455"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_params`

```python
get_params(deep=True, **kwargs)
```

Get parameters for this estimator.

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and  contained subobjects that are estimators.
- <b>`**kwargs`</b>:  any additional parameters to pass to the sklearn BaseEstimator class

**Returns:**

- <b>`params `</b>:  dict, Parameter names mapped to their values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L482"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `prune`

```python
prune(X, y, n_prune_neurons_percentage, **fit_params)
```

Prune a copy of this NeuralNetwork model.

This can be used when the number of neurons on the hidden layers is too high. For example, when creating a Neural Network model with `n_hidden_neurons_multiplier` high (3-4), it can be used to speed up the model inference in FHE. Many times, up to 50% of neurons can be pruned without losing accuracy, when using this function to fine-tune an already trained model with good accuracy. This method should be used once good accuracy is obtained.

**Args:**

- <b>`X `</b>:  training data, can be a torch.tensor, numpy.ndarray or pandas DataFrame
- <b>`y `</b>:  training targets, can be a torch.tensor, numpy.ndarray or pandas DataFrame
- <b>`n_prune_neurons_percentage `</b>:  percentage of neurons to remove. A value of 0 means  no neurons are removed and a value of 1.0 means 100% of neurons would be removed.
- <b>`fit_params`</b>:  additional parameters to pass to the forward method of the underlying  nn.Module

**Returns:**

- <b>`result`</b>:  a new pruned copy of this NeuralNetClassifier or NeuralNetRegressor

**Raises:**

- <b>`ValueError`</b>:  if the model has not been trained or if the model is one that has already  been pruned

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L557"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NeuralNetClassifier`

Scikit-learn interface for quantized FHE compatible neural networks.

This class wraps a quantized NN implemented using our Torch tools as a scikit-learn Estimator. It uses the skorch package to handle training and scikit-learn compatibility, and adds quantization and compilation functionality. The neural network implemented by this class is a multi layer fully connected network trained with Quantization Aware Training (QAT).

The datatypes that are allowed for prediction by this wrapper are more restricted than standard scikit-learn estimators as this class needs to predict in FHE and network inference executor is the NumpyModule.

Inputs that are float64 will be casted to float32 before training as this should not have a significant impact on the model's performances. If the targets are integers of lower bitwidth, they will be safely casted to int64. Else, an error is raised.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L582"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    *args,
    criterion=<class 'torch.nn.modules.loss.CrossEntropyLoss'>,
    classes=None,
    optimizer=<class 'torch.optim.adam.Adam'>,
    **kwargs
)
```

______________________________________________________________________

#### <kbd>property</kbd> base_estimator_type

______________________________________________________________________

#### <kbd>property</kbd> base_module_to_compile

Get the module that should be compiled to FHE. In our case this is a torch nn.Module.

**Returns:**

- <b>`module`</b> (nn.Module):  the instantiated torch module

______________________________________________________________________

#### <kbd>property</kbd> classes\_

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

______________________________________________________________________

#### <kbd>property</kbd> history

______________________________________________________________________

#### <kbd>property</kbd> input_quantizers

Get the input quantizers if the model is fitted.

**Returns:**

- <b>`List[UniformQuantizer]`</b>:  The input quantizers, if the model is fitted.

______________________________________________________________________

#### <kbd>property</kbd> n_bits_quant

Get the number of bits used for quantization.

This is stored by the torch.nn.module instance and thus cannot be retrieved until this instance is created.

**Returns:**

- <b>`int`</b>:  the number of bits to quantize the network

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L644"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X, y, **fit_params)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L682"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(X: ndarray, y: ndarray, *args, **kwargs) → Tuple[Any, Any]
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L455"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_params`

```python
get_params(deep=True, **kwargs)
```

Get parameters for this estimator.

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and  contained subobjects that are estimators.
- <b>`**kwargs`</b>:  any additional parameters to pass to the sklearn BaseEstimator class

**Returns:**

- <b>`params `</b>:  dict, Parameter names mapped to their values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L386"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep=True)
```

Get parameters for benchmark when cloning a skorch wrapped NN.

We must remove all parameters related to the module. Skorch takes either a class or a class instance for the `module` parameter. We want to pass our trained model, a class instance. But for this to work, we need to remove all module related constructor params. If not, skorch will instantiate a new class instance of the same type as the passed module see skorch net.py NeuralNet::initialize_instance

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and contained  subobjects that are estimators. Default to True.

**Returns:**

- <b>`params`</b> (dict):  parameters to create an equivalent fp32 sklearn estimator for benchmark

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L326"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `infer`

```python
infer(x, **fit_params)
```

Perform a single inference step on a batch of data.

This method is specific to Skorch estimators.

**Args:**

- <b>`x`</b> (torch.Tensor):  A batch of the input data, produced by a Dataset
- <b>`**fit_params (dict) `</b>:  Additional parameters passed to the `forward` method of  the module and to the `self.train_split` call.

**Returns:**
A torch tensor with the inference results for each item in the input

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L432"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `on_train_end`

```python
on_train_end(net, X=None, y=None, **kwargs)
```

Call back when training is finished by the skorch wrapper.

Check if the underlying neural net has a callback for this event and, if so, call it.

**Args:**

- <b>`net`</b>:  estimator for which training has ended (equal to self)
- <b>`X`</b>:  data
- <b>`y`</b>:  targets
- <b>`kwargs`</b>:  other arguments

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L624"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L691"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict_proba`

```python
predict_proba(X, execute_in_fhe=False)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L482"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `prune`

```python
prune(X, y, n_prune_neurons_percentage, **fit_params)
```

Prune a copy of this NeuralNetwork model.

This can be used when the number of neurons on the hidden layers is too high. For example, when creating a Neural Network model with `n_hidden_neurons_multiplier` high (3-4), it can be used to speed up the model inference in FHE. Many times, up to 50% of neurons can be pruned without losing accuracy, when using this function to fine-tune an already trained model with good accuracy. This method should be used once good accuracy is obtained.

**Args:**

- <b>`X `</b>:  training data, can be a torch.tensor, numpy.ndarray or pandas DataFrame
- <b>`y `</b>:  training targets, can be a torch.tensor, numpy.ndarray or pandas DataFrame
- <b>`n_prune_neurons_percentage `</b>:  percentage of neurons to remove. A value of 0 means  no neurons are removed and a value of 1.0 means 100% of neurons would be removed.
- <b>`fit_params`</b>:  additional parameters to pass to the forward method of the underlying  nn.Module

**Returns:**

- <b>`result`</b>:  a new pruned copy of this NeuralNetClassifier or NeuralNetRegressor

**Raises:**

- <b>`ValueError`</b>:  if the model has not been trained or if the model is one that has already  been pruned

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L699"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NeuralNetRegressor`

Scikit-learn interface for quantized FHE compatible neural networks.

This class wraps a quantized NN implemented using our Torch tools as a scikit-learn Estimator. It uses the skorch package to handle training and scikit-learn compatibility, and adds quantization and compilation functionality. The neural network implemented by this class is a multi layer fully connected network trained with Quantization Aware Training (QAT).

The datatypes that are allowed for prediction by this wrapper are more restricted than standard scikit-learn estimators as this class needs to predict in FHE and network inference executor is the NumpyModule.

Inputs and targets that are float64 will be casted to float32 before training as this should not have a significant impact on the model's performances. An error is raised if these values are not floating points.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L722"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(*args, optimizer=<class 'torch.optim.adam.Adam'>, **kwargs)
```

______________________________________________________________________

#### <kbd>property</kbd> base_estimator_type

______________________________________________________________________

#### <kbd>property</kbd> base_module_to_compile

Get the module that should be compiled to FHE. In our case this is a torch nn.Module.

**Returns:**

- <b>`module`</b> (nn.Module):  the instantiated torch module

______________________________________________________________________

#### <kbd>property</kbd> fhe_circuit

______________________________________________________________________

#### <kbd>property</kbd> history

______________________________________________________________________

#### <kbd>property</kbd> input_quantizers

Get the input quantizers if the model is fitted.

**Returns:**

- <b>`List[UniformQuantizer]`</b>:  The input quantizers, if the model is fitted.

______________________________________________________________________

#### <kbd>property</kbd> n_bits_quant

Get the number of bits used for quantization.

This is stored by the torch.nn.module instance and thus cannot be retrieved until this instance is created.

**Returns:**

- <b>`int`</b>:  the number of bits to quantize the network

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L751"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X, y, **fit_params)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L778"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit_benchmark`

```python
fit_benchmark(X: ndarray, y: ndarray, *args, **kwargs) → Tuple[Any, Any]
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L455"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_params`

```python
get_params(deep=True, **kwargs)
```

Get parameters for this estimator.

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and  contained subobjects that are estimators.
- <b>`**kwargs`</b>:  any additional parameters to pass to the sklearn BaseEstimator class

**Returns:**

- <b>`params `</b>:  dict, Parameter names mapped to their values.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L386"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_sklearn_params`

```python
get_sklearn_params(deep=True)
```

Get parameters for benchmark when cloning a skorch wrapped NN.

We must remove all parameters related to the module. Skorch takes either a class or a class instance for the `module` parameter. We want to pass our trained model, a class instance. But for this to work, we need to remove all module related constructor params. If not, skorch will instantiate a new class instance of the same type as the passed module see skorch net.py NeuralNet::initialize_instance

**Args:**

- <b>`deep`</b> (bool):  If True, will return the parameters for this estimator and contained  subobjects that are estimators. Default to True.

**Returns:**

- <b>`params`</b> (dict):  parameters to create an equivalent fp32 sklearn estimator for benchmark

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L326"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `infer`

```python
infer(x, **fit_params)
```

Perform a single inference step on a batch of data.

This method is specific to Skorch estimators.

**Args:**

- <b>`x`</b> (torch.Tensor):  A batch of the input data, produced by a Dataset
- <b>`**fit_params (dict) `</b>:  Additional parameters passed to the `forward` method of  the module and to the `self.train_split` call.

**Returns:**
A torch tensor with the inference results for each item in the input

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L432"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `on_train_end`

```python
on_train_end(net, X=None, y=None, **kwargs)
```

Call back when training is finished by the skorch wrapper.

Check if the underlying neural net has a callback for this event and, if so, call it.

**Args:**

- <b>`net`</b>:  estimator for which training has ended (equal to self)
- <b>`X`</b>:  data
- <b>`y`</b>:  targets
- <b>`kwargs`</b>:  other arguments

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L786"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `predict_proba`

```python
predict_proba(X, execute_in_fhe=False)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L482"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `prune`

```python
prune(X, y, n_prune_neurons_percentage, **fit_params)
```

Prune a copy of this NeuralNetwork model.

This can be used when the number of neurons on the hidden layers is too high. For example, when creating a Neural Network model with `n_hidden_neurons_multiplier` high (3-4), it can be used to speed up the model inference in FHE. Many times, up to 50% of neurons can be pruned without losing accuracy, when using this function to fine-tune an already trained model with good accuracy. This method should be used once good accuracy is obtained.

**Args:**

- <b>`X `</b>:  training data, can be a torch.tensor, numpy.ndarray or pandas DataFrame
- <b>`y `</b>:  training targets, can be a torch.tensor, numpy.ndarray or pandas DataFrame
- <b>`n_prune_neurons_percentage `</b>:  percentage of neurons to remove. A value of 0 means  no neurons are removed and a value of 1.0 means 100% of neurons would be removed.
- <b>`fit_params`</b>:  additional parameters to pass to the forward method of the underlying  nn.Module

**Returns:**

- <b>`result`</b>:  a new pruned copy of this NeuralNetClassifier or NeuralNetRegressor

**Raises:**

- <b>`ValueError`</b>:  if the model has not been trained or if the model is one that has already  been pruned
