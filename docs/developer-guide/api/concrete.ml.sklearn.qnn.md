<!-- markdownlint-disable -->

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.sklearn.qnn`

Scikit-learn interface for concrete quantized neural networks.

## **Global Variables**

- **MAXIMUM_TLU_BIT_WIDTH**

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L20"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `SparseQuantNeuralNetImpl`

Sparse Quantized Neural Network classifier.

This class implements an MLP that is compatible with FHE constraints. The weights and activations are quantized to low bit-width and pruning is used to ensure accumulators do not surpass an user-provided accumulator bit-width. The number of classes and number of layers are specified by the user, as well as the breadth of the network

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L29"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
    activation_function=<class 'torch.nn.modules.activation.ReLU'>
)
```

Sparse Quantized Neural Network constructor.

**Args:**

- <b>`input_dim`</b>:  Number of dimensions of the input data
- <b>`n_layers`</b>:  Number of linear layers for this network
- <b>`n_outputs`</b>:  Number of output classes or regression targets
- <b>`n_w_bits`</b>:  Number of weight bits
- <b>`n_a_bits`</b>:  Number of activation and input bits
- <b>`n_accum_bits`</b>:  Maximal allowed bit-width of intermediate accumulators
- <b>`n_hidden_neurons_multiplier`</b>:  A factor that is multiplied by the maximal number of  active (non-zero weight) neurons for every layer. The maximal number of neurons in  the worst case scenario is:  2^n_max-1  max_active_neurons(n_max, n_w, n_a) = floor(---------------------)  (2^n_w-1)\*(2^n_a-1) )  The worst case scenario for the bit-width of the accumulator is when all weights and  activations are maximum simultaneously. We set, for each layer, the total number of  neurons to be:  n_hidden_neurons_multiplier * max_active_neurons(n_accum_bits, n_w_bits, n_a_bits)  Through experiments, for typical distributions of weights and activations,  the default value for n_hidden_neurons_multiplier, 4, is safe to avoid overflow.
- <b>`activation_function`</b>:  a torch class that is used to construct activation functions in  the network (e.g. torch.ReLU, torch.SELU, torch.Sigmoid, etc)

**Raises:**

- <b>`ValueError`</b>:  if the parameters have invalid values or the computed accumulator bit-width  is zero

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L157"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `enable_pruning`

```python
enable_pruning()
```

Enable pruning in the network. Pruning must be made permanent to recover pruned weights.

**Raises:**

- <b>`ValueError`</b>:  if the quantization parameters are invalid

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L185"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L141"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `make_pruning_permanent`

```python
make_pruning_permanent()
```

Make the learned pruning permanent in the network.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L121"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `max_active_neurons`

```python
max_active_neurons()
```

Compute the maximum number of active (non-zero weight) neurons.

The computation is done using the quantization parameters passed to the constructor. Warning: With the current quantization algorithm (asymmetric) the value returned by this function is not guaranteed to ensure FHE compatibility. For some weight distributions, weights that are 0 (which are pruned weights) will not be quantized to 0. Therefore the total number of active quantized neurons will not be equal to max_active_neurons.

**Returns:**

- <b>`n`</b> (int):  maximum number of active neurons

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L196"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `on_train_end`

```python
on_train_end()
```

Call back when training is finished, can be useful to remove training hooks.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L201"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

Return the number of quantization bits.

This is stored by the torch.nn.module instance and thus cannot be retrieved until this instance is created.

**Returns:**

- <b>`n_bits`</b> (int):  the number of bits to quantize the network

**Raises:**

- <b>`ValueError`</b>:  with skorch estimators, the `module_` is not instantiated until .fit() is  called. Thus this estimator needs to be .fit() before we get the quantization number  of bits. If it is not trained we raise an exception

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L268"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_params_for_benchmark`

```python
get_params_for_benchmark()
```

Get parameters for benchmark when cloning a skorch wrapped NN.

We must remove all parameters related to the module. Skorch takes either a class or a class instance for the `module` parameter. We want to pass our trained model, a class instance. But for this to work, we need to remove all module related constructor params. If not, skorch will instantiate a new class instance of the same type as the passed module see skorch net.py NeuralNet::initialize_instance

**Returns:**

- <b>`params`</b> (dict):  parameters to create an equivalent fp32 sklearn estimator for benchmark

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L209"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L287"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L303"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `FixedTypeSkorchNeuralNet`

A mixin with a helpful modification to a skorch estimator that fixes the module type.

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L306"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L329"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NeuralNetClassifier`

Scikit-learn interface for quantized FHE compatible neural networks.

This class wraps a quantized NN implemented using our Torch tools as a scikit-learn Estimator. It uses the skorch package to handle training and scikit-learn compatibility, and adds quantization and compilation functionality. The neural network implemented by this class is a multi layer fully connected network trained with Quantization Aware Training (QAT).

The datatypes that are allowed for prediction by this wrapper are more restricted than standard scikit-learn estimators as this class needs to predict in FHE and network inference executor is the NumpyModule.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L346"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

Get the FHE circuit.

**Returns:**

- <b>`Circuit`</b>:  the FHE circuit

______________________________________________________________________

#### <kbd>property</kbd> history

______________________________________________________________________

#### <kbd>property</kbd> input_quantizers

Get the input quantizers.

**Returns:**

- <b>`List[Quantizer]`</b>:  the input quantizers

______________________________________________________________________

#### <kbd>property</kbd> n_bits_quant

Return the number of quantization bits.

This is stored by the torch.nn.module instance and thus cannot be retrieved until this instance is created.

**Returns:**

- <b>`n_bits`</b> (int):  the number of bits to quantize the network

**Raises:**

- <b>`ValueError`</b>:  with skorch estimators, the `module_` is not instantiated until .fit() is  called. Thus this estimator needs to be .fit() before we get the quantization number  of bits. If it is not trained we raise an exception

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L412"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X, y, **fit_params)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L306"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L268"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_params_for_benchmark`

```python
get_params_for_benchmark()
```

Get parameters for benchmark when cloning a skorch wrapped NN.

We must remove all parameters related to the module. Skorch takes either a class or a class instance for the `module` parameter. We want to pass our trained model, a class instance. But for this to work, we need to remove all module related constructor params. If not, skorch will instantiate a new class instance of the same type as the passed module see skorch net.py NeuralNet::initialize_instance

**Returns:**

- <b>`params`</b> (dict):  parameters to create an equivalent fp32 sklearn estimator for benchmark

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L209"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L287"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L395"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L429"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `NeuralNetRegressor`

Scikit-learn interface for quantized FHE compatible neural networks.

This class wraps a quantized NN implemented using our Torch tools as a scikit-learn Estimator. It uses the skorch package to handle training and scikit-learn compatibility, and adds quantization and compilation functionality. The neural network implemented by this class is a multi layer fully connected network trained with Quantization Aware Training (QAT).

The datatypes that are allowed for prediction by this wrapper are more restricted than standard scikit-learn estimators as this class needs to predict in FHE and network inference executor is the NumpyModule.

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L446"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

Get the FHE circuit.

**Returns:**

- <b>`Circuit`</b>:  the FHE circuit

______________________________________________________________________

#### <kbd>property</kbd> history

______________________________________________________________________

#### <kbd>property</kbd> input_quantizers

Get the input quantizers.

**Returns:**

- <b>`List[Quantizer]`</b>:  the input quantizers

______________________________________________________________________

#### <kbd>property</kbd> n_bits_quant

Return the number of quantization bits.

This is stored by the torch.nn.module instance and thus cannot be retrieved until this instance is created.

**Returns:**

- <b>`n_bits`</b> (int):  the number of bits to quantize the network

**Raises:**

- <b>`ValueError`</b>:  with skorch estimators, the `module_` is not instantiated until .fit() is  called. Thus this estimator needs to be .fit() before we get the quantization number  of bits. If it is not trained we raise an exception

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L473"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `fit`

```python
fit(X, y, **fit_params)
```

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L306"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L268"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_params_for_benchmark`

```python
get_params_for_benchmark()
```

Get parameters for benchmark when cloning a skorch wrapped NN.

We must remove all parameters related to the module. Skorch takes either a class or a class instance for the `module` parameter. We want to pass our trained model, a class instance. But for this to work, we need to remove all module related constructor params. If not, skorch will instantiate a new class instance of the same type as the passed module see skorch net.py NeuralNet::initialize_instance

**Returns:**

- <b>`params`</b> (dict):  parameters to create an equivalent fp32 sklearn estimator for benchmark

______________________________________________________________________

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L209"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="https://github.com/zama-ai/concrete-ml-internal/tree/main/src/concrete/ml/sklearn/qnn.py#L287"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
