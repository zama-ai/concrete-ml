<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/torch/lora.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.torch.lora`

This module contains classes for LoRA (Low-Rank Adaptation) training and custom layers.

## **Global Variables**

- **LINEAR_LAYERS**

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L321"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_remote_names`

```python
get_remote_names(
    model: Module,
    include_embedding_layers: bool = False
) → List[str]
```

Get names of modules to be executed remotely.

**Args:**

- <b>`model`</b> (torch.nn.Module):  The model to inspect.
- <b>`include_embedding_layers`</b> (bool):  Whether to include embedding layers.

**Returns:**

- <b>`List[str]`</b>:  List of module names to be executed remotely.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L22"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LoraTraining`

LoraTraining module for fine-tuning with LoRA in a hybrid model setting.

This class is designed to enable Low-Rank Adaptation (LoRA) fine-tuning in a hybrid model context. It allows selective execution of forward and backward passes in FHE.

The class replaces standard linear layers with custom layers that are compatible with LoRA and FHE operations. It provides mechanisms to toggle between calibration and optimization modes.

**Args:**

- <b>`inference_model`</b> (torch.nn.Module):  The base model to be fine-tuned.

<a href="../../../src/concrete/ml/torch/lora.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(inference_model) → None
```

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L132"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(inputs)
```

Forward pass of the LoRA training module.

**Args:**

- <b>`inputs`</b>:  A tuple containing input tensors and labels.

**Returns:**
A tuple containing the loss and gradient norm.

**Raises:**

- <b>`ValueError`</b>:  If the model does not return a loss when `self.loss_fn` is None.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L54"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `replace_layers_with_custom`

```python
replace_layers_with_custom(model: Module, skip_first: bool = True)
```

Replace linear layers with custom ones.

This method replaces eligible linear layers in the model with custom layers that are compatible with the LoRA training procedure.

**Args:**

- <b>`model`</b> (torch.nn.Module):  The model to replace layers in.
- <b>`skip_first`</b> (bool):  Whether to skip the first eligible layer.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L193"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `toggle_calibrate`

```python
toggle_calibrate(enable: bool = True)
```

Toggle calibration mode.

**Args:**

- <b>`enable`</b> (bool):  Whether to enable calibration mode.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L201"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `toggle_run_optimizer`

```python
toggle_run_optimizer(enable: bool = True)
```

Toggle optimizer execution.

**Args:**

- <b>`enable`</b> (bool):  Whether to enable optimizer execution.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L100"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `update_training_parameters`

```python
update_training_parameters(
    optimizer=None,
    lr_scheduler=None,
    loss_fn=None,
    training_args=None
)
```

Update training parameters for the LoRA module.

**Args:**

- <b>`optimizer`</b> (optional):  The optimizer to use for training.
- <b>`lr_scheduler`</b> (optional):  The learning rate scheduler to use for training.
- <b>`loss_fn`</b> (callable, optional):  Loss function to compute the loss.
- <b>`training_args`</b> (dict or namespace, optional):  Training arguments containing  'gradient_accumulation_steps' and 'max_grad_norm'.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L210"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ForwardModuleLinear`

Forward module for linear layers.

<a href="../../../src/concrete/ml/torch/lora.py#L213"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(weight, bias=None, weight_transposed=False)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L219"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(input_tensor)
```

Forward pass for linear layers.

**Args:**

- <b>`input_tensor`</b>:  The input tensor.

**Returns:**
The output tensor after applying the linear transformation.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L239"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BackwardModuleLinear`

Backward module for linear layers.

<a href="../../../src/concrete/ml/torch/lora.py#L242"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(weight, weight_transposed=False)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L247"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(grad_output)
```

Backward pass for linear layers.

**Args:**

- <b>`grad_output`</b>:  The gradient output tensor.

**Returns:**
The gradient input tensor after applying the backward pass.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L263"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CustomLinear`

Custom linear module.

<a href="../../../src/concrete/ml/torch/lora.py#L266"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(weight, bias=None, weight_transposed=False)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L271"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(input_tensor)
```

Forward pass of the custom linear module.

**Args:**

- <b>`input_tensor`</b>:  The input tensor.

**Returns:**
The output tensor after applying the custom linear module.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L283"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ForwardBackwardModule`

Custom autograd function for forward and backward passes.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L303"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `backward`

```python
backward(ctx, grad_output)
```

Backward pass of the custom autograd function.

**Args:**

- <b>`ctx`</b>:  The context object.
- <b>`grad_output`</b>:  The gradient output tensor.

**Returns:**
The gradient input tensor after applying the backward pass.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L286"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(ctx, input_tensor, forward_module, backward_module)
```

Forward pass of the custom autograd function.

**Args:**

- <b>`ctx`</b>:  The context object.
- <b>`input_tensor`</b>:  The input tensor.
- <b>`forward_module`</b>:  The forward module.
- <b>`backward_module`</b>:  The backward module.

**Returns:**
The output tensor after applying the forward pass.
