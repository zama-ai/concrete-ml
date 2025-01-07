<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/torch/lora.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.torch.lora`

This module contains classes for LoRA (Low-Rank Adaptation) FHE training and custom layers.

## **Global Variables**

- **LINEAR_LAYERS**

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L27"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `grad_to`

```python
grad_to(param, device: str) → None
```

Move parameter gradient to device.

**Args:**

- <b>`param`</b>:  torch parameter with gradient
- <b>`device`</b> (str):  target device for gradient

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L38"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `optimizer_to`

```python
optimizer_to(optim, device)
```

Move optimizer object to device.

**Args:**

- <b>`optim`</b>:  torch optimizer
- <b>`device`</b> (str):  target device for gradient

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L455"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `get_remote_names`

```python
get_remote_names(
    model: Module,
    include_embedding_layers: bool = False
) → List[str]
```

Get names of modules to be executed remotely.

**Args:**

- <b>`model`</b> (nn.Module):  The model to inspect.
- <b>`include_embedding_layers`</b> (bool):  Whether to include embedding layers.

**Returns:**

- <b>`List[str]`</b>:  List of module names to be executed remotely.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L55"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LoraTraining`

LoraTraining module for fine-tuning with LoRA in a hybrid model setting.

This class is designed to enable Low-Rank Adaptation (LoRA) fine-tuning in a hybrid model context. It allows selective execution of forward and backward passes in FHE.

The class replaces standard linear layers with custom layers that are compatible with LoRA and FHE operations. It provides mechanisms to toggle between calibration and optimization modes.

**Args:**

- <b>`model`</b> (torch.nn.Module):  The base model with LoRA layers to be fine-tuned.
- <b>`n_layers_to_skip_for_backprop`</b> (int):  Number of initial linear layers to keep as standard  layers. Since the first layer doesn't need backpropagation (no previous layer to  update), we typically skip 1 layer. Defaults to 1.
- <b>`loss_fn`</b> (callable, optional):  Loss function to compute the loss. If None, the model  is expected to return a loss.

<a href="../../../src/concrete/ml/torch/lora.py#L75"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(model, n_layers_to_skip_for_backprop=1, loss_fn=None)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L113"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `assert_has_lora_layers`

```python
assert_has_lora_layers(model)
```

Assert that the model contains LoRA layers.

**Args:**

- <b>`model`</b> (torch.nn.Module):  The model to check for LoRA layers.

**Raises:**

- <b>`ValueError`</b>:  If the model does not contain any LoRA layers.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L195"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `forward`

```python
forward(inputs: Tuple[Tensor, ]) → Tuple[Tensor, Union[Tensor, NoneType]]
```

Forward pass of the LoRA training module.

**Args:**

- <b>`inputs`</b> (tuple):  A tuple containing the input tensors.

**Returns:**
A tuple containing the original (unscaled) loss and None.

**Raises:**

- <b>`ValueError`</b>:  If the model does not return a loss and no loss function is provided.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L266"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `process_inputs`

```python
process_inputs(inputs: Any) → Tuple[Tensor, Tensor]
```

Process training inputs such as labels and attention mask.

**Args:**

- <b>`inputs`</b>:  a dict, BatchEncoding or tuple containing training data

**Returns:**

- <b>`res`</b>:  tuple containing the attention mask and the labels

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L141"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `replace_layers_with_custom`

```python
replace_layers_with_custom(
    model: Module,
    n_layers_to_skip_for_backprop: int
) → None
```

Replace linear layers with custom ones.

**Args:**

- <b>`model`</b> (nn.Module):  The model to replace layers in.
- <b>`n_layers_to_skip_for_backprop`</b> (int):  Number of initial linear layers to keep as standard  layers. Since the first layer doesn't need backpropagation (no previous layer to  update), we typically skip 1 layer.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L99"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_loss_scaling_factor`

```python
set_loss_scaling_factor(loss_scaling_factor: float)
```

Set a scaling factor for the loss to account for gradient accumulation.

This ensures that gradients are correctly averaged over multiple mini-batches when performing gradient accumulation, preventing them from being scaled up by the number of accumulation steps.

**Args:**

- <b>`loss_scaling_factor`</b> (float):  The number of gradient accumulation steps.  The loss will be divided by this factor  before backpropagation.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L187"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `toggle_calibrate`

```python
toggle_calibrate(enable: bool = True)
```

Toggle calibration mode.

**Args:**

- <b>`enable`</b> (bool):  Whether to enable calibration mode.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L298"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LoraTrainer`

Trainer class for LoRA fine-tuning with FHE support.

This class handles the training loop, optimizer, scheduler, and integrates with the hybrid model.

**Args:**

- <b>`model`</b> (nn.Module):  The base model with LoRA layers to be fine-tuned.
- <b>`optimizer`</b> (torch.optim.Optimizer):  Optimizer for training.
- <b>`loss_fn`</b> (callable):  Loss function to compute the loss.
- <b>`lr_scheduler`</b> (optional):  Learning rate scheduler.
- <b>`training_args`</b> (dict):  Training arguments.
- <b>`n_layers_to_skip_for_backprop`</b> (int):  Number of initial linear layers to keep as standard  layers. Since the first layer doesn't need backpropagation (no previous layer to  update), we typically skip 1 layer. Defaults to 1.

<a href="../../../src/concrete/ml/torch/lora.py#L315"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    model,
    optimizer,
    loss_fn=None,
    lr_scheduler=None,
    training_args=None,
    n_layers_to_skip_for_backprop=1
)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L354"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(inputset, n_bits=8)
```

Compile the hybrid model with the given input set.

**Args:**

- <b>`inputset`</b> (tuple):  Input set for compilation.
- <b>`n_bits`</b> (int):  Bit width for quantization.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L446"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_and_clear_private_info`

```python
save_and_clear_private_info(path)
```

Save the model and remove private information.

**Args:**

- <b>`path`</b> (str):  The path to save the model.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L365"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `train`

```python
train(
    train_loader: DataLoader,
    num_epochs: int = 10,
    fhe: str = 'simulate',
    device: str = 'cpu'
)
```

Train the model using the hybrid FHE model.

**Args:**

- <b>`train_loader`</b> (DataLoader):  DataLoader for training data.
- <b>`num_epochs`</b> (int):  Number of epochs to train.
- <b>`fhe`</b> (str):  FHE mode ('disable', 'simulate', 'execute' or 'torch').
- <b>`device`</b> (str):  A device string that is compatible with PyTorch, used for  client-side computations.
