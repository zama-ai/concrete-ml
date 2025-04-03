<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/torch/lora.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.torch.lora`

This module contains classes for LoRA (Low-Rank Adaptation) FHE training and custom layers.

## **Global Variables**

- **LINEAR_LAYERS**

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L33"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `setup_logger`

```python
setup_logger(log_file: str, level=20)
```

Set up a logger that logs to both console and a file.

**Args:**

- <b>`log_file`</b> (str):  The path to the log file.
- <b>`level`</b> (int):  The logging level.

**Returns:**

- <b>`logging.Logger`</b>:  The logger instance.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L67"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `grad_to`

```python
grad_to(param, device: str) → None
```

Move parameter gradient to device.

**Args:**

- <b>`param`</b>:  torch parameter with gradient
- <b>`device`</b> (str):  target device for gradient

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L78"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>function</kbd> `optimizer_to`

```python
optimizer_to(optim, device)
```

Move optimizer object to device.

**Args:**

- <b>`optim`</b>:  torch optimizer
- <b>`device`</b> (str):  target device for gradient

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L668"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/torch/lora.py#L95"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LoraTraining`

LoraTraining module for fine-tuning with LoRA in a hybrid model setting.

This class is designed to enable Low-Rank Adaptation (LoRA) fine-tuning in a hybrid model context. It allows selective execution of forward and backward passes in FHE.

The class replaces standard linear layers with custom layers that are compatible with LoRA and FHE operations.

**Args:**

- <b>`model`</b> (torch.nn.Module):  The base model with LoRA layers to be fine-tuned.
- <b>`n_layers_to_skip_for_backprop`</b> (int):  Number of initial linear layers to keep as standard  layers. Since the first layer doesn't need backpropagation (no previous layer to  update), we typically skip 1 layer. Defaults to 1.
- <b>`loss_fn`</b> (callable, optional):  Loss function to compute the loss. If None, the model  is expected to return a loss.

<a href="../../../src/concrete/ml/torch/lora.py#L114"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(model, n_layers_to_skip_for_backprop=1, loss_fn=None)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L152"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/torch/lora.py#L226"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/torch/lora.py#L297"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `process_inputs`

```python
process_inputs(
    inputs: Any
) → Tuple[Union[Tensor, NoneType], Union[Tensor, NoneType]]
```

Process training inputs such as labels and attention mask.

**Args:**

- <b>`inputs`</b>:  a dict, BatchEncoding or tuple containing training data

**Returns:**

- <b>`res`</b>:  tuple containing the attention mask and the labels

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L180"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/torch/lora.py#L138"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `set_loss_scaling_factor`

```python
set_loss_scaling_factor(loss_scaling_factor: float)
```

Set a scaling factor for the loss to account for gradient accumulation.

This ensures that gradients are correctly averaged over multiple mini-batches when performing gradient accumulation, preventing them from being scaled up by the number of accumulation steps.

**Args:**

- <b>`loss_scaling_factor`</b> (float):  The number of gradient accumulation steps.  The loss will be divided by this factor  before backpropagation.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L330"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `LoraTrainer`

Trainer class for LoRA fine-tuning with FHE support.

This class handles:

- Training loop
- Periodic logging and evaluation
- Loss tracking
- Integration with hybrid FHE model

**Args:**

- <b>`model`</b> (nn.Module):  The base model with LoRA layers to be fine-tuned.
- <b>`optimizer`</b> (torch.optim.Optimizer):  Optimizer for training.
- <b>`loss_fn`</b> (callable):  Loss function to compute the loss.
- <b>`lr_scheduler`</b> (optional):  Learning rate scheduler.
- <b>`training_args`</b> (dict):  Training arguments.
- <b>`n_layers_to_skip_for_backprop`</b> (int):  Number of initial linear layers to keep as standard  layers. Since the first layer doesn't need backpropagation (no previous layer to  update), we typically skip 1 layer. Defaults to 1.
- <b>`eval_loader`</b> (DataLoader, optional):  DataLoader for evaluation data.
- <b>`eval_metric_fn`</b> (callable, optional):  Function(model, eval_loader) -> dict of metrics.
- <b>`logging_steps`</b> (int, optional):  Log loss every N training steps. Defaults to 1.
- <b>`eval_steps`</b> (int, optional):  Evaluate on eval set every N training steps. Defaults to 10.
- <b>`train_log_path`</b> (str, optional):  Path to a log file for training. Defaults to "training.log".

<a href="../../../src/concrete/ml/torch/lora.py#L356"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(
    model,
    optimizer,
    loss_fn=None,
    lr_scheduler=None,
    training_args=None,
    n_layers_to_skip_for_backprop=1,
    eval_loader: Optional[DataLoader] = None,
    eval_metric_fn: Optional[Callable] = None,
    logging_steps: int = 1,
    eval_steps: int = 10,
    train_log_path: str = 'training.log',
    checkpoint_dir: str = None
)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L416"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `compile`

```python
compile(inputset, n_bits=8, use_dynamic_quantization=True)
```

Compile the hybrid model with the given input set.

**Args:**

- <b>`inputset`</b> (tuple):  Input set for compilation.
- <b>`n_bits`</b> (int):  Bit width for quantization.
- <b>`use_dynamic_quantization`</b> (bool):  Whether to use dynamic quantization.

**Returns:**

- <b>`Tuple[int, int]`</b>:  The epoch and global step of the latest checkpoint if found,  else (0, 0).

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L659"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_gradient_stats`

```python
get_gradient_stats()
```

Return recorded gradient statistics.

**Returns:**

- <b>`Dict[str, List[float]]`</b>:  Gradient statistics per layer over time.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L624"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `get_training_losses`

```python
get_training_losses()
```

Return all recorded training losses.

**Returns:**

- <b>`List[float]`</b>:  All recorded training losses.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L487"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `load_checkpoint`

```python
load_checkpoint(checkpoint_path: str)
```

Load a training checkpoint and restore model, optimizer, and lr_scheduler.

**Args:**

- <b>`checkpoint_path`</b> (str):  Path to the checkpoint file.

**Returns:**

- <b>`Tuple[int, int]`</b>:  The epoch and global step of the checkpoint.

**Raises:**

- <b>`FileNotFoundError`</b>:  If the checkpoint file is not found.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L632"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_and_clear_private_info`

```python
save_and_clear_private_info(path)
```

Save the model and remove private information.

**Args:**

- <b>`path`</b> (str):  The path to save the model.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L464"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `save_checkpoint`

```python
save_checkpoint(epoch: int, global_step: int)
```

Save a training checkpoint.

**Args:**

- <b>`epoch`</b> (int):  The current epoch number.
- <b>`global_step`</b> (int):  The current global step number.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/lora.py#L515"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `train`

```python
train(
    train_loader: DataLoader,
    num_epochs: int = 1,
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
