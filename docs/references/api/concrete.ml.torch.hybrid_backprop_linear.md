<!-- markdownlint-disable -->

<a href="../../../src/concrete/ml/torch/hybrid_backprop_linear.py#L0"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

# <kbd>module</kbd> `concrete.ml.torch.hybrid_backprop_linear`

Linear layer implementations for backprop FHE-compatible models.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_backprop_linear.py#L8"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ForwardModuleLinear`

Forward module for linear layers.

<a href="../../../src/concrete/ml/torch/hybrid_backprop_linear.py#L11"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(weight, bias=None, weight_transposed=False)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_backprop_linear.py#L17"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/torch/hybrid_backprop_linear.py#L37"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `BackwardModuleLinear`

Backward module for linear layers.

<a href="../../../src/concrete/ml/torch/hybrid_backprop_linear.py#L40"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(weight, weight_transposed=False)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_backprop_linear.py#L45"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/torch/hybrid_backprop_linear.py#L61"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `CustomLinear`

Custom linear module.

<a href="../../../src/concrete/ml/torch/hybrid_backprop_linear.py#L64"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

### <kbd>method</kbd> `__init__`

```python
__init__(weight, bias=None, weight_transposed=False)
```

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_backprop_linear.py#L69"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/torch/hybrid_backprop_linear.py#L81"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

## <kbd>class</kbd> `ForwardBackwardModule`

Custom autograd function for forward and backward passes.

______________________________________________________________________

<a href="../../../src/concrete/ml/torch/hybrid_backprop_linear.py#L101"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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

<a href="../../../src/concrete/ml/torch/hybrid_backprop_linear.py#L84"><img align="right" style="float:right;" src="https://img.shields.io/badge/-source-cccccc?style=flat-square"></a>

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
