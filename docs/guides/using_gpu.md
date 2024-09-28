# GPU acceleration

This document provides a complete instruction on using GPU acceleration with Concrete ML.

Concrete ML can compile both built-in and custom models using a CUDA-accelerated backend. However, once
a model is compiled for CUDA, executing it on a non-CUDA-enabled machine will raise an error.

## Support

| Feature     | Built-in models | Custom Models | Deployment | DataFrame |
| ----------- | --------------- | ------------- | ---------- | --------- |
| GPU support | ✅              | ✅            | ✅         | ❌        |
|             |                 |               |            |           |

{% hint style="warning" %}
When compiling a model for GPU, it will be assigned GPU-specific crypto-system parameters. These parameters are more constrained compared to CPU-specific ones.
As a result, the Concrete compiler may have difficulty finding suitable GPU-compatible crypto-parmaters for some models, leading to a `NoParametersFound` error.
{% endhint %}

## Performance

On high-end GPUs like V100, A100, or H100, the performance gains range from 1x to 10x compared to a desktop CPU. When compared to a high-end 64 or 96-core server CPU, the speed-up is typically around 1x to 3x.

On consumer grade GPUs such as GTX40xx or GTX30xx, there may be
little speedup or even a slowdown compared to execution
on a desktop CPU.

## Prerequisites

To use the CUDA enabled backend you need to install the GPU-enabled Concrete compiler:

```bash
pip install --extra-index-url https://pypi.zama.ai/gpu concrete-python
```

If you already have the same version of `concrete-python` installed, it will not be re-installed
automatically. In that case you may need to remove it manually before installing the GPU version.

```bash
pip uninstall concrete-python
pip install --extra-index-url https://pypi.zama.ai/gpu concrete-python
```

To switch back to the CPU-only version of the compiler, change the index-url to the
CPU-only repository or remove the index-url parameter alltogether:

```bash
pip uninstall concrete-python
pip install --extra-index-url https://pypi.zama.ai/cpu concrete-python
```

## Checking GPU can be enabled

To check if the CUDA acceleration is available you can use helper functions of `concrete-python`:

```python
import concrete.compiler; 
print("GPU enabled: ", concrete.compiler.check_gpu_enabled())
print("GPU available: ", concrete.compiler.check_gpu_available())
```

## Usage

To compile a model for cuda, simply supply the `device='cuda'` argument to its compilation function.
For built-in models, this function is `.compile`, while for custom models it can be
`compile_torch_model` or `compile_brevitas_qat_model`.
