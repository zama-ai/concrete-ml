# GPU acceleration

Concrete ML can compile both built-in and custom models using a CUDA-accelerated backend. Once
a model is compiled for CUDA, executing it on a non-CUDA enabled machine will result in
an error being raised.

## Support

| Feature     | Built-in models | Custom Models | Deployment | DataFrame |
| ----------- | --------------- | ------------- | ---------- | --------- |
| GPU support | ✅               | ✅             | ✅          | ❌         |
|             |                 |               |            |           |

{% hint style="warning" %}
When compiling a model for GPU it will be assigned gpu-specific crypto-system parameters. GPU-compatible
parameters are more constrained than CPU ones so, for some models, the Concrete compiler might have more difficulty finding GPU-compatible crypto-parmaters, resulting in the  `NoParametersFound` error.
{% endhint %}

## Performance

Performance gains between 1x-10x can be obtained on
high end GPUs such as V100, A100, H100, when compared to a desktop CPU. Compared to a high-end 64 or 96-core server CPU, speed-ups are around 1x-3x.

On consumer grade GPUs such as GTX40xx or GTX30xx there may be
little speedup or even a slowdown compared to executing
on a desktop CPU.

## Usage preqreuisites

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
