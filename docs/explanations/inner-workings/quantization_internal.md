# Quantization tools

## Quantizing data

Concrete ML has support for quantized ML models and also provides quantization tools for Quantization Aware Training and Post-Training Quantization. The core of this functionality is the conversion of floating point values to integers and back. This is done using `QuantizedArray` in `concrete.ml.quantization`.

The [`QuantizedArray`](../../references/api/concrete.ml.quantization.quantizers.md#class-quantizedarray) class takes several arguments that determine how float values are quantized:

- `n_bits` defines the precision used in quantization
- `values` are floating point values that will be converted to integers
- `is_signed` determines if the quantized integer values should allow negative values
- `is_symmetric` determines if the range of floating point values to be quantized should be taken as symmetric around zero

See also the [UniformQuantizer](../../references/api/concrete.ml.quantization.quantizers.md#class-uniformquantizer) reference for more information:

```python
from concrete.ml.quantization import QuantizedArray
import numpy
numpy.random.seed(0)
A = numpy.random.uniform(-2, 2, 10)
print("A = ", A)
# array([ 0.19525402,  0.86075747,  0.4110535,  0.17953273, -0.3053808,
#         0.58357645, -0.24965115,  1.567092 ,  1.85465104, -0.46623392])
q_A = QuantizedArray(7, A)
print("q_A.qvalues = ", q_A.qvalues)
# array([ 37,          73,          48,         36,          9,
#         58,          12,          112,        127,         0])
# the quantized integers values from A.
print("q_A.quantizer.scale = ", q_A.quantizer.scale)
# 0.018274684777173276, the scale S.
print("q_A.quantizer.zero_point = ", q_A.quantizer.zero_point)
# 26, the zero point Z.
print("q_A.dequant() = ", q_A.dequant())
# array([ 0.20102153,  0.85891018,  0.40204307,  0.18274685, -0.31066964,
#         0.58478991, -0.25584559,  1.57162289,  1.84574316, -0.4751418 ])
# Dequantized values.
```

It is also possible to use symmetric quantization, where the integer values are centered around 0:

<!--pytest-codeblocks:cont-->

```python
q_A = QuantizedArray(3, A)
print("Unsigned: q_A.qvalues = ", q_A.qvalues)
print("q_A.quantizer.zero_point = ", q_A.quantizer.zero_point)
# Unsigned: q_A.qvalues =  [2 4 2 2 0 3 0 6 7 0]
# q_A.quantizer.zero_point =  1

q_A = QuantizedArray(3, A, is_signed=True, is_symmetric=True)
print("Signed Symmetric: q_A.qvalues = ", q_A.qvalues)
print("q_A.quantizer.zero_point = ", q_A.quantizer.zero_point)
# Signed Symmetric: q_A.qvalues =  [ 0  1  1  0  0  1  0  3  3 -1]
# q_A.quantizer.zero_point =  0
```

In the following example, showing the de-quantization of model outputs, the `QuantizedArray` class is used in a different way. Here it uses pre-quantized integer values and has the `scale` and `zero-point` set explicitly. Once the `QuantizedArray` is constructed, calling `dequant()` will compute the floating point values corresponding to the integer values `qvalues`, which are the output of the `fhe_circuit.encrypt_run_decrypt(..)` call.

<!--pytest-codeblocks:cont-->

```python
import numpy
from concrete.ml.quantization.quantizers import QuantizationOptions

q_values = [0, 0, 1, 2, 3, -1]
QuantizedArray(
        q_A.quantizer.n_bits,
        q_values,
        value_is_float=False,
        options=q_A.quantizer.quant_options,
        stats=q_A.quantizer.quant_stats,
        params=q_A.quantizer.quant_params,
).dequant()

```

## Quantized modules

Machine learning models are implemented with a diverse set of operations, such as convolution, linear transformations, activation functions, and element-wise operations. When working with quantized values, these operations cannot be carried out in an equivalent way to floating point values. With quantization, it is necessary to re-scale the input and output values of each operation to fit in the quantization domain.

In Concrete ML, the quantized equivalent of a scikit-learn model or a PyTorch `nn.Module` is the `QuantizedModule`. Note that only inference is implemented in the `QuantizedModule`, and it is built through a conversion of the inference function of the corresponding scikit-learn or PyTorch module.

Built-in neural networks expose the `quantized_module` member, while a `QuantizedModule` is also the result of the compilation of custom models through `compile_torch_model` and `compile_brevitas_qat_model`.

The quantized versions of floating point model operations are stored in the `QuantizedModule`. The `ONNX_OPS_TO_QUANTIZED_IMPL` dictionary maps ONNX floating point operators (e.g., Gemm) to their quantized equivalent (e.g., QuantizedGemm). For more information on implementing these operations, please see the [FHE-compatible op-graph section](fhe-op-graphs.md).

The computation graph is taken from the corresponding floating point ONNX graph exported from scikit-learn [using HummingBird](external_libraries.md#hummingbird), or from the ONNX graph exported by PyTorch. Calibration is used to obtain quantized parameters for the operations in the `QuantizedModule`. Parameters are also determined for the quantization of inputs during model deployment.

{% hint style="info" %}
Calibration is the process of determining the typical distributions of values encountered for the intermediate values of a model during inference.

To perform calibration, an interpreter goes through the ONNX graph in [topological order](https://en.wikipedia.org/wiki/Topological_sorting) and stores the intermediate results as it goes. The statistics of these values determine quantization parameters.
{% endhint %}

That `QuantizedModule` generates the Concrete function that is compiled to FHE. The compilation will succeed if the intermediate values conform to the 16-bits precision limit of the Concrete stack. See [the compilation section](../compilation.md) for details.

## Resources

- Lei Mao's blog on quantization: [Quantization for Neural Networks](https://leimao.github.io/article/Neural-Networks-Quantization/)
- Google paper on neural network quantization and integer-only inference: [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
