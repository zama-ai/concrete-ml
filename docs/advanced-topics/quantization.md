# Quantization

Quantization is the process of constraining an input from a continuous or otherwise large set of values (such as the real numbers) to a discrete set (such as the integers).

This means that some accuracy in the representation is lost (e.g. a simple approach is to eliminate least-significant bits), but, in many cases in machine learning, it is possible to adapt the models to give meaningful results while using these smaller data types. This significantly reduces the number of bits necessary for intermediary results during the execution of these machine learning models.

Since FHE is currently limited to 8-bit integers, it is necessary to quantize models to make them compatible. As a general rule, the smaller the precision models use, the better the FHE performance.

## Overview

Let $$[\alpha, \beta ]$$ be the range of our value to quantize where $$\alpha$$ is the minimum and $$\beta$$ is the maximum. To quantize a range with floating point values (in $$\mathbb{R}$$) to integer values (in $$\mathbb{Z}$$), we first need to choose the data type that is going to be used. Concrete, the framework used by Concrete-ML, is currently limited to 8-bit integers, so this will be the value used in this example. Knowing the number of bits that can be used for a value in the range $$[\alpha, \beta ]$$, we can compute the `scale` $$S$$ of the quantization:

$$S = \frac{\beta - \alpha}{2^n - 1}$$

where $$n$$ is the number of bits ($$n \leq 8$$). For the sake of example, let's take $$n = 7$$.

In practice, the quantization scale is then $$S = \frac{\beta - \alpha}{127}$$. This means the gap between consecutive representable values cannot be smaller than $$S$$, which, in turn, means there can be a substantial loss of precision. Every interval of length $$S$$ will be represented by a value within the range $$[0..127]$$.

The other important parameter from this quantization schema is the `zero point` $$Z$$ value. This essentially brings the 0 floating point value to a specific integer. If the quantization scheme is asymmetric (quantized values are not centered in 0), the resulting integer will be in $$\mathbb{Z}$$.

$$Z = \mathtt{round} \left(- \frac{\alpha}{S} \right)$$

When using quantized values in a matrix multiplication or convolution, the equations for computing the result become more complex. The IntelLabs distiller quantization documentation provides a more [detailed explanation](https://intellabs.github.io/distiller/algo_quantization.html) of the maths to quantize values and how to keep computations consistent.

Quantization implemented in Concrete-ML is done in two ways:

1. The quantization is done automatically during the model's FHE compilation process. This approach requires little work by the user, but may not be a one-size-fits-all solution for all types of models. The final quantized model is FHE friendly and ready to predict over encrypted data. This approach is done using Post-Training Quantization (PTQ).
1. In some cases (when doing extreme quantization) PTQ is not sufficient to achieve a decent final model accuracy. Concrete-ML offer the possibility for the user to do quantization before compilation to FHE, for example using Quantization-Aware Training (QAT). This can be done by any means, including by using third-party frameworks. In this approach, the user is responsible for implementing a full-integer model respecting the FHE constraints.

## Quantizing data

Concrete-ML has support for quantized ML models as well as quantization tools. The core of this functionality is the conversion of floating point values to integers. This is done using `QuantizedArray` in `concrete.ml.quantization`.

The `QuantizedArray` class takes several arguments that determine how float values are quantized (see the [API reference](../_apidoc/concrete.ml.quantization.rst) for more information):

- `n_bits` that defines the precision of the quantization
- `values` are floating point values that will be converted to integers
- `is_signed` determines if the quantized integer values should allow negative values
- `is_symmetric` determines if the range of floating point values to be quantized should be taken as symmetric around zero

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

## Quantizing machine learning models

Machine learning models are implemented with a diverse set of operations, such as convolution, linear transformations, activation functions and element-wise operations. When working with quantized values, these operations cannot be carried out in the same way as for floating point values. With quantization, it is necessary to re-scale the input and output values of each operation to fit in the quantization domain.

The current version of Concrete only support up to 8-bits integers. This means that any floating point or large precision integer model will need to be converted to an 8-bit equivalent to be able to work with FHE. In most cases, this will require both [quantization](../advanced-topics/quantization.md) and [pruning](../advanced-topics/pruning.md).

If you try to compile a program using more than 8 bits, the compiler will throw an error, as shown in this example:

<!--pytest-codeblocks:skip-->

```python
import concrete.numpy as cnp

def f(x):
    return 42 * x

compiler = cnp.Compiler(f, {"x": "encrypted"})
circuit = compiler.compile(range(8))
```

Compiler output:

```
RuntimeError: max_bit_width of some nodes is too high for the current version of the compiler (maximum must be 7), which is not compatible with:

%0 = x                  # EncryptedScalar<uint3>
%1 = 42                 # ClearScalar<uint6>
%2 = mul(%0, %1)        # EncryptedScalar<uint9>
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ 9 bits is not supported for the time being
return %2
```

Notice that the maximum bit width, determined by the compiler, depends on the inputset passed to the `compile_on_inputset` function. In this case, the error is caused by the input value in the inputset that produces a result whose representation requires 9 bits. This input is the value 8, since 8 * 42 = 336, which is a 9-bit value.

{% hint style="info" %}
You can determine the number of bits necessary to represent an integer value with the formula:

$$n_{\mathsf{bits}}(x) = \mathsf{floor}(\mathsf{log}_2(x)) + 1$$

{% endhint %}

While this can seem like a major limitation, in practice machine learning models have features that use only a limited range of values. For example, if a feature takes a value that is limited to the range $$[1, 2)$$, in floating point this value is represented as $$2^0 * \mathsf{mantissa}$$ where $$\mathsf{mantissa}$$ is a number between 1 and 2. Generic floating point representation can support exponents between -126 and 127, allocating 8 bits to store the exponent. In our case, a single exponent value of 0 is needed. Knowing that, for our range, the exponent can only take a single value out for 253 possible ones. We can thus save the 8 bits allocated to the exponent, reducing the bit width necessary.

For a more practical example, the MNIST classification task consists of taking an image, a 28x28 array containing uint8 values representing a handwritten digit, and predicting whether it belongs to one of 10 classes: the digits from 0 to 9. The output is a one-hot vector which indicates the class to which a particular sample belongs.

The input contains 28x28x8 bits, so 6272 bits of information. In practice, you could still obtain good results on MNIST by thresholding the pixels to {0, 1} and training a model for this new binarized MNIST task. This means that in a real use case where you actually need to perform digit recognition, you could binarize your input on the fly, replacing each pixel with either 0 or 1. In doing so, you use 1 bit per pixel and now only have 784 bits of input data. It also means that if you are doing some accumulation (adding pixel values together), you are going to need accumulators that are smaller (adding 0s and 1s requires less space than adding values ranging from 0 to 255). An example of MNIST classification with a quantized neural network is given in the [CNN advanced example](../deep-learning/fhe_friendly_models.md).

This shows how adapting your data or model parameters can allow you to use models that may require smaller data types (i.e. use less precision) to perform their computations.

{% hint style="info" %}
Binarization is an extreme case of quantization which is introduced [here](../advanced-topics/quantization.md). You can also find further resources on the linked page.
{% endhint %}

{% hint style="info" %}
While applying quantization directly to the input features is mandatory to reduce the effective bit width of computations, a different and complementary approach is dimensionality reduction. This can be accomplished through Principal Component Analysis (PCA) as shown in the [Poisson Regression example](../built-in-models/ml_examples.md)
{% endhint %}

### Limitations for FHE friendly models

Recent quantization literature usually aims to make use of dedicated machine learning accelerators in a mixed setting where a CPU or General Purpose GPU (GPGPU) is also available. Thus, in literature, some floating point computation is assumed to be acceptable. This approach allows us to reach performance similar to those achieved by floating point models. In this popular mixed float-int setting, the input is usually left in floating point. This is also true for the first and last layers, which have more impact on the resulting model accuracy than hidden layers.

The core operations in several models (e.g., neural networks or linear models) are matrix multiplications (matmul) and convolutions, which both compute linear combinations of inputs (encrypted) and weights (in clear). The linear combination operation must be done such that the maximum value of its result requires at most 8 bits of precision.

For example, if you quantize your input and weights with $$n_{\mathsf{weights}}$$, $$n_{\mathsf{inputs}}$$ bits of precision, one can compute the maximum dimensionality of the input and weights before the matmul/convolution result could exceed the 8 bits as such:

$$\Omega = \mathsf{floor} \left( \frac{2^{n_{\mathsf{max}}} - 1}{(2^{n_{\mathsf{weights}}} - 1)(2^{n_{\mathsf{inputs}}} - 1)} \right)$$

where $$n_{\mathsf{max}} = 8$$ is the maximum precision allowed. For example, if we set $$n_{\mathsf{weights}} = 2$$ and $$n_{\mathsf{inputs}} = 2$$ with $$n_{\mathsf{max}} = 8$$, then we have the $$\Omega = 28$$ different inputs/weights are allowed in the linear combination.

Exceeding $$\Omega$$ dimensions in the input and weights, the risk of overflow increases quickly. It may happen that for some distributions of weights and values the computation does not overflow, but the risk increases rapidly with the number of dimensions.

Currently, Concrete-ML computes the number of bits needed for the computation depending on the inputset calibration data and does not allow the overflow to happen, raising an exception as shown previously.

### Model inputs and outputs

The models implemented in Concrete-ML provide features to let the user quantize the input data and dequantize the output data.

Here is a simple example showing how to perform inference, starting from float values and ending up with float values. Note that the FHE engine that is compiled for the ML models does not support data batching.

<!--pytest-codeblocks:skip-->

```python
# Assume quantized_module : QuantizedModule
#        data: numpy.ndarray of float

# Quantization is done in the clear
x_test_q = quantized_module.quantize_input(data)

for i in range(x_test_q.shape[0]):
    # Inputs must have size (1 x N) or (1 x C x H x W), we add the batch dimension with N=1
    x_q = np.expand_dims(x_test_q[i, :], 0)

    # Execute the model in FHE
    out_fhe = quantized_module.forward_fhe.encrypt_run_decrypt(x_q)

    # Dequantization is done in the clear
    output = quantized_module.dequantize_output(out_fhe)

    # For classifiers with multi-class outputs, the arg max is done in the clear
    y_pred = np.argmax(output, 1)
```

The functions `quantize_input` and `dequantize_output` make use of `QuantizedArray` described above. When the ML model `quantized_module` is calibrated, the min and max of the value distributions will be stored and applied to quantize/dequantize new data.

In the following example, `QuantizedArray` is used in a different way, using pre-quantized integer values and having the `scale` and `zero-point` set explicitly from calibration parameters. Once the `QuantizedArray` is constructed, calling `dequant()` will compute the floating point values corresponding to the integer values `qvalues`, which are the output of the `forward_fhe.encrypt_run_decrypt(..)` call.

```python
import numpy

def dequantize_output(self, qvalues: numpy.ndarray) -> numpy.ndarray:
    # .....
    QuantizedArray(
            output_layer.n_bits,
            qvalues,
            value_is_float=False,
            scale=output_layer.output_scale,
            zero_point=output_layer.output_zero_point,
        ).dequant()
    # ....
```

### Adding new quantized layers

Intermediary values computed during model inference might need to be re-scaled into the quantized domain of a subsequent model operator. For example, the output of a convolution layer in a neural network might have values that are 8 bits wide, with the next convolutional layer requiring that its inputs are at most 2 bits wide. In the non-encrypted realm, this implies that we need to make use of floating point operations. To make this work with integers as required by FHE, Concrete-ML uses a table lookup (TLU), which is a [way to encode univariate functions in FHE](https://docs.zama.ai/concrete-numpy/tutorials/table_lookup). Table lookups are expensive in FHE, and so should only be used when necessary.

The operations done by the activation function of a previous layer and additional re-scaling to the new quantized domain, which are all floating point operations, [can be fused to a single TLU](https://docs.zama.ai/concrete-numpy/tutorials/floating_points). Concrete-ML implements quantized operators that perform this fusion, significantly reducing the number of TLUs necessary to perform inference.

There are 3 types of operators:

1. Operators that perform linear combinations of encrypted and constant (clear) values. For example: matrix multiplication, convolution, addition
1. Operators that perform element-wise operations between two encrypted tensors. For example: addition
1. Element-wise, fixed-function operators which can be: addition with a constant, activation functions

The following example shows how to use the `_prepare_inputs_with_constants` helper function with `quantize_actual_values=True` to apply the quantization function to the input data of the `Gemm` operator. Since the quantization function uses floats and a non-linear function (`round`), a TLU will automatically be generated together with quantization.

<!--pytest-codeblocks:cont-->

```python
from concrete.ml.quantization import QuantizedOp, QuantizedArray

class QuantizedGemm(QuantizedOp):
    """Quantized Gemm op."""

    _impl_for_op_named: str = "Gemm"
    def q_impl(
        self,
        *q_inputs: QuantizedArray,
        **attrs,
    ) -> QuantizedArray:

        # ...

        prepared_inputs = self._prepare_inputs_with_constants(
            *q_inputs, calibrate=False, quantize_actual_values=True
        )

        q_input: QuantizedArray = prepared_inputs[0]
        q_weights: QuantizedArray = prepared_inputs[1]
        q_bias: Optional[QuantizedArray] = (
            None if len(prepared_inputs) == 2 or beta == 0 else prepared_inputs[2]
        )
```

TLU generation for element-wise operations can be delegated to Concrete-Numpy directly by calling the function's corresponding NumPy implementation, as defined in [ops_impl.py](../_apidoc/concrete.ml.onnx.html#module-concrete.ml.onnx.ops_impl).

<!--pytest-codeblocks:cont-->

```python
class QuantizedAbs(QuantizedOp):
    """Quantized Abs op."""

    _impl_for_op_named: str = "Abs"
```

## Resources

- IntelLabs distiller explanation of quantization: [Distiller documentation](https://intellabs.github.io/distiller/algo_quantization.html)
- Lei Mao's blog on quantization: [Quantization for Neural Networks](https://leimao.github.io/article/Neural-Networks-Quantization/)
- Google paper on neural network quantization and integer-only inference: [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
