# Quantization

```{note}
FIXME: two .md's have been concatenated, to be fusionned well
```

```{note}
from [Wikipedia](https://en.wikipedia.org/wiki/Quantization):

> Quantization is the process of constraining an input from a continuous or otherwise large set of values (such as the real numbers) to a discrete set (such as the integers).
```

## Why is it needed?

Modern computing has been using data types that are 32 or 64 bits wide for many years, for both integers and floating point values. Even bigger data types are available or can be constructed easily. However, due to the costly nature of FHE computations (see [limits of FHE](https://docs.zama.ai/concrete-numpy/stable/user/explanation/fhe_and_framework_limits.html)), using such types with FHE is impractical (or plain impossible) if we are to execute computations in a reasonable amount of time.

## The gist of quantization

The basic idea of quantization is to take a **range of values** that are represented by a _large_ data type and represent them using a single value of a _smaller_ data type. This means that some accuracy in the representation is lost (e.g. a simple approach is to eliminate least-significant bits), but, in many cases in machine learning, it is possible to adapt the models to give meaningful results while using these smaller data types. This significantly reduces the number of bits necessary for intermediary results during the execution of these machine learning models.

## Quantization in practice

Let's first define some notations. Let $ [\alpha, \beta ] $ be the range of our value to quantize where $ \alpha $ is the minimum and $ \beta $ is the maximum.

To quantize a range with floating point values (in $ \mathbb{R} $) to integer values (in $ \mathbb{Z} $), we first need to choose the data type that is going to be used. **Concrete-Library**, the backend library used by **Concrete-ML**, is currently limited to 8-bit integers, so we'll use this value for the example. Knowing the number of bits that can be used, for a value in the range $ [\alpha, \beta ] $, we can compute the `scale` $ S $ of the quantization:

$$ S =  \frac{\beta - \alpha}{2^n - 1} $$

where $ n $ is the number of bits ($n \leq 8$). For the sake of example, let's take $n = 7$.

In practice, the quantization scale is then $ S = \frac{\beta - \alpha}{127} $. This means the gap between consecutive representable values cannot be smaller than $ S $, which, in turn, means there can be a substantial loss of precision. Every interval of length $ S $ will be represented by a value within the range $ [0..127] $.

The other important parameter from this quantization schema is the `zero point` $ Z $ value. This essentially brings the 0 floating point value to a specific integer. If the quantization scheme is asymmetric (quantized values are not centered in 0), the resulting integer will be in $ \mathbb{Z} $.

$$ Z = \mathtt{round} \left(- \frac{\alpha}{S} \right) $$

When using quantized values in a matrix multiplication or convolution, the equations for computing the result are more involved. The IntelLabs distiller quantization documentation provides a more [detailed explanation](https://intellabs.github.io/distiller/algo_quantization.html) of the maths to quantize values and how to keep computations consistent.

Regarding quantization in **Concrete-ML** and FHE compilation, it is important to understand the difference between two approaches:

1. The quantization is done automatically during the model compilation stage (inside our framework). This approach requires little work by the user, but may not be a one-size-fits-all solution for all types of models that a user may want to implement.
1. The quantization is done by the user, before compilation to FHE; notably, the quantization is completely controlled by the user, and can be done by any means, including by using third-party frameworks. In this approach, the user is responsible for implementing their models directly with NumPy.

For the moment, the first method is applicable through the tools provided by in **Concrete-ML**, and the models implemented in our framework make use of this approach. When quantization is only performed in the compilation stage, the model training stage does not
take into account that the model will be quantized. This setting is called Post-Training Quantization (PTQ), and this is the approach
currently taken in **Concrete-ML**. PTQ is effective for moderate bit widths, such as 7-8 bits per weight and activation, but, for a model to be compatible with FHE constraints, we must quantize these values to as few as 2-3 bits. Thus, for models with more than a few neurons per layer, PTQ is not the optimal solution, and we plan to implement a more performant approach called Quantization Aware Training in the near future.

We detail the use of quantization within **Concrete-ML** [here](quantization.md).

## Resources

- IntelLabs distiller explanation of quantization: [Distiller documentation](https://intellabs.github.io/distiller/algo_quantization.html)
- Lei Mao's blog on quantization: [Quantization for Neural Networks](https://leimao.github.io/article/Neural-Networks-Quantization/)
- Google paper on neural network quantization and integer-only inference: [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)

## FIXME

In this section, we detail the usage of [quantization](quantization.md) in **Concrete-ML**.

## Quantizing data

Since quantization is [necessary to make ML models work in FHE](fhe_constraints.md), **Concrete-ML** implements quantized ML models to facilitate usage, but also exposes some quantization tools. The core of this functionality is the conversion of floating point values to integers, following the techniques described [here](quantization.md). We can apply this conversion using `QuantizedArray`, available in `concrete.ml.quantization`.

The `QuantizedArray` class takes several arguments that determine how float values are quantized:

- `n_bits` that defines the precision of the quantization
- `values` are floating point values that will be converted to integers
- `is_signed` determines if the quantized integer values should allow negative values
- `is_symmetric` determines if the range of floating point values to be quantized should be taken as symmetric around zero

Please see the [API reference](_apidoc/concrete.ml.quantization.rst) for more information.

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
print("q_A.scale = ", q_A.scale)
# 0.018274684777173276, the scale S.
print("q_A.zero_point = ", q_A.zero_point)
# 26, the zero point Z.
print("q_A.dequant() = ", q_A.dequant())
# array([ 0.20102153,  0.85891018,  0.40204307,  0.18274685, -0.31066964,
#         0.58478991, -0.25584559,  1.57162289,  1.84574316, -0.4751418 ])
# Dequantized values.
```

We can also use symmetric quantization, where the integer values are centered around 0 and may, thus,
take negative values.

<!--pytest-codeblocks:cont-->

```python
q_A = QuantizedArray(3, A)
print("Unsigned: q_A.qvalues = ", q_A.qvalues)
print("q_A.zero_point = ", q_A.zero_point)
# Unsigned: q_A.qvalues =  [2 4 2 2 0 3 0 6 7 0]
# q_A.zero_point =  1

q_A = QuantizedArray(3, A, is_signed=True, is_symmetric=True)
print("Signed Symmetric: q_A.qvalues = ", q_A.qvalues)
print("q_A.zero_point = ", q_A.zero_point)
# Signed Symmetric: q_A.qvalues =  [ 0  1  1  0  0  1  0  3  3 -1]
# q_A.zero_point =  0
```

## Machine learning models in the quantized realm

Machine learning models are implemented with a diverse set of operations, such as convolution, linear transformations, activation functions and element-wise operations. When working with quantized values, these operations cannot be carried out in the same way as for floating point values. With quantization, it is necessary to re-scale the input and output values of each operation to fit in the quantization domain.

### Model inputs and outputs

The ML models implemented in **Concrete-ML** provide features to let the user quantize the input data and dequantize the output data.

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

If we are to examine the operations done by `quantize_input` and `dequantize_output`, we will see
usage of the `QuantizedArray` described above. When the ML model `quantized_module` is calibrated, the
min and max of the value distributions will be recorded, and these are then applied to quantize/dequantize new data.

Here, a different usage of `QuantizedArray` is shown, where it is constructed from quantized integer values
and the `scale` and `zero-point` are set explicitly from calibration parameters. Once the `QuantizedArray` is constructed, calling `dequant()` will compute the floating point values corresponding to the integer values `qvalues`, which are the output of the
`forward_fhe.encrypt_run_decrypt(..)` call.

<!--pytest-codeblocks:skip-->

```python
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

Intermediary values computed during model inference might need to be re-scaled into the quantized domain of a subsequent model operator. For example, the output of a convolution layer in a neural network might have values that are 8 bits wide, but the next convolutional layer requires that its inputs are, at most, 2 bits wide. In the non-encrypted realm, this implies that we need to make use of floating point operations. In the FHE setting, where we only work with integers, this could be a problem, but, luckily, the FHE implementation behind **Concrete-ML** provides a solution. We essentially make use of a [table lookup](https://docs.zama.ai/concrete-numpy/stable/user/tutorial/table_lookup.html), which is later translated into a [Programmable Bootstrap (PBS)](https://whitepaper.zama.ai).

Of course, having a PBS for every quantized addition isn't recommended for computational cost reasons. Also, a PBS is currently only allowed for univariate operations (i.e. matrix multiplication can't be done in a PBS). Therefore, our quantized modules split the computation of floating point values and unsigned integers. Moreover, the operations done by the activation function of a previous layer and additional re-scaling to the new quantized domain, which are all floating point operations, [can be fused to a single TLU](https://docs.zama.ai/concrete-numpy/stable/dev/explanation/float-fusing.html). **Concrete-ML** implements quantized operators that perform this fusion, significantly reducing the number of TLUs necessary to perform inference.

We can distinguish three types of operators:

1. Operators that perform linear combinations of encrypted and constant (clear) values. For example: matrix multiplication, convolution, addition
1. Operators that perform element-wise operations between two encrypted tensors. For example: addition
1. Element-wise, fixed-function operators which can be: addition with a constant, activation functions

In the first category, we will find operators such as `Gemm`, which will quantize their inputs. Notice
that here we use the `_prepare_inputs_with_constants` helper function, with `quantize_actual_values=True`,
to apply the quantization function to the input data.
The quantization function operators using floating point and a non-linear
function, `round`, will thus produce a TLU, together with any preceding floating point operations.

<!--pytest-codeblocks:skip-->

```python
class QuantizedGemm(QuantizedOp):
    """Quantized Gemm op."""

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

For element-wise operations with a fixed function, we simply let **Concrete-Numpy** generate a TLU. To do so, we just need to
give this function the corresponding NumPy implementation, which must be defined in [ops_impl.py](_apidoc/concrete.ml.onnx.html#module-concrete.ml.onnx.ops_impl).

<!--pytest-codeblocks:skip-->

```python
class QuantizedAbs(QuantizedOp):
    """Quantized Abs op."""

    _impl_for_op_named: str = "Abs"
```
