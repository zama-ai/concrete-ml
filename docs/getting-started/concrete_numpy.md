# Key Concepts

Concrete-ML is built on top of Concrete-Numpy, which enables Numpy programs to be converted into FHE circuits.

The lifecycle of a Concrete ML program is as follows:

1. **Training.** A model is trained using plaintext inputs.
1. **Quantization.** The trained model is then converted into an integer equivalent using quantization, which can happen either during training (Quantization-Aware Training) or after training (Post-Training Quantization).
1. **Compilation.** Once the model is quantized, it is compiled using Concrete's FHE compiler to produce an equivalent FHE circuit. This circuit is represented as an MLIR program consisting of low level cryptographic operations. You can read more about FHE compilation [here](../advanced-topics/compilation.md), MLIR [here](https://mlir.llvm.org/) and about the low-level Concrete library [here](https://docs.zama.ai/concrete-core).
1. **Inference.** The compiled model can then be deployed to a server and used to run private inference on encrypted  inputs. You can see some examples [here](../built-in-models/ml_examples.md).

Here is an example for a simple linear regression model:

```python
import numpy
from concrete.numpy.compilation import compiler

# Let's assume Quantization has been applied and we are left with integers only.
# This is essentially the work of Concrete-ML

# Some parameters (weight and bias) for our model taking a single feature
w = [2]
b = 2

# The function that implements our model
@compiler({"x": "encrypted"})
def linear_model(x):
    return w @ x + b

# A representative inputset is needed to compile the function
# (used for tracing)
n_bits_input = 2
inputset = numpy.arange(0, 2**n_bits_input).reshape(-1, 1)
circuit = linear_model.compile(inputset)

# Use the API to get the maximum bitwidth in the circuit
max_bitwidth = circuit.graph.maximum_integer_bit_width()
print("Max bitwidth = ", max_bitwidth)
# Max bitwidth =  4

# Test our FHE inference
circuit.encrypt_run_decrypt(numpy.array([3]))
# 8

# Print the graph of the circuit
print(circuit)
# %0 = 2                     # ClearScalar<uint2>
# %1 = [2]                   # ClearTensor<uint2, shape=(1,)>
# %2 = x                     # EncryptedTensor<uint2, shape=(1,)>
# %3 = matmul(%1, %2)        # EncryptedScalar<uint3>
# %4 = add(%3, %0)           # EncryptedScalar<uint4>
# return %4
```

At this stage, we have everything we need to deploy the model using `Client` and `Server` from `concrete.numpy`. Please refer to the [Concrete-Numpy implementation](https://docs.zama.ai/concrete-numpy) for more information on the deployment.

## Quantization

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

## Quantized model accuracy

Quantization and dimensionality reduction reduce the bit width required to run the model and increase execution speed. These two tools are necessary to make models compatible with FHE constraints.

However, quantization and, especially, binarization, induce a loss in the accuracy of the model since its representation power is diminished. Carefully choosing a quantization approach for model parameters can alleviate accuracy loss, all the while allowing compilation to FHE.

The quantization of model parameters and model inputs is illustrated in the advanced examples for [Linear and Logistic Regressions](../built-in-models/ml_examples.md). Note that different quantization parameters are used for inputs and for model weights.

### Limitations for FHE friendly neural networks

Recent quantization literature usually aims to make use of dedicated machine learning accelerators in a mixed setting where a CPU or General Purpose GPU (GPGPU) is also available. Thus, in literature, some floating point computation is assumed to be acceptable. This approach allows us to reach performance similar to those achieved by floating point models. In this popular mixed float-int setting, the input is usually left in floating point. This is also true for the first and last layers, which have more impact on the resulting model accuracy than hidden layers.

However, in Concrete-ML, to respect FHE constraints, the inputs, the weights and the accumulator must all be represented with integers of a maximum of 8 bits.

Thus, in Concrete-ML, we also quantize the input data and network output activations in the same way as the rest of the network: everything is quantized to a specific number of bits. It turns out that the number of bits used for the input or the output of any activation function is crucial to comply with the constraint on accumulator width.

The core operations in neural networks are matrix multiplications (matmul) and convolutions, which both compute linear combinations of inputs (encrypted) and weights (in clear). The linear combination operation must be done such that the maximum value of its result requires at most 8 bits of precision.

For example, if you quantize your input and weights with $$n_{\mathsf{weights}}$$, $$n_{\mathsf{inputs}}$$ bits of precision, one can compute the maximum dimensionality of the input and weights before the matmul/convolution result could exceed the 8 bits as such:

$$\Omega = \mathsf{floor} \left( \frac{2^{n_{\mathsf{max}}} - 1}{(2^{n_{\mathsf{weights}}} - 1)(2^{n_{\mathsf{inputs}}} - 1)} \right)$$

where $$n_{\mathsf{max}} = 8$$ is the maximum precision allowed. For example, if we set $$n_{\mathsf{weights}} = 2$$ and $$n_{\mathsf{inputs}} = 2$$ with $$n_{\mathsf{max}} = 8$$, then we have the $$\Omega = 28$$ different inputs/weights are allowed in the linear combination.

Exceeding $$\Omega$$ dimensions in the input and weights, the risk of overflow increases quickly. It may happen that for some distributions of weights and values the computation does not overflow, but the risk increases rapidly with the number of dimensions.

Currently, Concrete-ML computes the number of bits needed for the computation depending on the inputset calibration data and does not allow the overflow to happen, raising an exception as shown previously.


