# Quantization

{% hint style='info' %}
from [Wikipedia](https://en.wikipedia.org/wiki/Quantization):

> Quantization is the process of constraining an input from a continuous or otherwise large set of values (such as the real numbers) to a discrete set (such as the integers).
> {% endhint %}

## Why is it needed?

Modern computing has been using data types that are 32 or 64 bits wide for many years, for both integers and floating point values. Even bigger data types are available or can be constructed easily. However, due to the costly nature of FHE computations (see [limits of FHE](https://docs.zama.ai/concrete-numpy/stable/user/explanation/fhe_and_framework_limits.html)), using such types with FHE is impractical (or plain impossible) if we are to execute computations in a reasonable amount of time.

## The gist of quantization

The basic idea of quantization is to take a **range of values** that are represented by a _large_ data type and represent them using a single value of a _smaller_ data type. This means that some accuracy in the representation is lost (e.g. a simple approach is to eliminate least-significant bits), but, in many cases in machine learning, it is possible to adapt the models to give meaningful results while using these smaller data types. This significantly reduces the number of bits necessary for intermediary results during the execution of these machine learning models.

## Quantization in practice

Let's first define some notations. Let \$$ [\alpha, \beta ] $$ be the range of our value to quantize where $$ \alpha $$ is the minimum and $$ \beta $\$ is the maximum.

To quantize a range with floating point values (in \$$ \mathbb{R} $$) to integer values (in $$ \mathbb{Z} $$), we first need to choose the data type that is going to be used. **Concrete-Library**, the backend library used by **Concrete-ML**, is currently limited to 7-bit integers, so we'll use this value for the example. Knowing the number of bits that can be used, for a value in the range $$ [\alpha, \beta ] $$, we can compute the `scale` $$ S $\$ of the quantization:

$$ S =  \frac{\beta - \alpha}{2^n - 1} $$

where \$$ n $\$ is the number of bits (here, 7).

In practice, the quantization scale is then \$$ S = \frac{\beta - \alpha}{127} $$. This means the gap between consecutive representable values cannot be smaller than $$ S $$, which, in turn, means there can be a substantial loss of precision. Every interval of length $$ S $$ will be represented by a value within the range $$ [0..127] $\$.

The other important parameter from this quantization schema is the `zero point` \$$ Z $$ value. This essentially brings the 0 floating point value to a specific integer. If the quantization scheme is asymmetric (quantized values are not centered in 0), the resulting integer will be in $$ \mathbb{Z} $\$.

$$ Z = \mathtt{round} \left(- \frac{\alpha}{S} \right) $$

When using quantized values in a matrix multiplication or convolution, the equations for computing the result are more involved. The IntelLabs distiller quantization documentation provides a more [detailed explanation](https://intellabs.github.io/distiller/algo_quantization.html) of the maths to quantize values and how to keep computations consistent.

Regarding quantization in **Concrete-ML** and FHE compilation, it is important to understand the difference between two approaches:

1. The quantization is done automatically during the model compilation stage (inside our framework). This approach requires little work by the user, but may not be a one-size-fits-all solution for all types of models that a user may want to implement.
1. The quantization is done by the user, before compilation to FHE; notably, the quantization is completely controlled by the user, and can be done by any means, including by using third-party frameworks. In this approach, the user is responsible for implementing their models directly with NumPy.

For the moment, the first method is applicable through the tools provided by in **Concrete-ML**, and the models implemented in our framework make use of this approach. When quantization is only performed in the compilation stage, the model training stage does not
take into account that the model will be quantized. This setting is called Post-Training Quantization (PTQ), and this is the approach
currently taken in **Concrete-ML**. PTQ is effective for moderate bit widths, such as 7-8 bits per weight and activation, but, for a model to be compatible with FHE constraints, we must quantize these values to as few as 2-3 bits. Thus, for models with more than a few neurons per layer, PTQ is not the optimal solution, and we plan to implement a more performant approach called Quantization Aware Training in the near future.

We detail the use of quantization within **Concrete-ML** [here](../../dev/explanation/use_quantization.md).

## Resources

- IntelLabs distiller explanation of quantization: [Distiller documentation](https://intellabs.github.io/distiller/algo_quantization.html)
- Lei Mao's blog on quantization: [Quantization for Neural Networks](https://leimao.github.io/article/Neural-Networks-Quantization/)
- Google paper on neural network quantization and integer-only inference: [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference](https://arxiv.org/abs/1712.05877)
