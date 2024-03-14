# Quantization

Quantization is the process of constraining an input from a continuous or otherwise large set of values (such as real numbers) to a discrete set (such as integers).

This means that some accuracy in the representation is lost (e.g., a simple approach is to eliminate least-significant bits). In many cases in machine learning, it is possible to adapt the models to give meaningful results while using these smaller data types. This significantly reduces the number of bits necessary for intermediary results during the execution of these machine learning models.

Since FHE is currently limited to 16-bit integers, it is necessary to quantize models to make them compatible. As a general rule, the smaller the bit-width of integer values used in models, the better the FHE performance. This trade-off should be taken into account when designing models, especially neural networks.

## Overview of quantization in Concrete ML

Quantization implemented in Concrete ML is applied in two ways:

1. Built-in models apply quantization internally and the user only needs to configure some quantization parameters. This approach requires little work by the user but may not be a one-size-fits-all solution for all types of models. The final quantized model is FHE-friendly and ready to predict over encrypted data. In this setting, Post-Training Quantization (PTQ) is used for linear models, data quantization is used for tree-based models and, finally, Quantization Aware Training (QAT) is included in the built-in neural network models.
1. For custom neural networks with more complex topology, obtaining FHE-compatible models with good accuracy requires QAT. Concrete ML offers the possibility for the user to perform quantization before compiling to FHE. This can be achieved through a third-party library that offers QAT tools, such as [Brevitas](https://github.com/Xilinx/brevitas) for PyTorch. In this approach, the user is responsible for implementing a full-integer model, respecting FHE constraints. Please refer to the [advanced QAT tutorial](../deep-learning/fhe_friendly_models.md) for tips on designing FHE neural networks.

{% hint style="info" %}
While Concrete ML quantizes machine learning models, the data that the client has is often in floating point. Concrete ML models provide APIs to quantize inputs and de-quantize outputs.

Note that the floating point input is quantized in the clear, meaning it is converted to integers before being encrypted. The model's outputs are also integers and decrypted before de-quantization.
{% endhint %}

## Basics of quantization

Let $$[\alpha, \beta ]$$ be the range of a value to quantize where $$\alpha$$ is the minimum and $$\beta$$ is the maximum. To quantize a range of floating point values (in $$\mathbb{R}$$) to integer values (in $$\mathbb{Z}$$), the first step is to choose the data type that is going to be used. Many ML models work with weights and activations represented as 8-bit integers, so this will be the value used in this example. Knowing the number of bits that can be used for a value in the range $$[\alpha, \beta ]$$, the `scale` $$S$$ can be computed :

$$S = \frac{\beta - \alpha}{2^n - 1}$$

where $$n$$ is the number of bits ($$n \leq 8$$). In the following, $$n = 8$$ is assumed.

In practice, the quantization scale is then $$S = \frac{\beta - \alpha}{255}$$. This means the gap between consecutive representable values cannot be smaller than $$S$$, which, in turn, means there can be a substantial loss of precision. Every interval of length $$S$$ will be represented by a value within the range $$[0..255]$$.

The other important parameter from this quantization schema is the `zero point` $$Z_p$$ value. This essentially brings the 0 floating point value to a specific integer. If the quantization scheme is asymmetric (quantized values are not centered in 0), the resulting $$Z_p$$ will be in $$\mathbb{Z}$$.

$$Z_p = \mathtt{round} \left(- \frac{\alpha}{S} \right)$$

When using quantized values in a matrix multiplication or convolution, the equations for computing the result become more complex. The IntelLabs Distiller documentation provides a more [detailed explanation](https://intellabs.github.io/distiller/algo_quantization.html) of the maths used to quantize values and how to keep computations consistent.

### Quantization special cases

Machine learning acceleration solutions are often based on integer computation of activations. To make quantization computations hardware-friendly, a popular approach is to ensure that scales are powers-of-two, which allows the replacement of the division in the equations above with a shift-right operation. TFHE also has a fast primitive for right bit-shift that enables acceleration in the special case of power-of-two scales.

### Configuring model quantization parameters

Built-in models provide a simple interface for configuring quantization parameters, most notably the number of bits used for inputs, model weights, intermediary values, and output values.

For [linear models](../built-in-models/linear.md), the quantization is done post-training. Thus, the model is trained in floating point, and then, the best integer weight representations are found, depending on the distribution of inputs and weights. For these models, the user selects the value of the `n_bits` parameter.

For linear models, `n_bits` is used to quantize both model inputs and weights. Depending on the number of features, you can use a single integer value for the `n_bits` parameter (e.g., a value between 2 and 7). When the number of features is high, the `n_bits` parameter should be decreased if you encounter compilation errors. It is also possible to quantize inputs and weights with different numbers of bits by passing a dictionary to `n_bits` containing the `op_inputs` and `op_weights` keys.

For [tree-based models](../built-in-models/tree.md), the training and test data is quantized. The maximum accumulator bit-width for a model trained with `n_bits=n` for this type of model is known beforehand: It will need `n+1` bits. Through experimentation, it was determined that, in many cases, a value of 5 or 6 bits gives the same accuracy as training in floating point and values above `n=7` do not increase model performance (but rather induce a strong slowdown).

{% hint style="info" %}
Tree-based models can directly control the accumulator bit-width used. If 6 or 7 bits are not sufficient to obtain good accuracy on your data-set, one option is to use an ensemble model (RandomForest or XGBoost) and increase the number of trees in the ensemble. This, however, will have a detrimental impact on FHE execution speed.
{% endhint %}

For built-in [neural networks](../built-in-models/neural-networks.md), several linear layers are used. Thus, the outputs of a layer are used as inputs to a new layer. Built-in neural networks use Quantization Aware Training. The parameters controlling the maximum accumulator bit-width are the number of weights and activation bits ( `module__n_w_bits`, `module__n_a_bits` ), but also the pruning factor. This factor is determined automatically by specifying a desired accumulator bit-width `module__n_accum_bits` and, optionally, a multiplier factor, `module__n_hidden_neurons_multiplier`.

{% hint style="info" %}
For built-in **neural networks**, the maximum accumulator bit-width cannot be precisely controlled. To use many input features and a high number of bits is beneficial for model accuracy, but it can conflict with the 16-bit accumulator constraint. Finding the best quantization parameters to maximize accuracy, while keeping the accumulator size down, can only be accomplished through experimentation.
{% endhint %}

### Quantizing model inputs and outputs

The models implemented in Concrete ML provide features to let the user quantize the input data and de-quantize the output data.

In a client/server setting, the client is responsible for quantizing inputs before sending them, encrypted, to the server. The client must then de-quantize the encrypted integer results received from the server. See the [Production Deployment](../guides/client_server.md) section for more details.

Here is a simple example showing how to perform inference, starting from float values and ending up with float values. The FHE engine that is compiled for ML models does not support data batching.

<!--pytest-codeblocks:skip-->

```python
# Assume 
#   quantized_module : QuantizedModule
#   x: numpy.ndarray (of float)

# Quantization is done in the clear
x_q = quantized_module.quantize_input(x)

# Forward in FHE (here with simulation)
q_y_proba = quantized_module.quantized_forward(x_q, fhe="simulate")

# De-quantization is done in the clear
y_proba = quantized_module.dequantize_output(q_y_proba)

# For classifiers with multi-class outputs, the arg max is done in the clear
y_pred = np.argmax(y_proba, 1)
```

Alternatively, the `forward` method groups the quantization, FHE execution and de-quantization steps all together.

<!--pytest-codeblocks:skip-->

```python
# Assume 
#   quantized_module : QuantizedModule
#   x: numpy.ndarray (of float)

# Forward in FHE (here with simulation). Quantization and de-quantization steps are still done in 
# the clear 
y_proba = quantized_module.forward(x, fhe="simulate")

# For classifiers with multi-class outputs, the arg max is done in the clear
y_pred = np.argmax(y_proba, 1)
```

## Resources

- IntelLabs distiller explanation of quantization: [Distiller documentation](https://intellabs.github.io/distiller/algo_quantization.html)
