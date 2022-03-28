```{note}
FIXME: Andrei to do
```

# Quantization

In this section we detail some usage of [quantization](../../user/explanation/quantization.md) as implemented in **Concrete**.

## Quantization Basics

**Concrete ML** implements some basic concepts of quantization. The very basic purpose of it is to convert floating point values to integers. We can apply such conversion using `QuantizedArray` available in `concrete.ml.quantization`.

`QuantizedArray` takes 2 arguments:

- `n_bits` that defines the precision of the quantization. Currently, `n_bits` is limited to 7, due to some **Concrete Library** limits.
- `values` that will be converted to integers

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

## Neural networks in the Quantized Realm

Neural networks are implemented with a diverse set of operations, such as convolution, linear transformations, activation functions and element-wise operations. When working with quantized values, these operations can not be carried out the same way as for floating point values. With quantization it is necessary to re-scale the input and output values of each operation to fit in the quantization domain.

Re-scaling raw input values to the quantized domain implies that we need to make use of floating point operations. In the FHE setting where we only work with integers, this could be a problem, but luckily, the FHE implementation behind **Concrete ML** provides a workaround. We essentially make use of a [table lookup](https://docs.zama.ai/concrete-numpy/stable/user/tutorial/table_lookup.html) which is later translated into a [PBS](https://whitepaper.zama.ai).

Of course, having a PBS for every quantized addition isn't recommended for computational cost reasons. Also, **Concrete ML** allows PBS only for univariate operations (i.e. matrix multiplication can't be done in a PBS). Therefore, our quantized modules split the computation of floating point values and unsigned integers as it is currently done in `concrete.ml.quantization.QuantizedLinear`.

The above operations are all implemented in **Concrete ML** and transparent to the user via our Quantized Modules.

**Concrete ML** allows you to convert numpy operations to their FHE counterparts. This essentially opens the door to any python computing framework such as [PyTorch](https://pytorch.org/). **Concrete ML** implements a torch to numpy converter that makes it easy for the user to use a torch model.

First we define a model:

<!--pytest-codeblocks:cont-->

```python
from torch import nn
import torch
class LogisticRegression(nn.Module):
    """LogisticRegression with Torch"""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=14, out_features=1)
        self.sigmoid1 = nn.Sigmoid()


    def forward(self, x):
        """Forward pass."""
        out = self.fc1(x)
        out = self.sigmoid1(out)
        return out


torch_model = LogisticRegression()
```

We then convert this model to numpy only operations:

<!--pytest-codeblocks:cont-->

```python
from concrete.ml.torch import NumpyModule

numpy_input = numpy.zeros(shape=(10, 14))
dummy_input = torch.from_numpy(numpy_input).float()
numpy_model = NumpyModule(torch_model, dummy_input)
```

The `NumpyModule` allows us to runs inference as for a `nn.Module`. Here, the prediction of the numpy module should be exactly the same.

We can then quantize the numpy module with `PostTrainingAffineQuantization` as follows:

<!--pytest-codeblocks:cont-->

```python
from concrete.ml.quantization import PostTrainingAffineQuantization

numpy_input = numpy.random.uniform(
    -1, 1, size=(10, 14)
)  # some input with 14 features to calibrate the quantization
n_bits = 2  # number of bits of precision for the weights, activation, inputs and outputs.
post_training_quant = PostTrainingAffineQuantization(n_bits, numpy_model)
quantized_numpy_module = post_training_quant.quantize_module(numpy_input)
```

Here, the quantized model takes a quantized array and runs inference in the quantized paradigm.

We can then easily verify that all models give similar predictions. Obviously, the `n_bits` chosen may adversely affect the prediction of the `quantized_numpy_module`. You can try increasing this parameter to see the effect on your model but keep in mind that the compilation will require all the values of your network to be less than 7 bits of precision.

<!--pytest-codeblocks:cont-->

```python
# Quantize input
qinput = quantized_numpy_module.quantize_input(numpy_input)
print("qinput = ", qinput.flatten())
# qinput =  [2 1 1 2 0 0 0 2 2 2 2 2 1 2 0 1 0 2 1 1 0 2 1 1 0 1 1 1 2 2 1 1 2 0 2 2 0
# 0 0 1 1 1 2 0 0 0 1 0 1 0 0 0 1 0 0 1 2 0 2 0 2 1 2 1 2 0 0 0 0 0 0 1 0 2
# 1 0 1 0 1 2 0 2 0 2 0 0 1 0 2 0 2 0 2 2 0 1 1 1 0 2 1 2 2 0 2 1 2 1 2 2 2
# 1 2 1 1 1 0 0 1 0 1 1 0 0 1 1 1 1 1 1 2 1 1 2 2 2 0 2 2 3]

prediction = quantized_numpy_module(qinput)
print("prediction = ", prediction.flatten())
# prediction =  [2 2 0 2 3 2 3 2 2 2]

dequant_manually_prediction = quantized_numpy_module.dequantize_output(prediction)
print("dequant_manually_prediction = ", dequant_manually_prediction.flatten())
# dequant_manually_prediction =  [0.50702845 0.50702845 0.38027134 0.50702845 0.570407   0.50702845
# 0.570407   0.50702845 0.50702845 0.50702845]

# Forward and Dequantize to get back to real values
dequant_prediction = quantized_numpy_module.forward_and_dequant(qinput)
print("dequant_prediction = ", dequant_prediction.flatten())
# dequant_prediction =  [0.50702845 0.50702845 0.38027134 0.50702845 0.570407   0.50702845
# 0.570407   0.50702845 0.50702845 0.50702845]

# Both dequant prediction should be equal
```

```{warning}
The current implementation of the framework parses the layers in the order of their definition in the nn.Module. Thus, the order of instantiation of the layers in the constructor (init function) is crucial for the conversion to numpy to work properly.
```

```{warning}
Do not reuse a layer or an activation multiple times in the forward (i.e. self.sigmoid for each layer activation) and always place them at the correct position (the order of appearance in the forward function) in the init function.
```

It is now possible to compile the `quantized_numpy_module`. Details on how to compile the model are available in the [torch compilation documentation](../../user/howto/simple_example_torch.md).

## Building your own QuantizedModule

**Concrete ML** also offers the possibility to build your own models and use them in the FHE settings. The `QuantizedModule` is a very simple abstraction that allows to create any model using the available operators:

- QuantizedSigmoid, the quantized version of `nn.Sigmoid`
- QuantizedLinear, the quantized version of `nn.Linear`
- QuantizedReLU6, the quantized version of `nn.ReLU6`

A well detailed example is available for a [Linear Regression](../../user/advanced_examples/LinearRegression.ipynb).

## Future releases

Currently, the quantization is only available via `PostTrainingAffineQuantization` which is a [popular](https://arxiv.org/pdf/1712.05877.pdf) approach for quantization but has some constraints.

In future releases we plan to offer the possibility to the user to apply quantization beforehand and convert the model directly to our `QuantizedModule`. This will allow users to take advantage of Quantization Aware Training (QAT) that allow neural networks to reach better accuracies.
