# Use **Concrete ML** ONNX Operator Support for FHE Model Conversion

Internally, **Concrete ML** uses [ONNX](https://github.com/onnx/onnx) operators as Intermediate Representation (or IR) for manipulating machine learning models produced through export for [torch](https://github.com/pytorch/pytorch), [Hummingbird](https://github.com/microsoft/hummingbird) and [skorch](https://github.com/skorch-dev/skorch). As ONNX is becoming the standard exchange format for neural networks, this allows **Concrete ML** to be flexible while also making model representation manipulation quite easy. In addition, it allows for straight-forward mapping to numpy operators, supported by **Concrete Numpy** to use the **Concrete** stack FHE conversion capabilities.

Here we list the operators that are supported as well as the operators that have a quantized version, which should allow you to perform automatic Post Training Quantization (PTQ) of your models.

```{note}
Please note that due to the current precision constraints from the **Concrete** stack, PTQ may produce circuits that have _worse_ accuracy than your original model.
```

## Ops supported for evaluation/numpy conversion

The following operators should be supported for evaluation and conversion to an equivalent numpy circuit. As long as your model converts to an ONNX using these operators, it _should_ be convertible to an FHE equivalent.

```{note}
Do note that all operators may not be fully supported for conversion to a circuit executable in FHE. You will get error messages should you use such an operator in a circuit you are trying to convert to FHE.
```

<!--- gen_supported_ops.py: inject supported operations for evaluation [BEGIN] -->

<!--- do not edit, auto generated part by `python3 gen_supported_ops.py` in docker -->

- Abs
- Acos
- Acosh
- Add
- Asin
- Asinh
- Atan
- Atanh
- Celu
- Clip
- Constant
- Conv
- Cos
- Cosh
- Div
- Elu
- Equal
- Erf
- Exp
- Gemm
- Greater
- HardSigmoid
- Identity
- LeakyRelu
- Less
- Log
- MatMul
- Mul
- Not
- Relu
- Reshape
- Selu
- Sigmoid
- Sin
- Sinh
- Softplus
- Sub
- Tan
- Tanh
- ThresholdedRelu

<!--- gen_supported_ops.py: inject supported operations for evaluation [END] -->

## Ops supported for Post Training Quantization

<!--- gen_supported_ops.py: inject supported operations for PTQ [BEGIN] -->

<!--- do not edit, auto generated part by `python3 gen_supported_ops.py` in docker -->

- Abs: QuantizedAbs
- Add: QuantizedAdd
- Celu: QuantizedCelu
- Clip: QuantizedClip
- Conv: QuantizedConv
- Elu: QuantizedElu
- Exp: QuantizedExp
- Gemm: QuantizedGemm
- HardSigmoid: QuantizedHardSigmoid
- Identity: QuantizedIdentity
- LeakyRelu: QuantizedLeakyRelu
- Linear: QuantizedLinear
- Log: QuantizedLog
- MatMul: QuantizedMatMul
- Relu: QuantizedRelu
- Reshape: QuantizedReshape
- Selu: QuantizedSelu
- Sigmoid: QuantizedSigmoid
- Softplus: QuantizedSoftplus
- Tanh: QuantizedTanh

<!--- gen_supported_ops.py: inject supported operations for PTQ [END] -->
