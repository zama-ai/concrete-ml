# ONNX Operator Support for FHE Model Conversion

Internally, **Concrete-ML** uses [ONNX](https://github.com/onnx/onnx) operators as intermediate representation (or IR) for manipulating machine learning models produced through export for [PyTorch](https://github.com/pytorch/pytorch), [Hummingbird](https://github.com/microsoft/hummingbird) and [skorch](https://github.com/skorch-dev/skorch). As ONNX is becoming the standard exchange format for neural networks, this allows **Concrete-ML** to be flexible while also making model representation manipulation quite easy. In addition, it allows for straight-forward mapping to NumPy operators, supported by **Concrete-Numpy** to use the **Concrete** stack FHE conversion capabilities.

In this page we list the operators that are supported.

## Ops supported for evaluation/NumPy conversion

The following operators have some support for evaluation and conversion to an equivalent NumPy circuit.
Do note that all operators may not be fully supported for conversion to a circuit executable in FHE. We sometimes implement only partially the operators, either because of some limits due to FHE or because we did not need more than special case for supporting e.g. PyTorch activations or scikit-learn models.

<!--- gen_supported_ops.py: inject supported operations for evaluation [BEGIN] -->

<!--- do not edit, auto generated part by `make supported_ops` -->

- Abs
- Acos
- Acosh
- Add
- Asin
- Asinh
- Atan
- Atanh
- AveragePool
- BatchNormalization
- Cast
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
- Flatten
- Gemm
- Greater
- HardSigmoid
- HardSwish
- Identity
- LeakyRelu
- Less
- Log
- MatMul
- Mul
- Not
- Or
- PRelu
- Pad
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
- Where

<!--- gen_supported_ops.py: inject supported operations for evaluation [END] -->
