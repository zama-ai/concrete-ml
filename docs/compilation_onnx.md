# Compilation of ONNX Models

```{note}
FIXME: to be done, Benoit, #959, #968
```

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
- Pow
- Relu
- Reshape
- Round
- Selu
- Sigmoid
- Sin
- Sinh
- Softplus
- Sub
- Tan
- Tanh
- ThresholdedRelu
- Transpose
- Where

<!--- gen_supported_ops.py: inject supported operations for evaluation [END] -->
