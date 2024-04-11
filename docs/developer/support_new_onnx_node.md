# Supporting New ONNX Nodes in Concrete ML

Concrete ML supports a wide range of models through the integration of ONNX nodes. In case a specific ONNX node is missing, developers need to add support for the new ONNX nodes.

## Operator Implementation

### [`ops_impl.py`](../../src/concrete/ml/onnx/ops_impl.py)

This file is responsible for implementing the computation of ONNX operators using floating-point arithmetic. The implementation should mirror the behavior of the corresponding ONNX operator precisely. This includes adhering to the expected inputs, outputs, and operational semantics.

Refer to the [ONNX documentation](https://github.com/onnx/onnx/blob/main/docs/Operators.md) to grasp the expected behavior, inputs and outputs of the operator.

### [`onnx_utils.py`](../../src/concrete/ml/onnx/onnx_utils.py)

After implementing the operator in `ops_impl.py`, you need to import it into `onnx_utils.py` and map it within the `ONNX_OPS_TO_NUMPY_IMPL` dictionary. This mapping is crucial for the framework to recognize and utilize the new operator.

## [`quantized_ops.py`](../../src/concrete/ml/quantization/quantized_ops.py)

Quantized operators handle integer arithmetic and its implementation is required for the new ONNX to be executed in FHE.

There exist two types of quantized operators:

- **Univariate Non-Linear Operators**: Such operator applies transformation on every element of the input without changing its shape. Sigmoid, Tanh, ReLU are examples of such operation. The sigmoid in this file is simply supported as follows:

<!--pytest-codeblocks:skip-->

```python
class QuantizedSigmoid(QuantizedOp):
    """Quantized sigmoid op."""

    _impl_for_op_named: str = "Sigmoid"
```

- **Linear Layers**: Linear layers like `Gemm` and `Conv` requires specific implementation for integer arithmetic. Please refer to the `QuantizedGemm` and `QuantizedConv` implementation for reference.

## Adding Tests

Proper testing is essential to ensure the correctness of the new ONNX node support.

There are many locations where tests can be added:

- [`test_onnx_ops_impl.py`](../../tests/onnx/test_onnx_ops_impl.py): Tests implementation of the ONNX node in floating points.
- [`test_quantized_ops.py`](../../tests/quantization/test_quantized_ops.py): Tests implementation of the ONNX node in integer arithmetic.
- Optional: [`test_compile_torch.py`](../../tests/torch/test_compile_torch.py): Tests a specific torch model implementation that contains the new ONNX operator. The model needs to be added to [`torch_models.py`](../../src/concrete/ml/pytest/torch_models.py).

## Update Documentation

Finally, update the documentation to reflect the newly supported ONNX node.

**Documentation File**: Edit the [docs/deep-learning/onnx_support.md](../deep-learning/onnx_support.md) file to include the new operator.
