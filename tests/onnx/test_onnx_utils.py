"""Test ONNX utils."""

import numpy as np
import onnx
import pytest

from concrete.ml.onnx.convert import OPSET_VERSION_FOR_ONNX_EXPORT
from concrete.ml.onnx.onnx_utils import check_onnx_model


# FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4604
@pytest.mark.skip()
def test_check_onnx_model_large():
    """Test that check_onnx_model can handle models larger than 2GB."""

    model = onnx.ModelProto()
    graph = onnx.GraphProto()
    graph.name = "LargeModel"

    # Create a large tensor (slightly over the 2GB limit)
    large_tensor = np.random.rand(1000, 1000, 550).astype(np.float32)
    tensor_proto = onnx.numpy_helper.from_array(large_tensor, name="large_tensor")

    graph.initializer.append(tensor_proto)
    model.graph.CopyFrom(graph)

    # Set ir_version
    model.ir_version = onnx.IR_VERSION

    # Add opset_import
    opset = model.opset_import.add()
    opset.version = OPSET_VERSION_FOR_ONNX_EXPORT

    # Test that onnx.checker.check_model raises an exception
    with pytest.raises(
        ValueError, match="Message onnx.ModelProto exceeds maximum protobuf size of 2GB:"
    ):
        onnx.checker.check_model(model)

    # Our custom check_onnx_model should work fine
    check_onnx_model(model)

    # Call check_onnx_model a second time to ensure the original model wasn't modified
    check_onnx_model(model)
