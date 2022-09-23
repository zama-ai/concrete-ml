# Table of contents

- [What is Concrete ML?](README.md)

## Getting Started

- [Installation](getting-started/pip_installing.md)
- [Key Concepts](getting-started/concepts.md)
- [Inference in the Cloud](getting-started/cloud.md)

## Built-in Models

- [Linear Models](built-in-models/linear.md)
- [Tree-based Models](built-in-models/tree.md)
- [Neural Networks](built-in-models/neural-networks.md)
- [Examples](built-in-models/ml_examples.md)

## Deep Learning

- [Using Torch](deep-learning/torch_support.md)
- [Using ONNX](deep-learning/onnx_support.md)
- [Step-by-step guide](deep-learning/fhe_friendly_models.md)
- [Examples](deep-learning/examples.md)
- [Debugging Models](deep-learning/fhe_assistant.md)

## Advanced topics

- [Quantization](advanced-topics/quantization.md)
- [Pruning](advanced-topics/pruning.md)
- [Compilation](advanced-topics/compilation.md)
- [Production Deployment](advanced-topics/client_server.md)
- [Advanced Features](advanced-topics/advanced_features.md)

## Developer Guide

- [Workflow](developer-guide/workflow/README.md)
  - [Set Up the Project](developer-guide/project_setup.md)
  - [Set Up Docker](developer-guide/docker_setup.md)
  - [Documentation](developer-guide/documenting.md)
  - [Support and Issues](developer-guide/debug_support_submit_issues.md)
  - [Contributing](developer-guide/contributing.md)
- [Inner workings](developer-guide/inner-workings/README.md)
  - [Importing ONNX](developer-guide/onnx_pipeline.md)
  - [Quantization tools](developer-guide/quantization_internal.md)
  - [FHE Op-graph design](developer-guide/fhe-op-graphs.md)
  - [External Libraries](developer-guide/external_libraries.md)
- [API](developer-guide/api/README.md)
  - [concrete.ml.common](developer-guide/api/concrete.ml.common.md)
  - [concrete.ml.common.check_inputs](developer-guide/api/concrete.ml.common.check_inputs.md)
  - [concrete.ml.common.debugging](developer-guide/api/concrete.ml.common.debugging.md)
  - [concrete.ml.common.debugging.custom_assert](developer-guide/api/concrete.ml.common.debugging.custom_assert.md)
  - [concrete.ml.common.utils](developer-guide/api/concrete.ml.common.utils.md)
  - [concrete.ml.deployment](developer-guide/api/concrete.ml.deployment.md)
  - [concrete.ml.deployment.fhe_client_server](developer-guide/api/concrete.ml.deployment.fhe_client_server.md)
  - [concrete.ml.onnx](developer-guide/api/concrete.ml.onnx.md)
  - [concrete.ml.onnx.convert](developer-guide/api/concrete.ml.onnx.convert.md)
  - [concrete.ml.onnx.onnx_model_manipulations](developer-guide/api/concrete.ml.onnx.onnx_model_manipulations.md)
  - [concrete.ml.onnx.onnx_utils](developer-guide/api/concrete.ml.onnx.onnx_utils.md)
  - [concrete.ml.onnx.ops_impl](developer-guide/api/concrete.ml.onnx.ops_impl.md)
  - [concrete.ml.quantization](developer-guide/api/concrete.ml.quantization.md)
  - [concrete.ml.quantization.base_quantized_op](developer-guide/api/concrete.ml.quantization.base_quantized_op.md)
  - [concrete.ml.quantization.post_training](developer-guide/api/concrete.ml.quantization.post_training.md)
  - [concrete.ml.quantization.quantized_module](developer-guide/api/concrete.ml.quantization.quantized_module.md)
  - [concrete.ml.quantization.quantized_ops](developer-guide/api/concrete.ml.quantization.quantized_ops.md)
  - [concrete.ml.quantization.quantizers](developer-guide/api/concrete.ml.quantization.quantizers.md)
  - [concrete.ml.sklearn](developer-guide/api/concrete.ml.sklearn.md)
  - [concrete.ml.sklearn.base](developer-guide/api/concrete.ml.sklearn.base.md)
  - [concrete.ml.sklearn.glm](developer-guide/api/concrete.ml.sklearn.glm.md)
  - [concrete.ml.sklearn.linear_model](developer-guide/api/concrete.ml.sklearn.linear_model.md)
  - [concrete.ml.sklearn.protocols](developer-guide/api/concrete.ml.sklearn.protocols.md)
  - [concrete.ml.sklearn.qnn](developer-guide/api/concrete.ml.sklearn.qnn.md)
  - [concrete.ml.sklearn.rf](developer-guide/api/concrete.ml.sklearn.rf.md)
  - [concrete.ml.sklearn.svm](developer-guide/api/concrete.ml.sklearn.svm.md)
  - [concrete.ml.sklearn.torch_module](developer-guide/api/concrete.ml.sklearn.torch_module.md)
  - [concrete.ml.sklearn.tree](developer-guide/api/concrete.ml.sklearn.tree.md)
  - [concrete.ml.sklearn.tree_to_numpy](developer-guide/api/concrete.ml.sklearn.tree_to_numpy.md)
  - [concrete.ml.sklearn.xgb](developer-guide/api/concrete.ml.sklearn.xgb.md)
  - [concrete.ml.torch](developer-guide/api/concrete.ml.torch.md)
  - [concrete.ml.torch.compile](developer-guide/api/concrete.ml.torch.compile.md)
  - [concrete.ml.torch.numpy_module](developer-guide/api/concrete.ml.torch.numpy_module.md)
  - [concrete.ml.version](developer-guide/api/concrete.ml.version.md)
