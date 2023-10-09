# Table of Contents

- [What is Concrete ML?](README.md)

## Getting Started

- [Installation](getting-started/pip_installing.md)
- [Key Concepts](getting-started/concepts.md)
- [Inference in the Cloud](getting-started/cloud.md)
- [Demos and Tutorials](getting-started/showcase.md)

## Built-in Models

- [Linear Models](built-in-models/linear.md)
- [Tree-based Models](built-in-models/tree.md)
- [Neural Networks](built-in-models/neural-networks.md)
- [Nearest Neighbors](built-in-models/nearest-neighbors.md)
- [Pandas](built-in-models/pandas.md)
- [Built-in Model Examples](built-in-models/ml_examples.md)

## Deep Learning

- [Using Torch](deep-learning/torch_support.md)
- [Using ONNX](deep-learning/onnx_support.md)
- [Step-by-step Guide](deep-learning/fhe_friendly_models.md)
- [Deep Learning Examples](deep-learning/examples.md)
- [Debugging Models](deep-learning/fhe_assistant.md)
- [Optimizing Inference](deep-learning/optimizing_inference.md)

## Deployment

- [Prediction with FHE](advanced-topics/prediction_with_fhe.md)
- [Hybrid models](advanced-topics/hybrid-models.md)
- [Production Deployment](advanced-topics/client_server.md)
- [Serialization](advanced-topics/serialization.md)

## Advanced topics

- [Quantization](advanced-topics/quantization.md)
- [Pruning](advanced-topics/pruning.md)
- [Compilation](advanced-topics/compilation.md)
- [Advanced Features](advanced-topics/advanced_features.md)

## Developer Guide

- [Workflow](developer-guide/workflow/README.md)
  - [Set Up the Project](developer-guide/project_setup.md)
  - [Set Up Docker](developer-guide/docker_setup.md)
  - [Documentation](developer-guide/documenting.md)
  - [Support and Issues](developer-guide/debug_support_submit_issues.md)
  - [Contributing](developer-guide/contributing.md)
- [Inner Workings](developer-guide/inner-workings/README.md)
  - [Importing ONNX](developer-guide/onnx_pipeline.md)
  - [Quantization Tools](developer-guide/quantization_internal.md)
  - [FHE Op-graph Design](developer-guide/fhe-op-graphs.md)
  - [External Libraries](developer-guide/external_libraries.md)
- [API](developer-guide/api/README.md)
  <!-- auto-created, do not edit, begin -->
  - [concrete.ml.common.check_inputs.md](developer-guide/api/concrete.ml.common.check_inputs.md)
  - [concrete.ml.common.debugging.custom_assert.md](developer-guide/api/concrete.ml.common.debugging.custom_assert.md)
  - [concrete.ml.common.debugging.md](developer-guide/api/concrete.ml.common.debugging.md)
  - [concrete.ml.common.md](developer-guide/api/concrete.ml.common.md)
  - [concrete.ml.common.serialization.decoder.md](developer-guide/api/concrete.ml.common.serialization.decoder.md)
  - [concrete.ml.common.serialization.dumpers.md](developer-guide/api/concrete.ml.common.serialization.dumpers.md)
  - [concrete.ml.common.serialization.encoder.md](developer-guide/api/concrete.ml.common.serialization.encoder.md)
  - [concrete.ml.common.serialization.loaders.md](developer-guide/api/concrete.ml.common.serialization.loaders.md)
  - [concrete.ml.common.serialization.md](developer-guide/api/concrete.ml.common.serialization.md)
  - [concrete.ml.common.utils.md](developer-guide/api/concrete.ml.common.utils.md)
  - [concrete.ml.deployment.deploy_to_aws.md](developer-guide/api/concrete.ml.deployment.deploy_to_aws.md)
  - [concrete.ml.deployment.deploy_to_docker.md](developer-guide/api/concrete.ml.deployment.deploy_to_docker.md)
  - [concrete.ml.deployment.fhe_client_server.md](developer-guide/api/concrete.ml.deployment.fhe_client_server.md)
  - [concrete.ml.deployment.md](developer-guide/api/concrete.ml.deployment.md)
  - [concrete.ml.deployment.server.md](developer-guide/api/concrete.ml.deployment.server.md)
  - [concrete.ml.deployment.utils.md](developer-guide/api/concrete.ml.deployment.utils.md)
  - [concrete.ml.onnx.convert.md](developer-guide/api/concrete.ml.onnx.convert.md)
  - [concrete.ml.onnx.md](developer-guide/api/concrete.ml.onnx.md)
  - [concrete.ml.onnx.onnx_impl_utils.md](developer-guide/api/concrete.ml.onnx.onnx_impl_utils.md)
  - [concrete.ml.onnx.onnx_model_manipulations.md](developer-guide/api/concrete.ml.onnx.onnx_model_manipulations.md)
  - [concrete.ml.onnx.onnx_utils.md](developer-guide/api/concrete.ml.onnx.onnx_utils.md)
  - [concrete.ml.onnx.ops_impl.md](developer-guide/api/concrete.ml.onnx.ops_impl.md)
  - [concrete.ml.pytest.md](developer-guide/api/concrete.ml.pytest.md)
  - [concrete.ml.pytest.torch_models.md](developer-guide/api/concrete.ml.pytest.torch_models.md)
  - [concrete.ml.pytest.utils.md](developer-guide/api/concrete.ml.pytest.utils.md)
  - [concrete.ml.quantization.base_quantized_op.md](developer-guide/api/concrete.ml.quantization.base_quantized_op.md)
  - [concrete.ml.quantization.md](developer-guide/api/concrete.ml.quantization.md)
  - [concrete.ml.quantization.post_training.md](developer-guide/api/concrete.ml.quantization.post_training.md)
  - [concrete.ml.quantization.quantized_module.md](developer-guide/api/concrete.ml.quantization.quantized_module.md)
  - [concrete.ml.quantization.quantized_module_passes.md](developer-guide/api/concrete.ml.quantization.quantized_module_passes.md)
  - [concrete.ml.quantization.quantized_ops.md](developer-guide/api/concrete.ml.quantization.quantized_ops.md)
  - [concrete.ml.quantization.quantizers.md](developer-guide/api/concrete.ml.quantization.quantizers.md)
  - [concrete.ml.search_parameters.md](developer-guide/api/concrete.ml.search_parameters.md)
  - [concrete.ml.search_parameters.p_error_search.md](developer-guide/api/concrete.ml.search_parameters.p_error_search.md)
  - [concrete.ml.sklearn.base.md](developer-guide/api/concrete.ml.sklearn.base.md)
  - [concrete.ml.sklearn.glm.md](developer-guide/api/concrete.ml.sklearn.glm.md)
  - [concrete.ml.sklearn.linear_model.md](developer-guide/api/concrete.ml.sklearn.linear_model.md)
  - [concrete.ml.sklearn.md](developer-guide/api/concrete.ml.sklearn.md)
  - [concrete.ml.sklearn.neighbors.md](developer-guide/api/concrete.ml.sklearn.neighbors.md)
  - [concrete.ml.sklearn.qnn.md](developer-guide/api/concrete.ml.sklearn.qnn.md)
  - [concrete.ml.sklearn.qnn_module.md](developer-guide/api/concrete.ml.sklearn.qnn_module.md)
  - [concrete.ml.sklearn.rf.md](developer-guide/api/concrete.ml.sklearn.rf.md)
  - [concrete.ml.sklearn.svm.md](developer-guide/api/concrete.ml.sklearn.svm.md)
  - [concrete.ml.sklearn.tree.md](developer-guide/api/concrete.ml.sklearn.tree.md)
  - [concrete.ml.sklearn.tree_to_numpy.md](developer-guide/api/concrete.ml.sklearn.tree_to_numpy.md)
  - [concrete.ml.sklearn.xgb.md](developer-guide/api/concrete.ml.sklearn.xgb.md)
  - [concrete.ml.torch.compile.md](developer-guide/api/concrete.ml.torch.compile.md)
  - [concrete.ml.torch.hybrid_model.md](developer-guide/api/concrete.ml.torch.hybrid_model.md)
  - [concrete.ml.torch.md](developer-guide/api/concrete.ml.torch.md)
  - [concrete.ml.torch.numpy_module.md](developer-guide/api/concrete.ml.torch.numpy_module.md)
  - [concrete.ml.version.md](developer-guide/api/concrete.ml.version.md)
  <!-- auto-created, do not edit, end -->
