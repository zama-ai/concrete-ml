# Table of contents

- [Welcome to Cocnrete ML](README.md)

## Getting Started

- [What is Concrete ML?](getting-started/readme.md)
- [Installation](getting-started/pip_installing.md)
- [Key concepts](getting-started/concepts.md)
- [Inference in the cloud](getting-started/cloud.md)

## Built-in Models

- [Linear models](built-in-models/linear.md)
- [Tree-based models](built-in-models/tree.md)
- [Neural networks](built-in-models/neural-networks.md)
- [Nearest neighbors](built-in-models/nearest-neighbors.md)
- [Pandas](built-in-models/pandas.md)
- [Encrypted training](built-in-models/training.md)

## Deep Learning

- [Using Torch](deep-learning/torch_support.md)
- [Using ONNX](deep-learning/onnx_support.md)
- [Step-by-step guide](deep-learning/fhe_friendly_models.md)
- [Debugging models](deep-learning/fhe_assistant.md)
- [Optimizing inference](deep-learning/optimizing_inference.md)

## Guides

- [Prediction with FHE](guides/prediction_with_fhe.md)
- [Production deployment](guides/client_server.md)
- [Hybrid models](guides/hybrid-models.md)
- [Serialization](guides/serialization.md)

## Tutorials

- [See all tutorials](tutorials/showcase.md)
- [Built-in model examples](tutorials/ml_examples.md)
- [Deep learning examples](tutorials/examples.md)

## References

- [API](references/api/README.md)
  - [concrete.ml.common.check_inputs.md](references/api/concrete.ml.common.check_inputs.md)
  - [concrete.ml.common.debugging.custom_assert.md](references/api/concrete.ml.common.debugging.custom_assert.md)
  - [concrete.ml.common.debugging.md](references/api/concrete.ml.common.debugging.md)
  - [concrete.ml.common.md](references/api/concrete.ml.common.md)
  - [concrete.ml.common.serialization.decoder.md](references/api/concrete.ml.common.serialization.decoder.md)
  - [concrete.ml.common.serialization.dumpers.md](references/api/concrete.ml.common.serialization.dumpers.md)
  - [concrete.ml.common.serialization.encoder.md](references/api/concrete.ml.common.serialization.encoder.md)
  - [concrete.ml.common.serialization.loaders.md](references/api/concrete.ml.common.serialization.loaders.md)
  - [concrete.ml.common.serialization.md](references/api/concrete.ml.common.serialization.md)
  - [concrete.ml.common.utils.md](references/api/concrete.ml.common.utils.md)
  - [concrete.ml.deployment.deploy_to_aws.md](references/api/concrete.ml.deployment.deploy_to_aws.md)
  - [concrete.ml.deployment.deploy_to_docker.md](references/api/concrete.ml.deployment.deploy_to_docker.md)
  - [concrete.ml.deployment.fhe_client_server.md](references/api/concrete.ml.deployment.fhe_client_server.md)
  - [concrete.ml.deployment.md](references/api/concrete.ml.deployment.md)
  - [concrete.ml.deployment.server.md](references/api/concrete.ml.deployment.server.md)
  - [concrete.ml.deployment.utils.md](references/api/concrete.ml.deployment.utils.md)
  - [concrete.ml.onnx.convert.md](references/api/concrete.ml.onnx.convert.md)
  - [concrete.ml.onnx.md](references/api/concrete.ml.onnx.md)
  - [concrete.ml.onnx.onnx_impl_utils.md](references/api/concrete.ml.onnx.onnx_impl_utils.md)
  - [concrete.ml.onnx.onnx_model_manipulations.md](references/api/concrete.ml.onnx.onnx_model_manipulations.md)
  - [concrete.ml.onnx.onnx_utils.md](references/api/concrete.ml.onnx.onnx_utils.md)
  - [concrete.ml.onnx.ops_impl.md](references/api/concrete.ml.onnx.ops_impl.md)
  - [concrete.ml.pytest.md](references/api/concrete.ml.pytest.md)
  - [concrete.ml.pytest.torch_models.md](references/api/concrete.ml.pytest.torch_models.md)
  - [concrete.ml.pytest.utils.md](references/api/concrete.ml.pytest.utils.md)
  - [concrete.ml.quantization.base_quantized_op.md](references/api/concrete.ml.quantization.base_quantized_op.md)
  - [concrete.ml.quantization.md](references/api/concrete.ml.quantization.md)
  - [concrete.ml.quantization.post_training.md](references/api/concrete.ml.quantization.post_training.md)
  - [concrete.ml.quantization.quantized_module.md](references/api/concrete.ml.quantization.quantized_module.md)
  - [concrete.ml.quantization.quantized_module_passes.md](references/api/concrete.ml.quantization.quantized_module_passes.md)
  - [concrete.ml.quantization.quantized_ops.md](references/api/concrete.ml.quantization.quantized_ops.md)
  - [concrete.ml.quantization.quantizers.md](references/api/concrete.ml.quantization.quantizers.md)
  - [concrete.ml.search_parameters.md](references/api/concrete.ml.search_parameters.md)
  - [concrete.ml.search_parameters.p_error_search.md](references/api/concrete.ml.search_parameters.p_error_search.md)
  - [concrete.ml.sklearn.base.md](references/api/concrete.ml.sklearn.base.md)
  - [concrete.ml.sklearn.glm.md](references/api/concrete.ml.sklearn.glm.md)
  - [concrete.ml.sklearn.linear_model.md](references/api/concrete.ml.sklearn.linear_model.md)
  - [concrete.ml.sklearn.md](references/api/concrete.ml.sklearn.md)
  - [concrete.ml.sklearn.neighbors.md](references/api/concrete.ml.sklearn.neighbors.md)
  - [concrete.ml.sklearn.qnn.md](references/api/concrete.ml.sklearn.qnn.md)
  - [concrete.ml.sklearn.qnn_module.md](references/api/concrete.ml.sklearn.qnn_module.md)
  - [concrete.ml.sklearn.rf.md](references/api/concrete.ml.sklearn.rf.md)
  - [concrete.ml.sklearn.svm.md](references/api/concrete.ml.sklearn.svm.md)
  - [concrete.ml.sklearn.tree.md](references/api/concrete.ml.sklearn.tree.md)
  - [concrete.ml.sklearn.tree_to_numpy.md](references/api/concrete.ml.sklearn.tree_to_numpy.md)
  - [concrete.ml.sklearn.xgb.md](references/api/concrete.ml.sklearn.xgb.md)
  - [concrete.ml.torch.compile.md](references/api/concrete.ml.torch.compile.md)
  - [concrete.ml.torch.hybrid_model.md](references/api/concrete.ml.torch.hybrid_model.md)
  - [concrete.ml.torch.md](references/api/concrete.ml.torch.md)
  - [concrete.ml.torch.numpy_module.md](references/api/concrete.ml.torch.numpy_module.md)
  - [concrete.ml.version.md](references/api/concrete.ml.version.md)

## Explanations

- [Quantization](explanations/quantization.md)
- [Pruning](explanations/pruning.md)
- [Compilation](explanations/compilation.md)
- [Advanced features](explanations/advanced_features.md)
- [Project architecture](explanations/inner-workings/README.md)
  - [Importing ONNX](explanations/inner-workings/onnx_pipeline.md)
  - [Quantization tools](explanations/inner-workings/quantization_internal.md)
  - [FHE Op-graph design](explanations/inner-workings/fhe-op-graphs.md)
  - [External libraries](explanations/inner-workings/external_libraries.md)

## Developer

- [Set up the project](developer/project_setup.md)
- [Set up Docker](developer/docker_setup.md)
- [Documentation](developer/documenting.md)
- [Support and issues](developer/debug_support_submit_issues.md)
- [Contributing](developer/contributing.md)
- [Release note](https://github.com/zama-ai/concrete-ml/releases)
- [Feature request](https://github.com/zama-ai/concrete-ml/issues/new?assignees=&labels=feature&projects=&template=feature_request.md)
- [Bug report](https://github.com/zama-ai/concrete-ml/issues/new?assignees=&labels=bug&projects=&template=bug_report.md)
