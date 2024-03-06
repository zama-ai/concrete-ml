# Table of contents

* [Welcome to Cocnrete ML](README.md)

## Getting Started

* [What is Concrete ML?](getting-started/readme.md)
* [Installation](getting-started/pip\_installing.md)
* [Key concepts](getting-started/concepts.md)
* [Inference in the cloud](getting-started/cloud.md)

## Built-in Models

* [Linear models](built-in-models/linear.md)
* [Tree-based models](built-in-models/tree.md)
* [Neural networks](built-in-models/neural-networks.md)
* [Nearest neighbors](built-in-models/nearest-neighbors.md)
* [Pandas](built-in-models/pandas.md)
* [Encrypted training](built-in-models/training.md)

## Deep Learning

* [Using Torch](deep-learning/torch\_support.md)
* [Using ONNX](deep-learning/onnx\_support.md)
* [Step-by-step guide](deep-learning/fhe\_friendly\_models.md)
* [Debugging models](deep-learning/fhe\_assistant.md)
* [Optimizing inference](deep-learning/optimizing\_inference.md)

## Guides

* [Prediction with FHE](guides/prediction\_with\_fhe.md)
* [Production deployment](guides/client\_server.md)
* [Hybrid models](guides/hybrid-models.md)
* [Serialization](guides/serialization.md)

## Tutorials

* [See all tutorials](tutorials/showcase.md)
* [Built-in model examples](tutorials/ml\_examples.md)
* [Deep learning examples](tutorials/examples.md)

## References

* [API](references/api/README.md)
  * [concrete.ml.common.check\_inputs.md](references/api/concrete.ml.common.check\_inputs.md)
  * [concrete.ml.common.debugging.custom\_assert.md](references/api/concrete.ml.common.debugging.custom\_assert.md)
  * [concrete.ml.common.debugging.md](references/api/concrete.ml.common.debugging.md)
  * [concrete.ml.common.md](references/api/concrete.ml.common.md)
  * [concrete.ml.common.serialization.decoder.md](references/api/concrete.ml.common.serialization.decoder.md)
  * [concrete.ml.common.serialization.dumpers.md](references/api/concrete.ml.common.serialization.dumpers.md)
  * [concrete.ml.common.serialization.encoder.md](references/api/concrete.ml.common.serialization.encoder.md)
  * [concrete.ml.common.serialization.loaders.md](references/api/concrete.ml.common.serialization.loaders.md)
  * [concrete.ml.common.serialization.md](references/api/concrete.ml.common.serialization.md)
  * [concrete.ml.common.utils.md](references/api/concrete.ml.common.utils.md)
  * [concrete.ml.deployment.deploy\_to\_aws.md](references/api/concrete.ml.deployment.deploy\_to\_aws.md)
  * [concrete.ml.deployment.deploy\_to\_docker.md](references/api/concrete.ml.deployment.deploy\_to\_docker.md)
  * [concrete.ml.deployment.fhe\_client\_server.md](references/api/concrete.ml.deployment.fhe\_client\_server.md)
  * [concrete.ml.deployment.md](references/api/concrete.ml.deployment.md)
  * [concrete.ml.deployment.server.md](references/api/concrete.ml.deployment.server.md)
  * [concrete.ml.deployment.utils.md](references/api/concrete.ml.deployment.utils.md)
  * [concrete.ml.onnx.convert.md](references/api/concrete.ml.onnx.convert.md)
  * [concrete.ml.onnx.md](references/api/concrete.ml.onnx.md)
  * [concrete.ml.onnx.onnx\_impl\_utils.md](references/api/concrete.ml.onnx.onnx\_impl\_utils.md)
  * [concrete.ml.onnx.onnx\_model\_manipulations.md](references/api/concrete.ml.onnx.onnx\_model\_manipulations.md)
  * [concrete.ml.onnx.onnx\_utils.md](references/api/concrete.ml.onnx.onnx\_utils.md)
  * [concrete.ml.onnx.ops\_impl.md](references/api/concrete.ml.onnx.ops\_impl.md)
  * [concrete.ml.pytest.md](references/api/concrete.ml.pytest.md)
  * [concrete.ml.pytest.torch\_models.md](references/api/concrete.ml.pytest.torch\_models.md)
  * [concrete.ml.pytest.utils.md](references/api/concrete.ml.pytest.utils.md)
  * [concrete.ml.quantization.base\_quantized\_op.md](references/api/concrete.ml.quantization.base\_quantized\_op.md)
  * [concrete.ml.quantization.md](references/api/concrete.ml.quantization.md)
  * [concrete.ml.quantization.post\_training.md](references/api/concrete.ml.quantization.post\_training.md)
  * [concrete.ml.quantization.quantized\_module.md](references/api/concrete.ml.quantization.quantized\_module.md)
  * [concrete.ml.quantization.quantized\_module\_passes.md](references/api/concrete.ml.quantization.quantized\_module\_passes.md)
  * [concrete.ml.quantization.quantized\_ops.md](references/api/concrete.ml.quantization.quantized\_ops.md)
  * [concrete.ml.quantization.quantizers.md](references/api/concrete.ml.quantization.quantizers.md)
  * [concrete.ml.search\_parameters.md](references/api/concrete.ml.search\_parameters.md)
  * [concrete.ml.search\_parameters.p\_error\_search.md](references/api/concrete.ml.search\_parameters.p\_error\_search.md)
  * [concrete.ml.sklearn.base.md](references/api/concrete.ml.sklearn.base.md)
  * [concrete.ml.sklearn.glm.md](references/api/concrete.ml.sklearn.glm.md)
  * [concrete.ml.sklearn.linear\_model.md](references/api/concrete.ml.sklearn.linear\_model.md)
  * [concrete.ml.sklearn.md](references/api/concrete.ml.sklearn.md)
  * [concrete.ml.sklearn.neighbors.md](references/api/concrete.ml.sklearn.neighbors.md)
  * [concrete.ml.sklearn.qnn.md](references/api/concrete.ml.sklearn.qnn.md)
  * [concrete.ml.sklearn.qnn\_module.md](references/api/concrete.ml.sklearn.qnn\_module.md)
  * [concrete.ml.sklearn.rf.md](references/api/concrete.ml.sklearn.rf.md)
  * [concrete.ml.sklearn.svm.md](references/api/concrete.ml.sklearn.svm.md)
  * [concrete.ml.sklearn.tree.md](references/api/concrete.ml.sklearn.tree.md)
  * [concrete.ml.sklearn.tree\_to\_numpy.md](references/api/concrete.ml.sklearn.tree\_to\_numpy.md)
  * [concrete.ml.sklearn.xgb.md](references/api/concrete.ml.sklearn.xgb.md)
  * [concrete.ml.torch.compile.md](references/api/concrete.ml.torch.compile.md)
  * [concrete.ml.torch.hybrid\_model.md](references/api/concrete.ml.torch.hybrid\_model.md)
  * [concrete.ml.torch.md](references/api/concrete.ml.torch.md)
  * [concrete.ml.torch.numpy\_module.md](references/api/concrete.ml.torch.numpy\_module.md)
  * [concrete.ml.version.md](references/api/concrete.ml.version.md)

## Explanations

* [Quantization](explanations/quantization.md)
* [Pruning](explanations/pruning.md)
* [Compilation](explanations/compilation.md)
* [Advanced features](explanations/advanced\_features.md)
* [Project architecture](explanations/inner-workings/README.md)
  * [Importing ONNX](explanations/inner-workings/onnx\_pipeline.md)
  * [Quantization tools](explanations/inner-workings/quantization\_internal.md)
  * [FHE Op-graph design](explanations/inner-workings/fhe-op-graphs.md)
  * [External libraries](explanations/inner-workings/external\_libraries.md)

## Developer

* [Set up the project](developer/project\_setup.md)
* [Set up Docker](developer/docker\_setup.md)
* [Documentation](developer/documenting.md)
* [Support and issues](developer/debug\_support\_submit\_issues.md)
* [Contributing](developer/contributing.md)
* [Release note](https://github.com/zama-ai/concrete-ml/releases)
* [Feature request](https://github.com/zama-ai/concrete-ml/issues/new?assignees=\&labels=feature\&projects=\&template=feature\_request.md)
* [Bug report](https://github.com/zama-ai/concrete-ml/issues/new?assignees=\&labels=bug\&projects=\&template=bug\_report.md)
