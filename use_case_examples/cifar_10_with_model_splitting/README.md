# CIFAR-10 FHE classification with 8-bit split VGG

In this [notebook](./Cifar10.ipynb) we show how to compile a splitted VGG-like model to classify CIFAR-10 images in FHE.

At the time of writing this notebook we have a 8-bit constraint on the accumulator size to be able to compile to FHE.

Thus we trained a VGG-Model with pruning and 2-bit weights to try to satisfy this constraint.

The first layer of any deep vision model usually being one of the main bottlenecks, with regard to the accumulator size constraint, we opted for splitting the model into 2 sub-modules:

- The first layer of the VGG model will run in floats and in clear,
- The rest of the network will run in integers and in FHE.

The method is generic and can be applied to any model.
Also, one thing that we should note, is that we could do any arbitrary number of computation before and after the FHE computations.
Any of these clear-data computations would have to be done on the client side.
By splitting a model this way we can preserve a decent accuracy, by reducing the impact of the quantization necessary for FHE computations, while preserving privacy with FHE.
But that means that the compilation and serving processes are a bit more intricate
We showcase how to compile this split model in the [following notebook](./Cifar10.ipynb).

To run this notebook properly you will need the usual Concrete-ML dependencies plus the extra dependencies from `requirements.txt` that you can install using `pip install -r requirements.txt` .

We also provide a script to run the model in FHE. On an AWS c6i.metal compute machine, to do the inference of one CIFAR-10 image, we got the following timings:

- Time to compile: 103 seconds
- Time to keygen: 639 seconds
- Time to infer: 37706 seconds (more than 10 hours)

Anyone can replicate the FHE inference using the [dedicated script](./fhe_inference.py).

The Pytorch model and the inference using Pytorch for the first layer and the virtual library for the encrypted part yielded the same top-k accuracies:

- top-1-accuracy: 0.6234
- top-2-accuracy: 0.8075
- top-3-accuracy: 0.8905

which are decent metrics for a traditional VGG model under such constraints.

The accuracy of the model running in FHE was not computed because of the computional cost it would require.
This is something we plan on measuring once FHE runtimes become more acceptable.

<!-- FIXME: Add more metrics: https://github.com/zama-ai/concrete-ml-internal/issues/2377 -->

<!-- FIXME: Add training scripts https://github.com/zama-ai/concrete-ml-internal/issues/2383 -->
