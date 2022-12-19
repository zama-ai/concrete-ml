# CIFAR-10 classification with a split clear/FHE model

In this [notebook](./Cifar10.ipynb) we show how train and compile a FHE VGG-like model to achieve a good speed/accuracy tradeoff on CIFAR-10 images.

## Model design

As there is a trade-off between accumulator bit-width and FHE inference speed, this tutorial targets
8-bit accumulators, to achieve faster FHE inference times. Moreover, we split the model in two parts to allow
higher precision in the input layer. For the FHE part of the model, we used pruning and 2-bit weights to try to satisfy this constraint.

The first layer of any deep vision model processes the raw images that are usually represented using 8-bit integers.
With respect to FHE constraints, such large bit-widths for inputs are a bottleneck with regards to the accumulator size constraint. Therefore, we opted to split the model into 2 sub-models:

- The first layer of the VGG model will run in floats and in clear,
- The rest of the network will run in integers and in FHE.

The method is generic and can be applied to any neural network model, but the compilation and deployment steps are a bit more intricate in this case. We show how to compile this split model in the [notebook](./Cifar10.ipynb).

## Running this example

To run this notebook properly you will need the usual Concrete-ML dependencies plus the extra dependencies from `requirements.txt` that you can install using `pip install -r requirements.txt` .

We also provide a script to run the model in FHE. On an AWS c6i.metal compute machine, to do the inference of one CIFAR-10 image, we got the following timings:

- Time to compile: 103 seconds
- Time to keygen: 639 seconds
- Time to infer: 37706 seconds (more than 10 hours)

## Results

Anyone can reproduce the FHE inference results using the [dedicated script](./fhe_inference.py).

The Pytorch model and the inference using Pytorch for the first layer and the virtual library for the encrypted part yielded the same top-k accuracies:

- top-1-accuracy: 0.6234
- top-2-accuracy: 0.8075
- top-3-accuracy: 0.8905

which are decent metrics for a traditional VGG model under such constraints.

The accuracy of the model running in FHE was not measured because of the computational cost it would require.
This is something we plan on measuring once FHE runtimes become more acceptable.

<!-- FIXME: Add more metrics: https://github.com/zama-ai/concrete-ml-internal/issues/2377 -->

<!-- FIXME: Add training scripts https://github.com/zama-ai/concrete-ml-internal/issues/2383 -->
