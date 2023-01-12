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

We also provide a script to run the model in FHE.

The naive approach to run this model in FHE would be to choose a low `p_error` or `global_p_error` and compile the model with it to run the FHE inference.
By trial and error we found that a `global_p_error` of 0.15 was one of the lowest value for which we could find crypto-parameters.
On an AWS c6i.metal compute machine, doing the inference of one CIFAR-10 image with a `global_p_error` of 0.15, we got the following timings:

- Time to compile: 112 seconds
- Time to keygen: 1231 seconds
- Time to infer: 35619 seconds (around 10 hours)

But this can be improved by searching for a better `p_error`.

One way to do this is to do a binary search using the Virtual Library to estimate the impact of the `p_error` on the final accuracy of our model.
Using the first 1000 samples of CIFAR-10 train set we ran the search to find the highest `p_error` such that the difference in accuracy between the Virtual Library and the clear model was below 1 point. This search yielded a `p_error` of approximately 0.05.
We use only a subset of the training set to make the search time acceptable, but one can either modify this number, or even do [bootstrapping](<https://en.wikipedia.org/wiki/Bootstrapping_(statistics)>), to have a better estimate.
We provide a [script](./p_error_search.py) to run the `p_error` search. Results may differ since it relies on random simulation.

Obviously the accuracy difference observed is only a simulation on these 1000 samples so a verification of this result is important to do. We validated this `p_error` choice by running 40 times the inference of the 1000 samples using the Virtual Library and the maximum difference in accuracy that we observed was of 2 points, which seemed relatively okay.

Once we had this `p_error` validated we re-run the FHE inference using this new `p_error`, on the same machine (c6i.metal) and got the following results:

- Time to compile: 109 seconds
- Time to keygen: 30 seconds
- Time to infer: 1738 seconds

We see a 20x improvement with a simple change in the `p_error` parameter, for more details on how to handle `p_error` please refer to the [documentation](../../docs/advanced-topics/advanced_features.md#approximate-computations).

## Results

Anyone can reproduce the FHE inference results using the [dedicated script](./fhe_inference.py).

The PyTorch model and the inference using PyTorch for the first layer and the Virtual Library for the encrypted part yielded the same top-k accuracies:

- top-1-accuracy: 0.6234
- top-2-accuracy: 0.8075
- top-3-accuracy: 0.8905

which are decent metrics for a traditional VGG model under such constraints.

The accuracy of the model running in FHE was not measured because of the computational cost it would require.
This is something we plan on measuring once FHE runtimes become more acceptable.

<!-- FIXME: Add more metrics: https://github.com/zama-ai/concrete-ml-internal/issues/2377 -->

<!-- FIXME: Add training scripts https://github.com/zama-ai/concrete-ml-internal/issues/2383 -->
