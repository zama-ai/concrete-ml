# Pruning

Pruning is a method to reduce neural network complexity, usually applied in order reduce the computation cost or memory size. Pruning is used in Concrete-ML to control the size of accumulators in neural networks, thus making them FHE compatible. See [here](../getting-started/concrete_numpy.md#limitations-for-fhe-friendly-neural-networks) for an explanation of the accumulator bitwidth constraints.

In neural networks, a neuron computes a linear combination of inputs and learned weights, then applies an activation function.

![Artificial Neuron (from: wikipedia)](../figures/Artificial_neuron.png)

The neuron computes:

$$y_k = \phi\left(\sum_i w_ix_i\right)$$

When building a full neural network, each layer will contain multiple neurons, which are connected to the neuron outputs of a previous layer or to the inputs.

![Fully Connected Neural Network](../figures/network.png)

For every neuron shown in each layer of the figure above, the linear combinations of inputs and learned weights are computed. Depending on the values of the inputs and weights, the sum $$v_k = \sum_i w_ix_i$$ - which for Concrete-ML neural networks is computed with integers - can take a range of different values.

To respect the bit width constraint of the FHE [Table Lookup](https://docs.zama.ai/concrete-numpy/tutorials/table_lookup), the values of the accumulator $$v_k$$ must remain small to be representable with only 8 bits. In other words, the values must be between 0 and 255.

Pruning a neural network entails fixing some of the weights $$w_k$$ to be zero during training. This is advantageous to meet FHE constraints, as irrespective of the distribution of $$x_i$$, multiplying these input values by 0 does not increase the accumulator value.

Fixing some of the weights to 0 makes the network graph look more similar to the following:

![Pruned Fully Connected Neural Network](../figures/prunednet.png)

While pruning weights can reduce the prediction performance of the neural network, studies show that a high level of pruning (above 50% \[^1\]) can often be applied. See here how Concrete-ML uses pruning in [Fully Connected Neural Networks](../_apidoc/concrete.ml.sklearn.html#concrete.ml.sklearn.qnn.NeuralNetClassifier).

\[^1\]: Han, Song & Pool, Jeff & Tran, John & Dally, William. (2015). Learning both Weights and Connections for Efficient Neural Networks.





