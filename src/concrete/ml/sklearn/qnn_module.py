"""Sparse Quantized Neural Network torch module."""
from typing import Set, Type

import brevitas.nn as qnn
import numpy
import torch
import torch.nn.utils.prune as pruning
from torch import nn

from ..common.debugging import assert_true
from ..common.utils import MAX_BITWIDTH_BACKWARD_COMPATIBLE


class SparseQuantNeuralNetwork(nn.Module):
    """Sparse Quantized Neural Network.

    This class implements an MLP that is compatible with FHE constraints. The weights and
    activations are quantized to low bit-width and pruning is used to ensure accumulators do not
    surpass an user-provided accumulator bit-width. The number of classes and number of layers
    are specified by the user, as well as the breadth of the network
    """

    # pylint: disable-next=too-many-arguments
    def __init__(
        self,
        input_dim: int,
        n_layers: int,
        n_outputs: int,
        n_hidden_neurons_multiplier: int = 4,
        n_w_bits: int = 3,
        n_a_bits: int = 3,
        n_accum_bits: int = MAX_BITWIDTH_BACKWARD_COMPATIBLE,
        n_prune_neurons_percentage: float = 0.0,
        activation_function: Type = nn.ReLU,
        quant_narrow: bool = False,
        quant_signed: bool = True,
    ):
        """Sparse Quantized Neural Network constructor.

        Args:
            input_dim (int): Number of dimensions of the input data.
            n_layers (int): Number of linear layers for this network.
            n_outputs (int): Number of output classes or regression targets.
            n_w_bits (int): Number of weight bits.
            n_a_bits (int): Number of activation and input bits.
            n_accum_bits (int): Maximal allowed bit-width of intermediate accumulators.
            n_hidden_neurons_multiplier (int): The number of neurons on the hidden will be the
                number of dimensions of the input multiplied by `n_hidden_neurons_multiplier`. Note
                that pruning is used to adjust the accumulator size to attempt to keep the maximum
                accumulator bit-width to `n_accum_bits`, meaning that not all hidden layer neurons
                will be active. The default value for `n_hidden_neurons_multiplier` is chosen for
                small dimensions of the input. Reducing this value decreases the FHE inference time
                considerably but also decreases the robustness and accuracy of model training.
            n_prune_neurons_percentage (float): The percentage of neurons to prune in the hidden
                layers. This can be used when setting `n_hidden_neurons_multiplier` with a high
                number (3-4), once good accuracy is obtained, in order to speed up the model in FHE.
            activation_function (Type): The activation function to use in the network
                (e.g., torch.ReLU, torch.SELU, torch.Sigmoid, ...).
            quant_narrow (bool): Whether this network should quantize the values using narrow range
                (e.g a 2-bits signed quantization uses [-1, 0, 1] instead of [-2, -1, 0, 1]).
            quant_signed (bool): Whether this network should quantize the values using signed
                integers.

        Raises:
            ValueError: If the parameters have invalid values or the computed accumulator bit-width
                is zero.
        """

        super().__init__()

        self.features = nn.Sequential()
        in_features = input_dim

        self.n_layers = n_layers

        if n_layers <= 0:
            raise ValueError(
                f"Invalid number of layers: {n_layers}, at least one intermediary layers is needed"
            )

        if n_w_bits <= 0 or n_a_bits <= 0:
            raise ValueError("The weight & activation quantization bit-width cannot be less than 1")

        for idx in range(n_layers):
            out_features = (
                n_outputs if idx == n_layers - 1 else int(input_dim * n_hidden_neurons_multiplier)
            )

            quant_name = f"quant{idx}"
            quantizer = qnn.QuantIdentity(
                bit_width=n_a_bits,
                return_quant_tensor=True,
                narrow_range=quant_narrow,
                signed=quant_signed,
            )

            layer_name = f"fc{idx}"
            layer = qnn.QuantLinear(
                in_features,
                out_features,
                True,
                weight_bit_width=n_w_bits,
                bias_quant=None,
                weight_narrow_range=quant_narrow,
                narrow_range=quant_narrow,
                signed=quant_signed,
            )

            self.features.add_module(quant_name, quantizer)
            self.features.add_module(layer_name, layer)

            if idx < n_layers - 1:
                self.features.add_module(f"act{idx}", activation_function())

            in_features = out_features

        self.n_w_bits = n_w_bits
        self.n_a_bits = n_a_bits
        self.n_accum_bits = n_accum_bits

        # Store input/output dimensions to check they are correct during .fit(X,y).
        # The X passed to .fit must not have different dimensions than the one given in this
        # constructor.
        self.n_outputs = n_outputs
        self.input_dim = input_dim

        self.n_prune_neurons_percentage = n_prune_neurons_percentage

        assert_true(
            self.n_prune_neurons_percentage >= 0 and self.n_prune_neurons_percentage < 1.0,
            "Pruning percentage must be expressed as a fraction between 0 and 1. A value of "
            " zero (0) means pruning is disabled",
        )
        self.pruned_layers: Set[nn.Module] = set()

        self.enable_pruning()

    def max_active_neurons(self) -> int:
        """Compute the maximum number of active (non-zero weight) neurons.

        The computation is done using the quantization parameters passed to the constructor.
        Warning: With the current quantization algorithm (asymmetric) the value returned by this
        function is not guaranteed to ensure FHE compatibility. For some weight distributions,
        weights that are 0 (which are pruned weights) will not be quantized to 0.
        Therefore the total number of active quantized neurons will not be equal to
        max_active_neurons.

        Returns:
            int: The maximum number of active neurons.
        """

        return int(
            numpy.floor(
                (2**self.n_accum_bits - 1) / (2**self.n_w_bits - 1) / (2**self.n_a_bits - 1)
            )
        )

    def make_pruning_permanent(self) -> None:
        """Make the learned pruning permanent in the network."""
        max_neuron_connections = self.max_active_neurons()

        prev_layer_keep_idxs = None
        layer_idx = 0
        # Iterate over all layers that have weights (Linear ones)
        for layer in self.features:
            if not isinstance(layer, (nn.Linear, qnn.QuantLinear)):
                continue

            layer_shape = layer.weight.shape

            # Compute the fan-in, the number of inputs to a neuron, the product of the kernel
            # width x height x in_channels.
            fan_in = numpy.prod(layer_shape[1:])

            # If this is a layer that should be pruned and is currently being pruned, make the
            # pruning permanent. This is done by multiplying the pruning mask tensor with the
            # weights and storing the result in the weight member.
            if layer in self.pruned_layers and fan_in > max_neuron_connections:
                pruning.remove(layer, "weight")
                self.pruned_layers.remove(layer)

            if self.n_prune_neurons_percentage > 0.0:
                weights = layer.weight.detach().numpy()

                # Pruning all layers except the last one, but for the last one
                # we still need to remove synapses of the previous layer's pruned neurons
                if layer_idx < self.n_layers - 1:
                    # Once pruning is disabled, the weights of some neurons become 0
                    # We need to find those neurons (columns in the weight matrix).
                    # Testing for floats equal to 0 is done using an epsilon
                    neurons_removed_idx = numpy.where(numpy.sum(numpy.abs(weights), axis=1) < 0.001)
                    idx = numpy.arange(weights.shape[0])
                    keep_idxs = numpy.setdiff1d(idx, neurons_removed_idx)
                else:
                    keep_idxs = numpy.arange(weights.shape[0])

                # Now we take the indices of the neurons kept for the previous layer
                # If this is the first layer all neurons are kept
                if prev_layer_keep_idxs is None:
                    prev_layer_keep_idxs = numpy.arange(weights.shape[1])

                # Remove the pruned neurons and the weights/synapses
                # that apply to neurons removed in the previous layer
                orig_weight = layer.weight.data.clone()
                transform_weight = orig_weight[keep_idxs]
                transform_weight = transform_weight[:, prev_layer_keep_idxs]

                # Replace the weight matrix of the current layer
                layer.weight = torch.nn.Parameter(transform_weight)

                # Eliminate the biases of the neurons that were removed in this layer
                if layer.bias is not None:
                    orig_bias = layer.bias.data.clone()
                    transform_bias = orig_bias[keep_idxs]
                    layer.bias = torch.nn.Parameter(transform_bias)

                # Save the indices of the neurons removed in this layer to
                # remove synapses in the next layer
                prev_layer_keep_idxs = keep_idxs

            layer_idx += 1

        assert_true(
            layer_idx == self.n_layers,
            "Not all layers in the network were examined as candidates for pruning",
        )

    def enable_pruning(self) -> None:
        """Enable pruning in the network. Pruning must be made permanent to recover pruned weights.

        Raises:
            ValueError: If the quantization parameters are invalid.
        """
        max_neuron_connections = self.max_active_neurons()

        if max_neuron_connections == 0:
            raise ValueError(
                "The maximum accumulator bit-width is too low "
                "for the quantization parameters requested. No neurons would be created in the "
                "requested configuration"
            )

        # Iterate over all layers that have weights (Linear ones)
        layer_idx = 0
        for layer in self.features:
            if not isinstance(layer, (nn.Linear, qnn.QuantLinear)):
                continue

            layer_shape = layer.weight.shape

            # Compute the fan-in, the number of inputs to a neuron, and the fan-out, the number of
            # neurons in the current layer.
            # The fan-in is the product of the kernel width x height x in_channels while the fan-out
            # is out_channels
            fan_in = numpy.prod(layer_shape[1:])
            fan_out = layer_shape[0]

            # To satisfy accumulator bit-width constraints each dot-product between an input line
            # and weight column must not exceed n_accum_bits bits. We thus prune the layer to have
            # at most max_neuron_connections non-zero weights
            if fan_in > max_neuron_connections and layer not in self.pruned_layers:
                pruning.l1_unstructured(
                    layer, "weight", (fan_in - max_neuron_connections) * fan_out
                )
                self.pruned_layers.add(layer)

            # If pruning is enabled, which is generally the case during training, a
            # forward hook is added in order to create a mask tensor (made of 0 or 1) that will
            # be multiplied with the weights during the forward pass.
            # This is done for all layers except the last one, which outputs the prediction
            if layer_idx < self.n_layers - 1 and self.n_prune_neurons_percentage > 0.0:
                # Use L2-norm structured pruning, using the torch ln_structured
                # function, with norm=2 and axis=0 (output/neuron axis)
                pruning.ln_structured(layer, "weight", self.n_prune_neurons_percentage, 2, 0)

            # Note this is counting only Linear layers
            layer_idx += 1

        assert_true(
            layer_idx == self.n_layers,
            "Not all layers in the network were examined as candidates for pruning",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): network input

        Returns:
            x (torch.Tensor): network prediction
        """
        return self.features(x)
