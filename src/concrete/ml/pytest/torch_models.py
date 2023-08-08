"""Torch modules for our pytests."""

# pylint: disable=too-many-lines
from typing import Union

import brevitas.nn as qnn
import numpy
import torch
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat
from torch import nn
from torch.nn.utils import prune

# pylint: disable=too-many-lines


class SimpleNet(torch.nn.Module):
    """Fake torch model used to generate some onnx."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = 2.2
        self.offset = 1.1

    def forward(self, inputs):
        """Forward function.

        Arguments:
            inputs: the inputs of the model.

        Returns:
            torch.Tensor: the result of the computation
        """
        res = (inputs * self.scale) + self.offset
        res = torch.relu(inputs)
        return res


class FCSmall(nn.Module):
    """Torch model for the tests."""

    def __init__(self, input_output, activation_function):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_output, out_features=input_output)
        self.act_f = activation_function()
        self.fc2 = nn.Linear(in_features=input_output, out_features=input_output)

    def forward(self, x):
        """Forward pass.

        Args:
            x: the input of the NN

        Returns:
            the output of the NN
        """
        out = self.fc1(x)
        out = self.act_f(out)
        out = self.fc2(out)

        return out


class FC(nn.Module):
    """Torch model for the tests."""

    def __init__(self, activation_function, input_output=32 * 32 * 3):
        super().__init__()
        self.fc1 = nn.Linear(in_features=input_output, out_features=128)
        self.act_1 = activation_function()
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.act_2 = activation_function()
        self.fc3 = nn.Linear(in_features=64, out_features=64)
        self.act_3 = activation_function()
        self.fc4 = nn.Linear(in_features=64, out_features=64)
        self.act_4 = activation_function()
        self.fc5 = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        """Forward pass.

        Args:
            x: the input of the NN

        Returns:
            the output of the NN
        """
        out = self.fc1(x)
        out = self.act_1(out)
        out = self.fc2(out)
        out = self.act_2(out)
        out = self.fc3(out)
        out = self.act_3(out)
        out = self.fc4(out)
        out = self.act_4(out)
        out = self.fc5(out)

        return out


class CNN(nn.Module):
    """Torch CNN model for the tests."""

    def __init__(self, input_output, activation_function):
        super().__init__()
        self.conv1 = nn.Conv2d(input_output, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.act_f = activation_function()

    def forward(self, x):
        """Forward pass.

        Args:
            x: the input of the NN

        Returns:
            the output of the NN
        """
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.act_f(self.fc1(x))
        x = self.act_f(self.fc2(x))
        x = self.fc3(x)
        return x


class CNNMaxPool(nn.Module):
    """Torch CNN model for the tests with a max pool."""

    def __init__(self, input_output, activation_function):
        super().__init__()
        self.conv1 = nn.Conv2d(input_output, 6, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.act_f = activation_function()

    def forward(self, x):
        """Forward pass.

        Args:
            x: the input of the NN

        Returns:
            the output of the NN
        """
        x = self.maxpool(torch.relu(self.conv1(x)))
        x = self.maxpool(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.act_f(self.fc1(x))
        x = self.act_f(self.fc2(x))
        x = self.fc3(x)
        return x


class CNNOther(nn.Module):
    """Torch CNN model for the tests."""

    def __init__(self, input_output, activation_function):
        super().__init__()

        self.activation_function = activation_function()
        self.conv1 = nn.Conv2d(input_output, 3, 3, stride=1, padding=1)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 3, 1)
        self.fc1 = nn.Linear(3 * 3 * 3, 5)
        self.fc2 = nn.Linear(5, 3)
        self.fc3 = nn.Linear(3, 2)

    def forward(self, x):
        """Forward pass.

        Args:
            x: the input of the NN

        Returns:
            the output of the NN

        """
        x = self.pool(self.activation_function(self.conv1(x)))
        x = self.activation_function(self.conv2(x))
        x = x.flatten(1)
        x = self.activation_function(self.fc1(x))
        x = self.activation_function(self.fc2(x))
        x = self.fc3(x)
        return x


class CNNInvalid(nn.Module):
    """Torch CNN model for the tests."""

    def __init__(self, activation_function, groups):
        super().__init__()

        padding = True

        self.activation_function = activation_function()
        self.flatten_function = lambda x: torch.flatten(x, 1)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.AvgPool2d(2, 2, padding=1) if padding else nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, groups=2) if groups else nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * (5 + padding * 1) * (5 + padding * 1), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.gather_slice = True

    def forward(self, x):
        """Forward pass.

        Args:
            x: the input of the NN

        Returns:
            the output of the NN
        """
        x = self.pool(self.activation_function(self.conv1(x)))
        x = self.pool(self.activation_function(self.conv2(x)))
        x = self.flatten_function(x)
        x = self.activation_function(self.fc1(x))
        x = self.activation_function(self.fc2(x))
        x = self.fc3(x)
        # Produce a Gather and Slice which are not supported
        if self.gather_slice:
            x = x[0, 0:-1:2]
        x = torch.mean(x)
        return x


class CNNGrouped(nn.Module):
    """Torch CNN model with grouped convolution for compile torch tests."""

    def __init__(self, input_output, activation_function, groups):
        super().__init__()

        self.activation_function = activation_function()
        self.conv1 = nn.Conv2d(input_output, 3, 3, stride=1, padding=1, dilation=1, groups=groups)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 3, 1, stride=1, padding=0, dilation=1, groups=3)
        self.fc1 = nn.Linear(3 * 3 * 3, 5)
        self.fc2 = nn.Linear(5, 3)
        self.fc3 = nn.Linear(3, 2)

    def forward(self, x):
        """Forward pass.

        Args:
            x: the input of the NN

        Returns:
            the output of the NN
        """
        x = self.pool(self.activation_function(self.conv1(x)))
        x = self.activation_function(self.conv2(x))
        x = x.flatten(1)
        x = self.activation_function(self.fc1(x))
        x = self.activation_function(self.fc2(x))
        x = self.fc3(x)
        return x


class NetWithLoops(torch.nn.Module):
    """Torch model, where we reuse some elements in a loop.

    Torch model, where we reuse some elements in a loop in the forward and don't expect the
    user to define these elements in a particular order.
    """

    def __init__(self, activation_function, input_output, n_fc_layers):
        super().__init__()
        self.fc1 = nn.Linear(input_output, 3)
        self.ifc = nn.Sequential()
        for i in range(n_fc_layers):
            self.ifc.add_module(f"fc{i+1}", nn.Linear(3, 3))
        self.out = nn.Linear(3, 1)
        self.act = activation_function()

    def forward(self, x):
        """Forward pass.

        Args:
            x: the input of the NN

        Returns:
            the output of the NN
        """
        x = self.act(self.fc1(x))
        for m in self.ifc:
            x = self.act(m(x))
        x = self.act(self.out(x))

        return x


class MultiInputNN(nn.Module):
    """Torch model to test multiple inputs forward."""

    def __init__(self, input_output, activation_function):  # pylint: disable=unused-argument
        super().__init__()
        self.act = activation_function()

    def forward(self, x, y):
        """Forward pass.

        Args:
            x: the first input of the NN
            y: the second input of the NN

        Returns:
            the output of the NN
        """
        return self.act(x + y)


class MultiInputNNConfigurable(nn.Module):
    """Torch model to test multiple inputs forward."""

    layer1: nn.Module
    layer2: nn.Module

    def __init__(self, use_conv, use_qat, input_output, n_bits):  # pylint: disable=unused-argument
        super().__init__()

        if use_conv:
            self.layer1 = nn.Conv2d(input_output[0], input_output[0], 1, 1, 0)
            self.layer2 = nn.Conv2d(input_output[0], input_output[0], 1, 1, 0)
        else:
            self.layer1 = nn.Linear(input_output, input_output)
            self.layer2 = nn.Linear(input_output, input_output)

    def forward(self, x, y):
        """Forward pass.

        Args:
            x: the first input of the NN
            y: the second input of the NN

        Returns:
            the output of the NN
        """
        x = self.layer1(x)
        y = self.layer2(y)
        return self.layer1(x + y)


class MultiInputNNDifferentSize(nn.Module):
    """Torch model to test multiple inputs with different shape in the forward pass."""

    def __init__(
        self,
        input_output,
        activation_function=None,
        is_brevitas_qat=False,
        n_bits=3,
    ):  # pylint: disable=unused-argument
        super().__init__()

        # input_output is expected to be a list of two integers representing in and out features
        # for both x and y
        if is_brevitas_qat:
            # n_bits is used for quantizing both the inputs and the weights, therefore we need
            # to make sure that it is at least 2 bits
            assert n_bits > 1, "Weights cannot be quantized over a single bit"

            self.quant1 = qnn.QuantIdentity(bit_width=n_bits)
            self.quant2 = qnn.QuantIdentity(bit_width=n_bits)
            self.quant3 = qnn.QuantIdentity(bit_width=n_bits)
            self.layer1 = qnn.QuantLinear(
                input_output[0], input_output[0], bias=False, weight_bit_width=n_bits
            )
            self.layer2 = qnn.QuantLinear(
                input_output[1], input_output[0], bias=False, weight_bit_width=n_bits
            )

        else:
            self.layer1 = nn.Linear(input_output[0], input_output[0])
            self.layer2 = nn.Linear(input_output[1], input_output[0])

        self.is_brevitas_qat = is_brevitas_qat

    def forward(self, x, y):
        """Forward pass.

        Args:
            x: The first input of the NN.
            y: The second input of the NN.

        Returns:
            The output of the NN.
        """
        if self.is_brevitas_qat:
            x = self.layer1(self.quant1(x))
            y = self.layer2(self.quant2(y))
            return self.layer1(self.quant3(x + y))

        x = self.layer1(x)
        y = self.layer2(y)
        return self.layer1(x + y)


class BranchingModule(nn.Module):
    """Torch model with some branching and skip connections."""

    # pylint: disable-next=unused-argument
    def __init__(self, input_output, activation_function):
        super().__init__()

        self.act = activation_function()

    def forward(self, x):
        """Forward pass.

        Args:
            x: the input of the NN

        Returns:
            the output of the NN
        """
        return x + self.act(x + 1.0) - self.act(x * 2.0)


class BranchingGemmModule(nn.Module):
    """Torch model with some branching and skip connections."""

    def __init__(self, input_output, activation_function):
        super().__init__()

        self.act = activation_function()
        self.fc1 = nn.Linear(input_output, input_output)

    def forward(self, x):
        """Forward pass.

        Args:
            x: the input of the NN

        Returns:
            the output of the NN
        """
        return x + self.act(x + 1.0) - self.act(self.fc1(x * 2.0))


class UnivariateModule(nn.Module):
    """Torch model that calls univariate and shape functions of torch."""

    # pylint: disable-next=unused-argument
    def __init__(self, input_output, activation_function):
        super().__init__()

        self.act = activation_function()

    def forward(self, x):
        """Forward pass.

        Args:
            x: the input of the NN

        Returns:
            the output of the NN
        """
        x = x.view(-1, 1)
        x = torch.reshape(x, (-1, 1))
        x = x.flatten(1)
        x = self.act(torch.abs(torch.exp(torch.log(1.0 + torch.sigmoid(x)))))
        return x


class StepActivationModule(nn.Module):
    """Torch model implements a step function that needs Greater, Cast and Where."""

    # pylint: disable-next=unused-argument
    def __init__(self, input_output, activation_function):
        super().__init__()

        self.act = activation_function()

    def forward(self, x):
        """Forward pass with a quantizer built into the computation graph.

        Args:
            x: the input of the NN

        Returns:
            the output of the NN
        """

        def step(x, bias):
            """Forward-step function for quantization.

            Args:
                x: the input
                bias: the bias

            Returns:
                one step further

            """
            y = torch.zeros_like(x)
            mask = torch.gt(x - bias, 0.0)
            y[mask] = 1.0
            return y

        x = step(x, 0.5) * 2.0
        x = self.act(x)
        return x


class NetWithConcatUnsqueeze(torch.nn.Module):
    """Torch model to test the concat and unsqueeze operators."""

    def __init__(self, activation_function, input_output, n_fc_layers):
        super().__init__()
        self.fc1 = nn.Linear(input_output, 3)
        self.ifc = nn.Sequential()
        for i in range(n_fc_layers):
            self.ifc.add_module(f"fc{i+1}", nn.Linear(3, 3))
        self.out = nn.Linear(3, 1)
        self.act = activation_function()

    def forward(self, x):
        """Forward pass.

        Args:
            x: the input of the NN

        Returns:
            the output of the NN
        """
        x = self.act(self.fc1(x))
        results = []
        for module in self.ifc:
            results.append(module(x))

        # Use torch.stack which creates Unsqueeze operators for each module
        # and a final Concat operator.
        return torch.stack(results)


class MultiOpOnSingleInputConvNN(nn.Module):
    """Network that applies two quantized operations on a single input."""

    def __init__(self, can_remove_input_tlu: bool):
        super().__init__()
        self.can_remove_input_tlu = can_remove_input_tlu
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(1, 8, 3)

    def forward(self, x):
        """Forward pass.

        Args:
            x: the input of the NN

        Returns:
            the output of the NN
        """

        # The quantizer for this network can be moved in the clear if the
        # input is fed directly to two conv layers that have the same quantizer.
        # To ensure the quantizer is performed in FHE, a univariate op is applied
        # before _one_ of the convolutions
        y = x if self.can_remove_input_tlu else torch.sigmoid(x)
        layer1_out = torch.relu(self.conv1(y))
        layer2_out = torch.relu(self.conv2(x))
        return layer1_out + layer2_out


class FCSeq(nn.Module):
    """Torch model that should generate MatMul->Add ONNX patterns.

    This network generates additions with a constant scalar
    """

    def __init__(self, input_output, act):
        super().__init__()
        self.feat = nn.Sequential()
        in_features = input_output
        self.n_layers = 2
        self.biases = [torch.Tensor(size=(1,)) for _ in range(self.n_layers)]
        for b in self.biases:
            nn.init.uniform_(b)

        for idx in range(self.n_layers):
            out_features = in_features if idx == self.n_layers - 1 else in_features
            layer_name = f"fc{idx}"
            layer = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
            self.feat.add_module(layer_name, layer)
            in_features = out_features

        self.act = act()

    def forward(self, x):
        """Forward pass.

        Args:
            x: the input of the NN

        Returns:
            the output of the NN
        """
        for idx, layer in enumerate(self.feat):
            x = self.act(layer(x) + self.biases[idx])
        return x


class FCSeqAddBiasVec(nn.Module):
    """Torch model that should generate MatMul->Add ONNX patterns.

    This network tests the addition with a constant vector
    """

    def __init__(self, input_output, act):
        super().__init__()
        self.feat = nn.Sequential()
        in_features = input_output
        self.n_layers = 2
        self.biases = [torch.Tensor(size=(input_output,)) for _ in range(self.n_layers)]
        for b in self.biases:
            nn.init.uniform_(b)

        for idx in range(self.n_layers):
            out_features = in_features if idx == self.n_layers - 1 else in_features
            layer_name = f"fc{idx}"
            layer = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
            self.feat.add_module(layer_name, layer)
            in_features = out_features

        self.act = act()

    def forward(self, x):
        """Forward pass.

        Args:
            x: the input of the NN

        Returns:
            the output of the NN
        """
        for idx, layer in enumerate(self.feat):
            x = self.act(layer(x) + self.biases[idx])
        return x


class TinyCNN(nn.Module):
    """A very small CNN."""

    def __init__(self, n_classes, act) -> None:
        """Create the tiny CNN with two conv layers.

        Args:
            n_classes: number of classes
            act: the activation
        """
        super().__init__()

        self.conv1 = nn.Conv2d(1, 2, 2, stride=1, padding=0)
        self.avg_pool1 = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(2, n_classes, 2, stride=1, padding=0)
        self.act = act()
        self.n_classes = n_classes

    def forward(self, x):
        """Forward the two layers with the chosen activation function.

        Args:
            x: the input of the NN

        Returns:
            the output of the NN
        """
        x = self.act(self.avg_pool1(self.conv1(x)))
        x = self.act(self.conv2(x))
        return x


class TinyQATCNN(nn.Module):
    """A very small QAT CNN to classify the sklearn digits data-set.

    This class also allows pruning to a maximum of 10 active neurons, which
    should help keep the accumulator bit-width low.
    """

    def __init__(self, n_classes, n_bits, n_active, signed, narrow) -> None:
        """Construct the CNN with a configurable number of classes.

        Args:
            n_classes (int): number of outputs of the neural net
            n_bits (int): number of weight and activation bits for quantization
            n_active (int): number of active (non-zero weight) neurons to keep
            signed (bool): whether quantized integer values are signed
            narrow (bool): whether the range of quantized integer values is narrow/symmetric
        """
        super().__init__()

        a_bits = n_bits
        w_bits = n_bits

        self.n_active = n_active

        q_args = {"signed": signed, "narrow_range": narrow}

        self.quant1 = qnn.QuantIdentity(bit_width=a_bits, return_quant_tensor=True, **q_args)
        self.conv1 = qnn.QuantConv2d(
            1, 2, 3, stride=1, padding=0, weight_bit_width=w_bits, **q_args
        )
        self.quant2 = qnn.QuantIdentity(bit_width=a_bits, return_quant_tensor=True, **q_args)
        self.conv2 = qnn.QuantConv2d(
            2, 3, 3, stride=2, padding=0, weight_bit_width=w_bits, **q_args
        )
        self.quant3 = qnn.QuantIdentity(bit_width=a_bits, return_quant_tensor=True, **q_args)
        self.conv3 = qnn.QuantConv2d(
            3, 16, 2, stride=1, padding=0, weight_bit_width=w_bits, **q_args
        )

        self.quant4 = qnn.QuantIdentity(bit_width=a_bits, return_quant_tensor=True, **q_args)
        self.fc1 = qnn.QuantLinear(16, n_classes, weight_bit_width=3, bias=True, **q_args)

        # Enable pruning, prepared for training
        self.toggle_pruning(True)

    def toggle_pruning(self, enable):
        """Enable or remove pruning.

        Args:
            enable: if we enable the pruning or not

        """

        # Maximum number of active neurons (i.e., corresponding weight != 0)

        # Go through all the convolution layers
        for layer in (self.conv1, self.conv2, self.conv3):
            s = layer.weight.shape

            # Compute fan-in (number of inputs to a neuron)
            # and fan-out (number of neurons in the layer)
            layer_size = [s[0], numpy.prod(s[1:])]

            # The number of input neurons (fan-in) is the product of
            # the kernel width x height x inChannels.
            if layer_size[1] > self.n_active:
                if enable:
                    # This will create a forward hook to create a mask tensor that is multiplied
                    # with the weights during forward. The mask will contain 0s or 1s
                    prune.l1_unstructured(
                        layer, "weight", (layer_size[1] - self.n_active) * layer_size[0]
                    )
                else:
                    # When disabling pruning, the mask is multiplied with the weights
                    # and the result is stored in the weights member
                    prune.remove(layer, "weight")

    def forward(self, x):
        """Run inference on the tiny CNN, apply the decision layer on the reshaped conv output.

        Args:
            x: the input to the NN

        Returns:
            the output of the NN

        """

        x = self.quant1(x)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.quant2(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.quant3(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.quant4(x)
        x = x.view(-1, 16)
        x = self.fc1(x)
        return x

    def test_torch(self, test_loader):
        """Test the network: measure accuracy on the test set.

        Args:
            test_loader: the test loader

        Returns:
            res: the number of correctly classified test examples

        """

        # Freeze normalization layers
        self.eval()

        all_y_pred = numpy.zeros((len(test_loader)), dtype=numpy.int64)
        all_targets = numpy.zeros((len(test_loader)), dtype=numpy.int64)

        # Iterate over the batches
        idx = 0
        for data, target in test_loader:
            # Accumulate the ground truth labels
            endidx = idx + target.shape[0]
            all_targets[idx:endidx] = target.numpy()

            # Run forward and get the raw predictions first
            raw_pred = self(data).detach().numpy()

            # Get the predicted class id, handle NaNs
            if numpy.any(numpy.isnan(raw_pred)):
                output = -1  # pragma: no cover
            else:
                output = raw_pred.argmax(1)

            all_y_pred[idx:endidx] = output

            idx += target.shape[0]

        # Print out the accuracy as a percentage
        n_correct = numpy.sum(all_targets == all_y_pred)
        return n_correct


class SimpleQAT(nn.Module):
    """Torch model implements a step function that needs Greater, Cast and Where."""

    def __init__(self, input_output, activation_function, n_bits=2, disable_bit_check=False):
        super().__init__()

        self.act = activation_function()
        self.fc1 = nn.Linear(input_output, input_output)

        # Create pre-quantized weights
        # Note the weights in the network are not integers, but uniformly spaced float values
        # that are selected from a discrete set
        weight_scale = 1.5

        n_bits_weights = n_bits

        # Generate the pattern 0, 1, ..., 2^N-1, 0, 1, .. 2^N-1, 0, 1..
        all_weights = numpy.mod(
            numpy.arange(numpy.prod(self.fc1.weight.shape)), 2**n_bits_weights
        )

        # Shuffle the pattern and reshape to weight shape
        numpy.random.shuffle(all_weights)
        int_weights = all_weights.reshape(self.fc1.weight.shape)

        # A check that is used to ensure this class is used correctly
        # but we may want to disable it to check that the QAT import catches the error
        if not disable_bit_check:
            # Ensure we have the correct max/min that produces the correct scale in Quantized Array
            assert numpy.max(int_weights) - numpy.min(int_weights) == (2**n_bits_weights - 1)

        # We want signed weights, so offset the generated weights
        int_weights = int_weights - 2 ** (n_bits_weights - 1)

        # Initialize with scaled float weights
        self.fc1.weight.data = torch.from_numpy(int_weights * weight_scale).float()

        self.n_bits = n_bits

    def forward(self, x):
        """Forward pass with a quantizer built into the computation graph.

        Args:
            x: the input of the NN

        Returns:
            the output of the NN
        """

        def step(x, bias):
            """Forward-step function for quantization.

            Args:
                x: the input of the layer
                bias: the bias

            Returns:
                the output of the layer

            """
            y = torch.zeros_like(x)
            mask = torch.gt(x - bias, 0.0)
            y[mask] = 1.0
            return y

        # A step quantizer with steps at -5, 0, 5, ...
        # For example at n_bits == 2
        #         /  0  if x < -5                \
        # f(x) = {   5  if x >= 0 and x < 5       }
        #         \  10 if x >= 5 and x < 10     /
        #          \ 15 if x >= 10              /

        x_q = step(x, -5)
        for i in range(1, 2**self.n_bits - 1):
            x_q += step(x, (i - 1) * 5)

        x_q = x_q.mul(5)

        result_fc1 = self.fc1(x_q)

        return self.act(result_fc1)


class QATTestModule(nn.Module):
    """Torch model that implements a simple non-uniform quantizer."""

    def __init__(self, activation_function):
        super().__init__()

        self.act = activation_function()

    def forward(self, x):
        """Forward pass with a quantizer built into the computation graph.

        Args:
            x: the input of the NN

        Returns:
            the output of the NN
        """

        def step(x, bias):
            """Forward-step function for quantization.

            Args:
                x: the input of the layer
                bias: the bias

            Returns:
                the output of the layer
            """
            y = torch.zeros_like(x)
            mask = torch.gt(x - bias, 0.0)
            y[mask] = 1.0
            return y

        x = step(x, 0.5) * 2.0
        x = self.act(x)
        return x


class SingleMixNet(nn.Module):
    """Torch model that with a single conv layer that produces the output, e.g., a blur filter."""

    mixing_layer: Union[nn.Module, nn.Sequential]

    def __init__(self, use_conv, use_qat, inp_size, n_bits):
        super().__init__()

        if use_conv:
            # Initialize a blur filter float weights
            np_weights = numpy.asarray([[[[1, 1, 1], [1, 4, 1], [1, 1, 1]]]])

            if use_qat:
                self.mixing_layer = nn.Sequential(
                    qnn.QuantIdentity(bit_width=n_bits),
                    qnn.QuantConv2d(1, 1, 3, stride=1, bias=True, weight_bit_width=n_bits),
                )
                layer_obj = self.mixing_layer[1]
            else:
                self.mixing_layer = nn.Conv2d(1, 1, 3, stride=1, bias=True)
                layer_obj = self.mixing_layer
        else:
            # Initialize a linear layer with 1s
            np_weights = numpy.asarray([[1] * inp_size])
            if use_qat:
                self.mixing_layer = nn.Sequential(
                    qnn.QuantIdentity(bit_width=n_bits),
                    qnn.QuantLinear(inp_size, inp_size, bias=True, weight_bit_width=n_bits),
                )
                layer_obj = self.mixing_layer[1]
            else:
                self.mixing_layer = nn.Linear(inp_size, inp_size, bias=True)
                layer_obj = self.mixing_layer

        layer_obj.weight.data = torch.from_numpy(np_weights).float()
        layer_obj.bias.data = torch.rand(size=(1,))

    def forward(self, x):
        """Execute the single convolution.

        Args:
            x: the input of the NN

        Returns:
            the output of the NN
        """

        return self.mixing_layer(x)


class DoubleQuantQATMixNet(nn.Module):
    """Torch model that with two different quantizers on the input.

    Used to test that it keeps the input TLU.
    """

    def __init__(self, use_conv, use_qat, inp_size, n_bits):  # pylint: disable=unused-argument
        super().__init__()

        # A first quantizer
        self.quant1 = qnn.QuantIdentity(bit_width=n_bits)
        # A different quantizer
        self.quant2 = qnn.QuantIdentity(bit_width=n_bits + 1)
        self.quant3 = qnn.QuantIdentity(bit_width=n_bits)

        if use_conv:
            self.mixing_layer = qnn.QuantConv2d(
                1, 1, 3, stride=1, bias=True, weight_bit_width=n_bits
            )
        else:
            self.mixing_layer = qnn.QuantLinear(
                inp_size, inp_size, bias=True, weight_bit_width=n_bits
            )

    def forward(self, x):
        """Execute the single convolution.

        Args:
            x: the input of the NN

        Returns:
            the output of the NN
        """

        left_x1 = self.quant1(x)
        right_x2 = self.quant2(x)
        return self.mixing_layer(self.quant3(left_x1 + right_x2))


class TorchSum(nn.Module):
    """Torch model to test the ReduceSum ONNX operator in a leveled circuit."""

    def __init__(self, dim=(0,), keepdim=True):
        """Initialize the module.

        Args:
            dim (Tuple[int]): The axis along which the sum should be executed
            keepdim (bool): If the output should keep the same dimension as the input or not
        """
        # Torch sum doesn't seem to handle dim=None, as opposed to its documentation. Instead, the
        # op should be called without it. Additionally, in this exact case, keepdim parameter is not
        # supported and the op behaves as if it has been set to False.
        assert dim is not None, "Dim parameter should not be set to None."

        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.tensor): The input of the model

        Returns:
            torch_sum (torch.tensor): The sum of the input's tensor elements along the given axis
        """
        torch_sum = x.sum(dim=self.dim, keepdim=self.keepdim)
        return torch_sum


class TorchSumMod(TorchSum):
    """Torch model to test the ReduceSum ONNX operator in a circuit containing a PBS."""

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.tensor): The input of the model

        Returns:
            torch_sum (torch.tensor): The sum of the input's tensor elements along the given axis
        """
        torch_sum = x.sum(dim=self.dim, keepdim=self.keepdim)

        # Add an additional operator that requires a TLU in order to force this circuit to
        # handle a PBS without actually changing the results
        torch_sum = torch_sum + torch_sum % 2 - torch_sum % 2
        return torch_sum


class NetWithConstantsFoldedBeforeOps(nn.Module):
    """Torch QAT model that does not quantize the inputs."""

    def __init__(
        self,
        hparams: dict,
        bits: int,
        act_quant=Int8ActPerTensorFloat,
        weight_quant=Int8WeightPerTensorFloat,
    ):
        super().__init__()
        self.hparams = hparams
        self.dense1 = qnn.QuantLinear(
            hparams["n_feats"],
            hparams["hidden_dim"],
            weight_quant=weight_quant,
            weight_bit_width=bits,
            bias=True,
        )
        self.dp1 = qnn.QuantDropout(0.1)
        self.act1 = qnn.QuantReLU(act_quant=act_quant, bit_width=bits)

        self.dense2 = qnn.QuantLinear(
            hparams["hidden_dim"],
            1,
            weight_bit_width=bits,
            weight_quant=weight_quant,
            bias=True,
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.tensor): The input of the model

        Returns:
            torch.tensor: Output of the network
        """

        # Note here that the input is not quantized, it is passed to the linear layer directly
        x = self.dense1(x)
        x = self.dp1(x)
        x = self.act1(x)

        x = self.dense2(x)
        return x


class ShapeOperationsNet(nn.Module):
    """Torch QAT model that reshapes the input."""

    def __init__(self, is_qat):
        super().__init__()
        self.is_qat = is_qat
        if is_qat:
            self.input_quant = qnn.QuantIdentity(bit_width=8)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.tensor): The input of the model

        Returns:
            torch.tensor: Output of the network
        """

        def shufflenet_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
            """Shuffle the channels: split them into two groups and then recombine them.

            In ShuffleNet, conv ops operate over the two branches, but here they are skipped
            for simplicity.

            Args:
                x (torch.Tensor): the tensor that will be shuffled
                groups (int): the number of groups of channels to

            Returns:
                torch.Tensor: the tensor containing the groups of channels of the input tensor
            """

            batchsize, num_channels, height, width = x.size()
            channels_per_group = num_channels // groups

            x = x.reshape(batchsize, groups, channels_per_group, height, width)
            x = torch.transpose(x, 1, 2).contiguous()
            x = x.reshape(batchsize, -1, height, width)

            return x

        if self.is_qat:
            x = self.input_quant(x)
        chunk1, chunk2 = x.chunk(2, dim=1)
        out = torch.cat((chunk1, chunk2), dim=1)
        return shufflenet_shuffle(out, 2)


class PaddingNet(nn.Module):
    """Torch QAT model that applies various padding patterns."""

    def __init__(self):
        super().__init__()

        # Use a QAT network to allow the torch result to be the same as the Concrete ML result
        self.input_quant = qnn.QuantIdentity(bit_width=8)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.tensor): The input of the model

        Returns:
            torch.tensor: Output of the network
        """

        # Quantize input
        x = self.input_quant(x)

        # Torch pads starting from the last dimensions of the tensor moving forward, with
        # potentially different padding at the start/end of the axes
        # for example a 4d tensor NCHW, padded with [1, 2, 2, 3] is padded
        # along the last 2 dimensions, with 1 cell to the left and 2 to the right (dimension 4: W)
        # and 2 cells at the top and 3 at the bottom (dimension 3: H)
        x = torch.nn.functional.pad(x, (3, 2))
        x = torch.nn.functional.pad(x, (1, 2, 3, 4))

        # Concrete ML only supports padding on the last two dimensions as this is the
        # most common setting
        x = torch.nn.functional.pad(x, (1, 1, 2, 2, 0, 0, 0, 0))
        return x


class QuantCustomModel(nn.Module):
    """A small quantized network with Brevitas, trained on make_classification."""

    def __init__(
        self,
        input_shape: int,
        output_shape: int,
        hidden_shape: int = 100,
        n_bits: int = 5,
        act_quant=Int8ActPerTensorFloat,
        weight_quant=Int8WeightPerTensorFloat,
    ):
        """Quantized Torch Model with Brevitas.

        Args:
            input_shape (int): Input size
            output_shape (int): Output size
            hidden_shape (int): Hidden size
            n_bits (int): Bit of quantization
            weight_quant (brevitas.quant): Quantization protocol of weights
            act_quant (brevitas.quant): Quantization protocol of activations.

        """
        super().__init__()

        self.quant_input = qnn.QuantIdentity(
            bit_width=n_bits, act_quant=act_quant, return_quant_tensor=True
        )
        self.linear1 = qnn.QuantLinear(
            in_features=input_shape,
            out_features=hidden_shape,
            weight_bit_width=n_bits,
            weight_quant=weight_quant,
            bias=True,
            return_quant_tensor=True,
        )

        self.relu1 = qnn.QuantReLU(return_quant_tensor=True, bit_width=n_bits, act_quant=act_quant)
        self.linear2 = qnn.QuantLinear(
            in_features=hidden_shape,
            out_features=hidden_shape,
            weight_bit_width=n_bits,
            weight_quant=weight_quant,
            bias=True,
            return_quant_tensor=True,
        )

        self.relu2 = qnn.QuantReLU(return_quant_tensor=True, bit_width=n_bits, act_quant=act_quant)

        self.linear3 = qnn.QuantLinear(
            in_features=hidden_shape,
            out_features=output_shape,
            weight_bit_width=n_bits,
            weight_quant=weight_quant,
            bias=True,
            return_quant_tensor=True,
        )

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.tensor): The input of the model.

        Returns:
            torch.tensor: Output of the network.
        """
        x = self.quant_input(x)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x.value


class TorchCustomModel(nn.Module):
    """A small network with Brevitas, trained on make_classification."""

    def __init__(self, input_shape, hidden_shape, output_shape):
        """Torch Model.

        Args:
            input_shape (int): Input size
            output_shape (int): Output size
            hidden_shape (int): Hidden size
        """
        super().__init__()
        self.linear1 = nn.Linear(input_shape, hidden_shape)
        self.linear2 = nn.Linear(hidden_shape, hidden_shape)
        self.linear3 = nn.Linear(hidden_shape, output_shape)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.tensor): The input of the model.

        Returns:
            torch.tensor: Output of the network.
        """
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class ConcatFancyIndexing(nn.Module):
    """Concat with fancy indexing."""

    def __init__(
        self, input_shape, hidden_shape, output_shape, n_bits: int = 4, n_blocks: int = 3
    ) -> None:
        """Torch Model.

        Args:
            input_shape (int):  Input size
            output_shape (int): Output size
            hidden_shape (int): Hidden size
            n_bits (int):       Number of bits
            n_blocks (int):     Number of blocks
        """
        super().__init__()

        self.n_blocks = n_blocks
        self.quant_1 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.fc1 = qnn.QuantLinear(input_shape, hidden_shape, bias=False, weight_bit_width=n_bits)

        self.quant_concat = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)

        self.quant_2 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.fc2 = qnn.QuantLinear(
            hidden_shape * self.n_blocks, hidden_shape, bias=True, weight_bit_width=n_bits
        )

        self.quant_3 = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
        self.fc4 = qnn.QuantLinear(hidden_shape, output_shape, bias=True, weight_bit_width=n_bits)

    def forward(self, x):
        """Forward pass.

        Args:
            x (torch.tensor): The input of the model.

        Returns:
            torch.tensor: Output of the network.
        """
        x_pre = []

        for i in range(self.n_blocks):
            x_block = x[:, i, :]
            q1_out = self.quant_1(x_block)
            fc1_out = self.fc1(q1_out)
            q_concat_out = self.quant_concat(fc1_out)

            x_pre.append(q_concat_out)

        x_pre_concat = torch.cat(x_pre, dim=1)
        x = self.quant_2(x_pre_concat)
        x = torch.relu(self.fc2(x))
        x = self.fc4(self.quant_3(x))
        return x
