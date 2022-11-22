"""Torch modules for our pytests."""
import numpy
import torch
from torch import nn


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


class CNNOther(nn.Module):
    """Torch CNN model for the tests."""

    def __init__(self, input_output, activation_function):
        super().__init__()

        self.activation_function = activation_function()
        self.conv1 = nn.Conv2d(input_output, 3, 3, 1, 1)
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

    def __init__(self, activation_function, padding, groups, gather_slice):
        super().__init__()

        self.activation_function = activation_function()
        self.flatten_function = lambda x: torch.flatten(x, 1)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.AvgPool2d(2, 2, padding=1) if padding else nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, groups=2) if groups else nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * (5 + padding * 1) * (5 + padding * 1), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.gather_slice = gather_slice

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
            return x[0, 0:-1:2]
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
            y: the secon input of the NN

        Returns:
            the output of the NN
        """
        return self.act(x + y)


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
        results = torch.stack(results)
        return results


class MultiOpOnSingleInputConvNN(nn.Module):
    """Network that applies two quantized operations on a single input."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(1, 8, 3)

    def forward(self, x):
        """Forward pass.

        Args:
            x: the input of the NN

        Returns:
            the output of the NN
        """
        layer1_out = torch.relu(self.conv1(x))
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
        for idx, l in enumerate(self.feat):
            x = self.act(l(x) + self.biases[idx])
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
        for idx, l in enumerate(self.feat):
            x = self.act(l(x) + self.biases[idx])
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
