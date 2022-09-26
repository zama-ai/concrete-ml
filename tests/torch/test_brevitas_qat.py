"""Tests with brevitas quantization aware training."""

import brevitas.nn as qnn
import numpy
import pytest
import torch
import torch.utils
from concrete.numpy.compilation.configuration import Configuration
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader, TensorDataset

from concrete.ml.torch.compile import compile_brevitas_qat_model


class TinyCNN(nn.Module):
    """A very small QAT CNN to classify the sklearn digits dataset.

    This class also allows pruning to a maximum of 10 active neurons, which
    should help keep the accumulator bit width low.
    """

    def __init__(self, n_classes, n_bits, n_active) -> None:
        """Construct the CNN with a configurable number of classes."""
        super().__init__()

        a_bits = n_bits
        w_bits = n_bits

        self.n_active = n_active

        # This network has a total complexity of 1216 MAC
        self.quant1 = qnn.QuantIdentity(bit_width=a_bits, return_quant_tensor=True)
        self.conv1 = qnn.QuantConv2d(1, 2, 3, stride=1, padding=0, weight_bit_width=w_bits)
        self.quant2 = qnn.QuantIdentity(bit_width=a_bits, return_quant_tensor=True)
        self.conv2 = qnn.QuantConv2d(2, 3, 3, stride=2, padding=0, weight_bit_width=w_bits)
        self.quant3 = qnn.QuantIdentity(bit_width=a_bits, return_quant_tensor=True)
        self.conv3 = qnn.QuantConv2d(3, 16, 2, stride=1, padding=0, weight_bit_width=w_bits)
        self.quant4 = qnn.QuantIdentity(bit_width=a_bits, return_quant_tensor=True)
        self.fc1 = qnn.QuantLinear(16, n_classes, weight_bit_width=3, bias=True)

        # Enable pruning, prepared for training
        self.toggle_pruning(True)

    def toggle_pruning(self, enable):
        """Enables or removes pruning."""

        # Maximum number of active neurons (i.e. corresponding weight != 0)

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
        """Run inference on the tiny CNN, apply the decision layer on the reshaped conv output."""

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
        """Test the network: measure accuracy on the test set."""

        # Freeze normalization layers
        self.eval()

        all_y_pred = numpy.zeros((len(test_loader)), dtype=numpy.int32)
        all_targets = numpy.zeros((len(test_loader)), dtype=numpy.int32)

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
                output = -1
            else:
                output = raw_pred.argmax(1)
            all_y_pred[idx:endidx] = output

            idx += target.shape[0]

        # Print out the accuracy as a percentage
        n_correct = numpy.sum(all_targets == all_y_pred)
        return n_correct / len(test_loader)


@pytest.mark.parametrize("qat_bits", [3, 7])
def test_brevitas_tinymnist_cnn(qat_bits):  # pylint: disable=too-many-statements
    """Train, execute and test a QAT CNN on a small version of MNIST."""

    # And some helpers for visualization.
    x_all, y_all = load_digits(return_X_y=True)

    # The sklearn Digits dataset, though it contains digit images, keeps these images in vectors
    # so we need to reshape them to 2D first. The images are 8x8 px in size and monochrome
    x_all = numpy.expand_dims(x_all.reshape((-1, 8, 8)), 1)

    x_train, x_test, y_train, y_test = train_test_split(
        x_all, y_all, test_size=0.25, shuffle=True, random_state=numpy.random.randint(0, 2**15)
    )

    def train_one_epoch(net, optimizer, train_loader):
        # Cross Entropy loss for classification when not using a softmax layer in the network
        loss = nn.CrossEntropyLoss()

        net.train()
        avg_loss = 0
        for data, target in train_loader:
            optimizer.zero_grad()
            output = net(data)
            loss_net = loss(output, target.long())
            loss_net.backward()
            optimizer.step()
            avg_loss += loss_net.item()

        return avg_loss / len(train_loader)

    # Prepare the data:
    # Create a train data loader
    train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    train_dataloader = DataLoader(train_dataset, batch_size=64)

    # Create a test data loader to supply batches for network evaluation (test)
    test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
    test_dataloader = DataLoader(test_dataset)

    trained_ok = False

    while not trained_ok:
        # Create the tiny CNN module with 10 output classes
        net = TinyCNN(10, qat_bits, 5 if qat_bits <= 3 else 20)

        # Train a single epoch to have a fast test, accuracy should still be the same VL vs torch
        # But train 3 epochs for the VL test to check that training works well
        n_epochs = 1 if qat_bits <= 3 else 3

        # Train the network with Adam, output the test set accuracy every epoch
        optimizer = torch.optim.Adam(net.parameters())
        for _ in range(n_epochs):
            train_one_epoch(net, optimizer, train_dataloader)

        # Finally, disable pruning (sets the pruned weights to 0)
        net.toggle_pruning(False)

        torch_acc = net.test_torch(test_dataloader)

        # If torch_acc was zero, training was wrong and there were NaNs in the weights
        # Retrain while training is bad
        trained_ok = torch_acc > 0

    cfg = Configuration(
        dump_artifacts_on_unexpected_failures=False,
        enable_unsafe_features=True,  # This is for our tests only, never use that in prod
    )

    def test_with_concrete(quantized_module, test_loader, use_fhe, use_vl):
        """Test a neural network that is quantized and compiled with Concrete-ML."""

        # When running in FHE, we cast inputs to uint8, but when running using the Virtual Lib (VL)
        # we may want inputs to exceed 8b to test quantization performance. Thus,
        # for VL we cast to int32
        dtype_inputs = numpy.uint8 if use_fhe else numpy.int32
        all_y_pred = numpy.zeros((len(test_loader)), dtype=numpy.int32)
        all_targets = numpy.zeros((len(test_loader)), dtype=numpy.int32)

        # Iterate over the test batches and accumulate predictions and ground truth
        # labels in a vector
        idx = 0
        for data, target in test_loader:
            data = data.numpy()
            # Quantize the inputs and cast to appropriate data type
            x_test_q = quantized_module.quantize_input(data).astype(dtype_inputs)

            # Accumulate the ground truth labels
            endidx = idx + target.shape[0]
            all_targets[idx:endidx] = target.numpy()

            # Iterate over single inputs
            for i in range(x_test_q.shape[0]):
                # Inputs must have size (N, C, H, W), we add the batch dimension with N=1
                x_q = numpy.expand_dims(x_test_q[i, :], 0)

                # Execute either in FHE (compiled or VL) or just in quantized
                if use_fhe or use_vl:
                    out_fhe = quantized_module.forward_fhe.encrypt_run_decrypt(x_q)
                    output = quantized_module.dequantize_output(out_fhe)
                else:
                    output = quantized_module.forward_and_dequant(x_q)

                # Take the predicted class from the outputs and store it
                y_pred = numpy.argmax(output, 1)
                all_y_pred[idx] = y_pred
                idx += 1

        # Compute and report results
        n_correct = numpy.sum(all_targets == all_y_pred)
        return n_correct / len(test_loader)

    net.eval()

    q_module_vl = compile_brevitas_qat_model(
        net,
        x_train,
        n_bits={
            "model_inputs": 7,
            "op_inputs": qat_bits,
            "op_weights": qat_bits,
            "model_outputs": 7,
        },
        configuration=cfg,
        use_virtual_lib=True,
    )

    accum_bits = q_module_vl.forward_fhe.graph.maximum_integer_bit_width()

    vl_acc = test_with_concrete(
        q_module_vl,
        test_dataloader,
        use_fhe=False,
        use_vl=True,
    )

    if qat_bits <= 3:
        # Make sure this network can compile to FHE if qat_bits is less than 3
        assert accum_bits <= 8
    else:
        # Accept, at most, 5 pct points accuracy difference. This can be due to the 7b input
        # output quantization. VL accuracy can be larger than fp32 accuracy though.
        assert vl_acc - torch_acc >= -0.05
        assert vl_acc > 0.2
