import os
from pathlib import Path
from typing import cast

import numpy as np
import py_progress_tracker as progress
import torch
import torch.utils
from common import BENCHMARK_CONFIGURATION, run_and_report_classification_metrics
from concrete.numpy.compilation.circuit import Circuit
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from concrete.ml.torch.compile import compile_torch_model

N_EPOCHS = 30
TINYCNN_CHECKPOINT_FILE = "tiny_mnist.pth"

N_MAX_COMPILE_FHE = int(os.environ.get("N_MAX_COMPILE_FHE", 1000))
N_MAX_RUN_FHE = int(os.environ.get("N_MAX_RUN_FHE", 100))


class TinyCNN(nn.Module):
    """A very small CNN to classify the sklearn digits dataset.

    This class also allows pruning to a maximum of 10 active neurons, which
    should help with keeping the accumulator bitwidth low.
    """

    def __init__(self, n_classes) -> None:
        """Construct the CNN with a configurable number of classes."""
        super().__init__()

        # This network has a total complexity of 1216 MAC
        self.conv1 = nn.Conv2d(1, 2, 3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(2, 3, 3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(3, 16, 2, stride=1, padding=0)
        self.fc1 = nn.Linear(16, n_classes)

        # Enable pruning, prepared for training
        self.toggle_pruning(True)

    def toggle_pruning(self, enable):
        """Enables or removes pruning."""

        # Maximum number of active neurons (i.e. corresponding weight != 0)
        n_active = 10

        # Go through all the convolution layers
        for layer in (self.conv1, self.conv2, self.conv3):
            s = layer.weight.shape

            # Compute fan-in (number of inputs to a neuron)
            # and fan-out (number of neurons in the layer)
            st = [s[0], np.prod(s[1:])]

            # The number of input neurons (fan-in) is the product of
            # the kernel width x height x inChannels.
            if st[1] > n_active:
                if enable:
                    # This will create a forward hook to create a mask tensor that is multiplied
                    # with the weights during forward. The mask will contain 0s or 1s
                    prune.l1_unstructured(layer, "weight", (st[1] - n_active) * st[0])
                else:
                    # When disabling pruning the mask is multiplied with the weights
                    # and the result stored in the weights member
                    prune.remove(layer, "weight")

    def forward(self, x):
        """Run inference on the tiny CNN, apply the decision layer on the reshaped conv output."""

        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = x.view(-1, 16)
        x = self.fc1(x)
        return x


def make_dataset():
    """Download the digits dataset."""
    X, y = load_digits(return_X_y=True)
    return X, y


def train_one_epoch(epoch, net, optimizer, train_loader):
    """Run a training epoch."""

    # Cross Entropy loss for classification when not using a softmax layer in the network
    loss = nn.CrossEntropyLoss()

    # Set the network to training mode (enable gradient computation on normalization layers)
    net.train()

    avg_loss = 0
    for data, target in train_loader:
        # Clear the gradients
        optimizer.zero_grad()
        # Run forward
        output = net(data)
        # Compute the loss
        loss_net = loss(output, target.long())
        # Compute gradients with backprop
        loss_net.backward()
        # Update the weights with on the gradients
        optimizer.step()

        avg_loss += loss_net.item()

    print(f"Train epoch {epoch} loss: {avg_loss / len(train_loader)}")


def test_torch(net, test_loader, metric_id_prefix, metric_label_prefix, epoch=None):
    """Test the network: measure accuracy on the test set."""

    # Freeze normalization layers
    net.eval()

    all_y_pred = np.zeros((len(test_loader)), dtype=np.int32)
    all_targets = np.zeros((len(test_loader)), dtype=np.int32)

    # Iterate over the batches
    idx = 0
    for data, target in test_loader:
        # Accumulate the ground truth labels
        endidx = idx + target.shape[0]
        all_targets[idx:endidx] = target.numpy()

        # Run forward and get the predicted class id
        output = net(data).argmax(1).detach().numpy()
        all_y_pred[idx:endidx] = output

        idx += target.shape[0]

    # Print out the accuracy as a percentage
    if metric_id_prefix is not None and metric_label_prefix is not None:
        run_and_report_classification_metrics(
            all_targets, all_y_pred, metric_id_prefix, metric_label_prefix
        )
    else:
        # Print out the accuracy as a percentage, used during training
        n_correct = np.sum(all_targets == all_y_pred)
        print(f"Training epoch {epoch} accuracy: {n_correct / len(test_loader) * 100}")


def test_concrete(
    quantized_module, test_loader, metric_id_prefix, metric_label_prefix, use_fhe, use_vl
):
    """Test a compiled network."""

    all_y_pred = np.zeros((len(test_loader)), dtype=np.int32)
    all_targets = np.zeros((len(test_loader)), dtype=np.int32)

    # Iterate over the test batches and accumulate predictions and ground truth labels in a vector
    idx = 0
    for data, target in tqdm(test_loader):
        data = data.numpy()
        # Quantize the inputs
        x_test_q = quantized_module.quantize_input(data)

        # Accumulate the ground truth labels
        endidx = idx + target.shape[0]
        all_targets[idx:endidx] = target.numpy()

        # Iterate over single inputs
        for i in range(x_test_q.shape[0]):
            # Inputs must have size (N, C, H, W), we add the batch dimension with N=1
            x_q = np.expand_dims(x_test_q[i, :], 0)

            # Execute either in FHE (compiled or VL) or just in quantized
            if use_fhe or use_vl:
                out_fhe = quantized_module.forward_fhe.run(x_q)
                output = quantized_module.dequantize_output(out_fhe)
            else:
                output = quantized_module.forward_and_dequant(x_q)

            # Take the predicted class from the outputs and store it
            y_pred = np.argmax(output, 1)
            all_y_pred[idx] = y_pred
            idx += 1

    # Compute and report results
    run_and_report_classification_metrics(
        all_targets, all_y_pred, metric_id_prefix, metric_label_prefix
    )


@progress.track(
    [
        {
            "id": f"cnn_mnist_{n_bits}",
            "name": f"CNN on MNIST {n_bits}b",
            "parameters": {"n_bits": n_bits},
            "samples": 1,
        }
        for n_bits in range(2, 10)
    ]
)
def main(n_bits):
    """Runs a CNN benchmark for the tiny CNN on the small MNIST dataset.

    Train the network for 30 epochs to each about 80-85% accuracy. Then compile the network to FHE
    and check that the results match with the quantized-clear version. Compute accuracies for
    various quantization precisions.
    """

    X, y = make_dataset()

    # Standardize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # The sklearn Digits dataset, though it contains digit images, keeps these images in vectors
    # so we need to reshape them to 2d first. The images are 8x8 px in size and monochrome
    X = np.expand_dims(X.reshape((-1, 8, 8)), 1)

    # Split the data to train/test
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, shuffle=True, random_state=42
    )

    # Create the tiny CNN with 10 output classes
    net = TinyCNN(10)

    # Create a test data loader to supply batches for network evaluation (test)
    test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
    test_dataloader = DataLoader(test_dataset)

    # If a checkpoint is available, we should use it
    if Path(TINYCNN_CHECKPOINT_FILE).is_file():
        # Disable pruning since we won't train the network
        net.toggle_pruning(False)

        # Load the state dictionary
        state_dict = torch.load(TINYCNN_CHECKPOINT_FILE)
        net.load_state_dict(state_dict)
    else:
        # Create a train data loader
        train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
        train_dataloader = DataLoader(train_dataset, batch_size=64)

        # Train the network with Adam, output the test set accuracy every epoch
        optimizer = torch.optim.Adam(net.parameters())
        for epoch in range(N_EPOCHS):
            train_one_epoch(epoch, net, optimizer, train_dataloader)
            test_torch(net, test_dataloader, None, None, epoch=epoch)

        # Finally save a checkpoint on the trained network to speed up further tests
        net.toggle_pruning(False)
        torch.save(net.state_dict(), TINYCNN_CHECKPOINT_FILE)

    # Run a test in fp32 to establish accuracy
    test_torch(net, test_dataloader, "torch-fp32", "Torch fp32")

    # Compile and test the network with the Virtual Lib on the whole test set
    q_module_vl = compile_torch_model(
        net,
        x_train,
        n_bits=n_bits,
        use_virtual_lib=True,
        configuration=BENCHMARK_CONFIGURATION,
    )

    assert isinstance(q_module_vl.forward_fhe, Circuit)
    vfhe_circuit = cast(Circuit, q_module_vl.forward_fhe)
    # Despite casting and the assert, pylint still does not consider this a Circuit
    # pylint: disable=no-member
    print(f"Max n_bits during inference: {vfhe_circuit.get_max_bit_width()}")
    # pylint: enable=no-member

    test_concrete(
        q_module_vl,
        test_dataloader,
        "quantized-clear",
        "Quantized Clear",
        use_fhe=False,
        use_vl=True,
    )

    # Only do FHE tests for 2 bits
    if os.environ.get("BENCHMARK_NO_FHE", "0") == "0" and n_bits == 2:
        # Select a smaller set for FHE tests
        small_test_dataset = TensorDataset(
            torch.Tensor(x_test[0:N_MAX_RUN_FHE, ::]), torch.Tensor(y_test[0:N_MAX_RUN_FHE])
        )
        small_test_dataloader = DataLoader(small_test_dataset)

        # Now compile and run the FHE evaluation on a small set in virtual lib mode
        q_module_2b_vl = compile_torch_model(
            net,
            x_train[0:N_MAX_COMPILE_FHE, ::],
            n_bits=2,
            use_virtual_lib=True,
            configuration=BENCHMARK_CONFIGURATION,
        )

        test_concrete(
            q_module_2b_vl,
            small_test_dataloader,
            "quant-clear-fhe-set",
            "Quantized Clear on FHE set",
            use_fhe=False,
            use_vl=True,
        )

        # Now compile and run the FHE evaluation on a small set
        q_module_2b_fhe = compile_torch_model(
            net,
            x_train[0:N_MAX_COMPILE_FHE, ::],
            n_bits=2,
            use_virtual_lib=False,
            configuration=BENCHMARK_CONFIGURATION,
        )

        test_concrete(
            q_module_2b_fhe, small_test_dataloader, "fhe", "FHE", use_fhe=True, use_vl=False
        )
