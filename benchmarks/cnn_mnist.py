import argparse
import os
import random
import time
from pathlib import Path
from typing import cast

import numpy as np
import py_progress_tracker as progress
import torch
import torch.utils
from common import BENCHMARK_CONFIGURATION, run_and_report_classification_metrics, seed_everything
from concrete.numpy.compilation.circuit import Circuit
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from concrete.ml.torch.compile import compile_torch_model

# To define the length of train sample for compilation
N_MAX_COMPILE_FHE = int(os.environ.get("N_MAX_COMPILE_FHE", 1000))


class TinyCNN(nn.Module):
    """A very small CNN to classify the sklearn digits dataset. Typically 30 epochs reach
    about 80-85% accuracy

    This class also allows pruning to a maximum of 10 active neurons, which
    should help with keeping the accumulator bitwidth low.
    """

    def __init__(self, n_classes) -> None:
        """Construct the NN with a configurable number of classes."""
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
        """Run inference on the NN, apply the decision layer on the reshaped conv output."""

        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = x.view(-1, 16)
        x = self.fc1(x)
        return x


class DeepAndNarrowNN(nn.Module):
    """A deep and narrow NN to classify the sklearn digits dataset. Typically 30 epochs reach
    about BCM% accuracy

    This class also allows pruning to a maximum of 10 active neurons, which
    should help with keeping the accumulator bitwidth low.
    """

    def __init__(self, n_classes) -> None:
        """Construct the NN with a configurable number of classes."""
        super().__init__()

        # This network has a total complexity of BCM MAC
        self.conv1 = nn.Conv2d(1, 2, 3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(2, 3, 3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(3, 16, 1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv7 = nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv8 = nn.Conv2d(16, 16, 2, stride=1, padding=0)
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
        """Run inference on the NN, apply the decision layer on the reshaped conv output."""

        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.conv4(x)
        x = torch.relu(x)
        x = self.conv5(x)
        x = torch.relu(x)
        x = self.conv6(x)
        x = torch.relu(x)
        x = self.conv7(x)
        x = torch.relu(x)
        x = self.conv8(x)
        x = torch.relu(x)
        x = x.view(-1, 16)
        x = self.fc1(x)
        return x


class ShallowAndWideNN(nn.Module):
    """A shallow and wide NN to classify the sklearn digits dataset. Typically 30 epochs reach
    about 90-95% accuracy

    This class also allows pruning to a maximum of 10 active neurons, which
    should help with keeping the accumulator bitwidth low.
    """

    def __init__(self, n_classes) -> None:
        """Construct the NN with a configurable number of classes."""
        super().__init__()

        # This network has a total complexity of BCM MAC
        self.conv1 = nn.Conv2d(1, 2, 3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(2, 50, 4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(50, 16, 2, stride=1, padding=0)
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
        """Run inference on the NN, apply the decision layer on the reshaped conv output."""

        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = x.view(-1, 16)
        x = self.fc1(x)
        return x


class DeepAndWideNN(nn.Module):
    """A deep and wide NN to classify the sklearn digits dataset. Typically 30 epochs reach
    about BCM% accuracy

    This class also allows pruning to a maximum of 10 active neurons, which
    should help with keeping the accumulator bitwidth low.
    """

    def __init__(self, n_classes) -> None:
        """Construct the NN with a configurable number of classes."""
        super().__init__()

        # This network has a total complexity of BCM MAC
        self.conv1 = nn.Conv2d(1, 2, 3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(2, 3, 3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(3, 16, 1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv5 = nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv6 = nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv7 = nn.Conv2d(16, 16, 1, stride=1, padding=0)
        self.conv8 = nn.Conv2d(16, 16, 2, stride=1, padding=0)
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
        """Run inference on the NN, apply the decision layer on the reshaped conv output."""

        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = self.conv4(x)
        x = torch.relu(x)
        x = self.conv5(x)
        x = torch.relu(x)
        x = self.conv6(x)
        x = torch.relu(x)
        x = self.conv7(x)
        x = torch.relu(x)
        x = self.conv8(x)
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

    print(f"Train epoch {epoch} loss: {avg_loss / len(train_loader):.2f}")


def test_torch(net, test_loader, metric_id_prefix, metric_label_prefix, epoch=None):
    """Test the network: measure accuracy on the test set."""

    # Freeze normalization layers
    net.eval()

    all_y_pred = np.zeros((len(test_loader)), dtype=np.int64)
    all_targets = np.zeros((len(test_loader)), dtype=np.int64)

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
            all_targets, all_y_pred, metric_id_prefix, metric_label_prefix, use_f1=False
        )
    else:
        # Print out the accuracy as a percentage, used during training
        n_correct = np.sum(all_targets == all_y_pred)
        print(f"Training epoch {epoch} accuracy: {n_correct / len(test_loader) * 100:.2f}")


def test_concrete(
    quantized_module, test_loader, metric_id_prefix, metric_label_prefix, args, use_fhe, use_vl
):
    """Test a compiled network."""
    assert use_fhe ^ use_vl

    if args.mlir_only:
        # If called with a virtual circuit, we don't do anything
        if use_fhe:
            print("MLIR:", quantized_module.forward_fhe.mlir)
        return

    all_y_pred = np.zeros((len(test_loader)), dtype=np.int64)
    all_targets = np.zeros((len(test_loader)), dtype=np.int64)

    if use_fhe:
        t_start = time.time()
        quantized_module.forward_fhe.keygen()
        duration = time.time() - t_start
        progress.measure(id="fhe-keygen-time", label="FHE Key Generation Time", value=duration)

    t_start = time.time()

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
            if use_fhe:
                output = quantized_module.forward_fhe.encrypt_run_decrypt(x_q)
            elif use_vl:
                output = quantized_module.forward_fhe.simulate(x_q)
            else:
                # Here, that's with quantized module, but we could remove it, it will never be used
                output = quantized_module.forward(x_q)

            # Dequantize the integer predictions
            output = quantized_module.dequantize_output(output)

            # Take the predicted class from the outputs and store it
            y_pred = np.argmax(output, 1)
            all_y_pred[idx] = y_pred
            idx += 1

    if use_fhe:
        assert idx == args.fhe_samples, f"{idx=} {args.fhe_samples=}"

    # Compute and report results
    run_and_report_classification_metrics(
        all_targets, all_y_pred, metric_id_prefix, metric_label_prefix, use_f1=False
    )

    duration = time.time() - t_start
    duration_per_sample = duration / idx

    progress.measure(
        id=metric_id_prefix + "-execution-time-per-sample",
        label="Execution Time per sample for " + metric_id_prefix,
        value=duration_per_sample,
    )


def train_mnist(args, net_name):
    """Train the network for some epoch"""
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

    # Create the model with 10 output classes
    if net_name == "tiny_cnn":
        net = TinyCNN(10)
    elif net_name == "deep_and_narrow_nn":
        net = DeepAndNarrowNN(10)
    elif net_name == "shallow_and_wide_nn":
        net = ShallowAndWideNN(10)
    elif net_name == "deep_and_wide_nn":
        net = DeepAndWideNN(10)
    else:
        raise ValueError(f"bad name {net_name}")

    # Create a test data loader to supply batches for network evaluation (test)
    test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
    test_dataloader = DataLoader(test_dataset)

    # If a checkpoint is available, we should use it
    pth_file = "mnist_" + net_name

    print()

    if Path(pth_file).is_file() and args.use_checkpoint:

        print(f"Skip training for {net_name}: use {pth_file}")

        # Disable pruning since we won't train the network
        net.toggle_pruning(False)

        # Load the state dictionary
        state_dict = torch.load(pth_file)
        net.load_state_dict(state_dict)
    else:

        print(f"Training for {net_name} for {args.epochs} epochs")

        # Create a train data loader
        train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
        train_dataloader = DataLoader(train_dataset, batch_size=64)

        # Train the network with Adam, output the test set accuracy every epoch
        optimizer = torch.optim.Adam(net.parameters())
        for epoch in range(args.epochs):
            train_one_epoch(epoch, net, optimizer, train_dataloader)
            test_torch(net, test_dataloader, None, None, epoch=epoch)

        # Finally save a checkpoint on the trained network to speed up further tests
        net.toggle_pruning(False)
        torch.save(net.state_dict(), pth_file)

    print()

    return net, x_train, x_test, y_test, test_dataloader


def test_net_mnist(args, net, x_train, x_test, y_test, test_dataloader, n_bits):
    """Runs a NN benchmark on the small MNIST dataset.

    Run on VL the full test-set, then, if execute_in_fhe is set, run on VL and later in FHE the
    partial test-set. Compute accuracies.
    """

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
    print(f"Selected n_bits = {n_bits}")
    print(f"Max numbers of bits during inference: {vfhe_circuit.graph.maximum_integer_bit_width()}")
    # pylint: enable=no-member

    print("Testing in VL, full dataset")
    test_concrete(
        q_module_vl,
        test_dataloader,
        "quantized-clear",
        "Quantized Clear",
        args,
        use_fhe=False,
        use_vl=True,
    )

    # Do FHE tests
    if args.execute_in_fhe:
        # Select a smaller set for FHE tests
        small_test_dataset = TensorDataset(
            torch.Tensor(x_test[0 : args.fhe_samples, ::]),
            torch.Tensor(y_test[0 : args.fhe_samples]),
        )
        small_test_dataloader = DataLoader(small_test_dataset)

        # Now compile and run the FHE evaluation on a small set in virtual lib mode
        q_module_vl = compile_torch_model(
            net,
            x_train[0:N_MAX_COMPILE_FHE, ::],
            n_bits=n_bits,
            use_virtual_lib=True,
            configuration=BENCHMARK_CONFIGURATION,
        )

        print("Testing in VL, partial dataset")
        test_concrete(
            q_module_vl,
            small_test_dataloader,
            "quant-clear-fhe-set",
            "Quantized Clear on FHE set",
            args,
            use_fhe=False,
            use_vl=True,
        )

        # Now compile and run the FHE evaluation on a small set
        q_module_fhe = compile_torch_model(
            net,
            x_train[0:N_MAX_COMPILE_FHE, ::],
            n_bits=n_bits,
            use_virtual_lib=False,
            configuration=BENCHMARK_CONFIGURATION,
        )

        print("Testing in FHE, partial dataset")
        test_concrete(
            q_module_fhe, small_test_dataloader, "fhe", "FHE", args, use_fhe=True, use_vl=False
        )


def argument_manager():
    # Manage arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlir_only", type=int, help="Only dump MLIR (no inference)")
    parser.add_argument("--verbose", action="store_true", help="show more information on stdio")
    parser.add_argument(
        "--seed",
        type=int,
        default=random.randint(0, 2**32 - 1),
        help="set the seed for reproducibility",
    )
    parser.add_argument(
        "--n_bits",
        type=int,
        nargs="+",
        default=range(2, 9),
        help="n_bits values",
    )
    parser.add_argument(
        "--list_of_networks",
        type=str,
        nargs="+",
        default=["tiny_cnn", "deep_and_narrow_nn", "shallow_and_wide_nn", "deep_and_wide_nn"],
        choices=["tiny_cnn", "deep_and_narrow_nn", "shallow_and_wide_nn", "deep_and_wide_nn"],
        help="which NN to benchmark",
    )
    parser.add_argument(
        "--model_samples",
        type=int,
        default=1,
        help="number of model samples (ie, overwrite PROGRESS_SAMPLES)",
    )
    parser.add_argument(
        "--fhe_samples", type=int, default=1, help="number of FHE samples on which to predict"
    )
    parser.add_argument(
        "--execute_in_fhe",
        action="store_true",
        help="force to execute in FHE (default is to use should_test_config_in_fhe function)",
    )
    parser.add_argument(
        "--use_checkpoint",
        action="store_true",
        help="don't do training, reuse .pth file",
    )
    parser.add_argument("--epochs", type=int, default=1, help="number of epochs for the training")

    args = parser.parse_args()

    return args


def main():

    # Parameters by the user
    args = argument_manager()

    # Seed everything we can
    seed_everything(args.seed)

    print()
    print(f"Using --seed {args.seed}")
    print(f"Networks {args.list_of_networks}")
    print("Epochs:", args.epochs)
    print("n_bits:", list(args.n_bits))
    print("Do FHE:", args.execute_in_fhe)

    if args.execute_in_fhe:
        print("Number of FHE samples:", args.fhe_samples)

    net = {}

    for net_name in args.list_of_networks:
        net[net_name], x_train, x_test, y_test, test_dataloader = train_mnist(args, net_name)

    # There is a mistake by pylint in next loop, which believes that net_name may not be defined
    net_name = ""

    @progress.track(
        [
            {
                "id": f"cnn_mnist_{n_bits}b",
                "name": f"{net_name} on MNIST with {n_bits}b",
                "samples": args.model_samples,
                "parameters": {"n_bits": n_bits, "net_name": net_name},
            }
            for n_bits in args.n_bits
            for net_name in args.list_of_networks
        ]
    )
    def perform_mnist_benchmark(n_bits, net_name):
        """
        This is the test function called by the py-progress module. It just calls the
        benchmark function with the right parameter combination
        """
        test_net_mnist(args, net[net_name], x_train, x_test, y_test, test_dataloader, n_bits)


if __name__ == "__main__":
    main()
