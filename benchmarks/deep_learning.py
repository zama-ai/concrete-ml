import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import py_progress_tracker as progress
import torch
import torch.utils
from common import (
    benchmark_generator,
    benchmark_name_generator,
    run_and_report_classification_metrics,
    seed_everything,
)
from concrete.fhe.compilation.circuit import Circuit
from sklearn.datasets import load_digits
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from concrete.ml.common.utils import get_model_name
from concrete.ml.quantization.quantized_module import QuantizedModule
from concrete.ml.torch.compile import compile_torch_model

# To define the length of train sample for compilation
N_MAX_COMPILE_FHE = 200


class _CustomCNN(nn.Module):
    """A CNN to classify images.

    This class also allows pruning, which should help with keeping the accumulator bit-width low.
    This is done by defining a maximum number of active neurons (i.e., weight != 0) allowed
    as inputs to other neurons.
    """

    def __init__(
        self,
        n_classes,
        hidden_size,
        activation_function,
        n_active_neurons,
        n_deep_conv=None,
    ):
        """Construct the CNN with a configurable number of classes."""
        super().__init__()

        self.conv_layers = nn.Sequential()
        self.conv_layers.add_module(
            "conv1", nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0)
        )
        self.conv_layers.add_module(
            "conv2",
            nn.Conv2d(
                in_channels=2,
                out_channels=hidden_size,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
        )

        if n_deep_conv is not None:
            for i in range(n_deep_conv):
                self.conv_layers.add_module(
                    f"deep_conv{i}",
                    nn.Conv2d(
                        in_channels=hidden_size,
                        out_channels=hidden_size,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                    ),
                )

        self.conv_layers.add_module(
            "conv3",
            nn.Conv2d(in_channels=hidden_size, out_channels=16, kernel_size=2, stride=1, padding=0),
        )
        self.fc1 = nn.Linear(16, n_classes)
        self.act = activation_function
        self.n_active_neurons = n_active_neurons

        # Enable pruning for training
        self.toggle_pruning(True)

    def toggle_pruning(self, enable):
        """Enable or remove pruning."""
        # Iterate over all the convolution layers
        for layer in self.conv_layers:
            layer_shape = layer.weight.shape

            # Compute the fan-in, the number of inputs to a neuron, and the fan-out, the number of
            # neurons in the current layer.
            # The fan-in is the product of the kernel width x height x in_channels while the fan-out
            # is out_channels
            fan_in = np.prod(layer_shape[1:])
            fan_out = layer_shape[0]

            # If there are more inputs than the maximum amount allowed, prune the layer
            if fan_in > self.n_active_neurons:
                # If pruning is enabled, which is generally the case during training, a
                # forward hook is added in order to create a mask tensor (made of 0 or 1) that will
                # be multiplied with the weights during the forward pass.
                if enable:
                    prune.l1_unstructured(
                        layer, "weight", (fan_in - self.n_active_neurons) * fan_out
                    )

                # Else, the mask is multiplied with the weights and the result is stored in the
                # weight member.
                # This is mostly done before saving or loading pruned networks into pre-trained
                # files as these features is not properly handled in Torch
                # https://discuss.pytorch.org/t/proper-way-to-load-a-pruned-network/77694
                else:
                    prune.remove(layer, "weight")

    def forward(self, x):
        """Run the inference.

        The decision layer (the linear layer) is applied on a reshaped output of the last
        convolutional layer.
        """
        # Run the convolutional layers
        for layer in self.conv_layers:
            x = layer(x)
            x = self.act(x)

        # Squeeze size (batch, n, 1, 1) to size (batch, n) and run the final linear layer
        x = x.flatten(start_dim=1)
        x = self.fc1(x)

        return x


class ShallowNarrowCNN(_CustomCNN):
    """A shallow and narrow CNN to classify images."""

    def __init__(self, n_classes):
        super().__init__(
            n_classes=n_classes,
            hidden_size=3,
            activation_function=torch.relu,
            n_active_neurons=10,
        )


class ShallowWideCNN(_CustomCNN):
    """A shallow and wide CNN to classify images.

    This CNN considers 92 filters in the middle hidden layer in order to maximize CPU usage
    with parallelized computations.
    """

    def __init__(self, n_classes):
        super().__init__(
            n_classes=n_classes,
            hidden_size=92,
            activation_function=torch.relu,
            n_active_neurons=10,
        )


class Deep20CNN(_CustomCNN):
    """A deep CNN using 20 hidden layers to classify images."""

    def __init__(self, n_classes):
        super().__init__(
            n_classes=n_classes,
            hidden_size=3,
            activation_function=torch.relu,
            n_active_neurons=10,
            n_deep_conv=20,
        )


class Deep50CNN(_CustomCNN):
    """A deep CNN using 50 hidden layers to classify images."""

    def __init__(self, n_classes):
        super().__init__(
            n_classes=n_classes,
            hidden_size=3,
            activation_function=torch.relu,
            n_active_neurons=10,
            n_deep_conv=50,
        )


class Deep100CNN(_CustomCNN):
    """A deep CNN using 100 hidden layers to classify images."""

    def __init__(self, n_classes):
        super().__init__(
            n_classes=n_classes,
            hidden_size=3,
            activation_function=torch.relu,
            n_active_neurons=10,
            n_deep_conv=100,
        )


CNN_CLASSES = [ShallowNarrowCNN, Deep20CNN, Deep50CNN, Deep100CNN, ShallowWideCNN]
CNN_STRING_TO_CLASS = {cnn_model.__name__: cnn_model for cnn_model in CNN_CLASSES}

CNN_DATASETS = ["MNIST"]


def load_data(dataset: str = "mnist") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the data.

    Args:
        dataset (str): the data-set to use.

    Returns:
        x_train, x_test, y_train, y_test (Tuple): The input and target values to
            use for training and evaluating.

    Raise:
        ValueError: If dataset is not currently handled.
    """

    if dataset == "MNIST":
        # Load the MNIST input and target values
        X, y = load_digits(return_X_y=True)

        # Standardize the data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # The scikit-learn MNIST data-set keeps the images flattened, we therefore need to reshape
        # them to 2D array of 8x8 (grayscale) in order to be able to apply convolutions in the
        # first layer.
        X = np.expand_dims(X.reshape((-1, 8, 8)), 1)

        # Split the data into train and test subsets
        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, shuffle=True, random_state=42
        )

    else:
        raise ValueError(f"Wrong data-set. Expected one of {CNN_DATASETS}, got {dataset}.")

    return x_train, x_test, y_train, y_test


def get_data_loader(
    x: np.ndarray, y: np.ndarray, batch_size: int = 1, samples: Optional[int] = None
):
    """Build a data loader.

    Args:
        x (np.ndarray): The input values.
        y (np.ndarray): The target values.
        batch_size (int): The batch size to consider. Default to 1.
        samples (Optional[int]). The number of samples to consider. Default to None, indicating that
            the whole data-set should be taken.

    Returns:
        data_loader (DataLoader): The batched data loader.
    """
    # Create a TensorDataset object from the data
    if samples is None:
        dataset = TensorDataset(torch.Tensor(x), torch.Tensor(y))

    else:
        dataset = TensorDataset(torch.Tensor(x[0:samples, ::]), torch.Tensor(y[0:samples]))

    # Create a DataLoader object with data batches
    data_loader = DataLoader(dataset, batch_size=batch_size)

    return data_loader


def get_pt_file(model_name: str, dataset: str) -> Path:
    """Retrieve the model's pre-trained file associated to the given data-set.

    Args:
        model_name (str): The model's name.
        dataset (str): the data-set's name.

    Returns:
        Path: The model's pre-trained file.
    """
    return Path(__file__).resolve().parent / f"pre_trained_models/{dataset}_{model_name}.pt"


def load_pre_trained_cnn_model(cnn_model: Any, dataset: str) -> Any:
    """Load the model's pre-trained weights.

    Args:
        cnn_model (Any): The instantiated model.
        dataset (str): the data-set's name.

    Returns:
        Any: The pre-trained model.
    """
    model_name = get_model_name(cnn_model)

    # Retrieve the pre-trained file path
    pt_file = get_pt_file(model_name, dataset)

    if not pt_file.is_file():
        raise ValueError(
            f"Pre-trained file for model {model_name} cannot be found. "
            f"Expected path {pt_file.resolve()}"
        )

    print(f"Loading {pt_file.name}\n")

    # Disable pruning as Torch doesn't allow to easily load and save pruned networks
    # https://discuss.pytorch.org/t/proper-way-to-load-a-pruned-network/77694
    cnn_model.toggle_pruning(False)

    # Load the weights and update the model
    state_dict = torch.load(pt_file)
    cnn_model.load_state_dict(state_dict)

    return cnn_model


def train_one_epoch(
    cnn_model: Any,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss: nn.Module,
):
    """Train the model for a single epoch.

    Args:
        cnn_model (Any): The instantiated model to train.
        train_loader (DataLoader): The batched train data to consider.
        optimizer (torch.optim.Optimizer): The optimizer to use for computing the gradients.
        loss (nn.Module): The loss function to use.
    """

    # Set the network to training mode (enable gradient computation on normalization layers)
    cnn_model.train()

    accumulated_loss = 0
    for data, target in train_loader:
        # Clear the gradients
        optimizer.zero_grad()

        # Run the forward pass
        output = cnn_model(data)

        # Compute the loss
        loss_value = loss(output, target.long())

        # Compute the gradients with backpropagation
        loss_value.backward()

        # Update the weights using the gradients
        optimizer.step()

        # Store the computed loss
        accumulated_loss += loss_value.item()

    print(f"Loss: {accumulated_loss / len(train_loader):.2f}")


def train_cnn_model(cnn_class: Any, dataset: str, epochs: int, batch_size: int):
    """Train the model on the data-set for several epochs and save the weights in a file.

    Args:
        cnn_model (Any): The instantiated model to train.
        dataset (str): the data-set to train on.
        epochs (int): The number of epochs to consider.
        batch_size (int): The training batches' size.

        Return:
            Any: The trained model.
    """

    # Retrieve the model's name
    model_name = get_model_name(cnn_class)

    print(f"Training {model_name} on {dataset} for {epochs} epochs\n")

    # Load the data and split it in train and test subsets
    x_train, x_test, y_train, y_test = load_data(dataset)

    # Extract the number of classes in the data set
    n_classes = len(np.unique(y_train))

    # Create the model with n_classes output classes
    cnn_model = cnn_class(n_classes)

    # Create the train and test data loaders
    train_loader = get_data_loader(x_train, y_train, batch_size=batch_size)
    test_loader = get_data_loader(x_test, y_test, batch_size=batch_size)

    # Use Adam for training
    optimizer = torch.optim.Adam(cnn_model.parameters())

    # Use the cross Entropy loss for classification when not using a softmax layer in the CNN
    loss = nn.CrossEntropyLoss()

    # Train the model for several epochs and evaluate its test accuracy at each step
    for epoch in range(epochs):
        print(f"Training epoch {epoch+1}:")

        train_one_epoch(cnn_model, train_loader, optimizer, loss)
        evaluate_module(
            framework="torch",
            module=cnn_model,
            test_loader=test_loader,
            n_classes=n_classes,
            metric_id_prefix=None,
            metric_label_prefix=None,
            train=True,
        )

    # Disable pruning as Torch doesn't allow to easily load and save pruned networks
    # https://discuss.pytorch.org/t/proper-way-to-load-a-pruned-network/77694
    cnn_model.toggle_pruning(False)

    # Retrieve the pre-trained file path
    pt_file = get_pt_file(model_name, dataset)

    # Save the pre-trained model's weights in a pt file
    torch.save(cnn_model.state_dict(), pt_file)

    return cnn_model


def report_compiler_feedback(fhe_circuit: Circuit):
    """Report all compiler feedback values.

    Args:
        fhe_circuit (Circuit): The FHE circuit to consider.
    """
    compiler_feedback_names = [
        "complexity",
        "size_of_secret_keys",
        "size_of_bootstrap_keys",
        "size_of_keyswitch_keys",
        "size_of_inputs",
        "size_of_outputs",
        "p_error",
        "global_p_error",
    ]

    for compiler_feedback_name in compiler_feedback_names:
        compiler_feedback_value = getattr(fhe_circuit, compiler_feedback_name)

        progress.measure(
            id=f"compiler-{compiler_feedback_name}",
            label=f"Compiler {compiler_feedback_name}",
            value=compiler_feedback_value,
        )


def concrete_inference(quantized_module: QuantizedModule, x: np.ndarray, in_fhe: bool):
    """Execute the model's inference using Concrete ML (quantized clear or FHE).

    Args:
        quantized_module (QuantizedModule): The quantized module representing the model.
        x (np.ndarray): The input data.
        in_fhe (bool): Indicate if the inference should be executed in FHE.

    Returns:
        y_preds (np.ndarray): The model's predictions.
        q_y_pred_proba (np.ndarray): The model's predicted quantized probabilities.
    """

    # Quantize the inputs
    q_x = quantized_module.quantize_input(x)

    fhe_mode = "execute" if in_fhe else "disable"

    # Execute the forward pass, in FHE or in the clear
    q_y_pred_proba = quantized_module.quantized_forward(q_x, fhe=fhe_mode)

    # De-quantize the output probabilities
    y_pred_proba = quantized_module.dequantize_output(q_y_pred_proba)

    # Apply the argmax in the clear
    y_pred = np.argmax(y_pred_proba, 1)

    return y_pred, q_y_pred_proba


def torch_inference(cnn_model: nn.Module, x: torch.Tensor):
    """Execute the model's inference using Torch (in float, in the clear).

    Args:
        cnn_model (nn.Module): The torch model.
        x (torch.Tensor): The input data.

    Returns:
        y_preds (np.ndarray): The model's predictions.
        y_pred_proba (np.ndarray): The model's predicted probabilities.
    """
    # Freeze normalization layers
    cnn_model.eval()

    # Execute the inference in the clear
    y_pred_proba = cnn_model(x)

    # Apply the argmax in the clear
    y_pred = y_pred_proba.argmax(1)

    return y_pred.detach().numpy(), y_pred_proba.detach().numpy()


def evaluate_module(
    framework: str,
    module: Union[nn.Module, QuantizedModule],
    test_loader: DataLoader,
    n_classes: int,
    metric_id_prefix: Optional[str] = None,
    metric_label_prefix: Optional[str] = None,
    train: bool = False,
    in_fhe: bool = False,
):
    """Evaluate several metrics using a Torch or Concrete ML module.

    Args:
        framework (str): The framework to evaluate, either 'concrete' or 'torch'.
        module (Union[nn.Module, QuantizedModule]): The Torch or Concrete ML module representing
            the model to evaluate.
        test_loader (DataLoader): The test data loader.
        n_classes (int): The number of classes to target.
        metric_id_prefix (Optional[str]): The id's prefix to consider when tracking the metrics.
            Default to None.
        metric_label_prefix (Optional[str]): The label's prefix to consider when tracking the
            metrics. Default to None.
        train (bool): Indicate if the evaluation is done during training. If so, the test accuracy
            is printed but not tracked. This parameter cannot be set while using the Concrete ML
            framework. Default to False.
        in_fhe (bool): Indicate if the inference should be executed in FHE. This parameter cannot
            be set while using the torch framework. Default to False.

    Returns:
        y_preds_proba (np.ndarray): The model's predicted probabilities (quantized form for the
            'concrete' framework).
    """

    assert framework in [
        "concrete",
        "torch",
    ], f"Wrong framework. Expected one of 'torch' or 'concrete', got {framework}."

    if framework == "torch":
        assert (
            not in_fhe
        ), "Torch models can't be executed in FHE. Either use 'concrete' or set 'in_fhe' to False."
    else:
        assert not train, "Training can only be done using Torch models."

    # If the module is a QuantizedModule whose inference will be executed in FHE, generated the
    # keys and track their generation time
    if isinstance(module, QuantizedModule) and in_fhe:
        keygen_start = time.time()
        module.fhe_circuit.keygen()
        keygen_time = time.time() - keygen_start
        progress.measure(id="fhe-keygen-time", label="FHE Key Generation Time", value=keygen_time)

    # Retrieve the batches' size as well as the total amount of values in the test data loader
    batch_size = test_loader.batch_size if test_loader.batch_size is not None else 1
    total_size = len(test_loader) * batch_size

    # Initialize the arrays for storing the predicted values, probabilities and ground truth
    # target labels
    y_preds = np.zeros((total_size), dtype=np.float64)
    y_preds_proba = np.zeros((total_size, n_classes), dtype=np.float64)
    targets = np.zeros((total_size), dtype=np.float64)

    inference_start = time.time()

    with tqdm(total=total_size) as progress_bar:

        # Iterate over the test batches and store the predicted values as well as the ground
        # truth labels
        for batch_i, (data, target) in enumerate(test_loader):

            # Execute Concrete ML's inference
            if framework == "concrete":
                y_pred, y_pred_proba = concrete_inference(module, data.numpy(), in_fhe)

            # Else, execute torch's inference
            else:
                y_pred, y_pred_proba = torch_inference(module, data)

            # Store the predicted values and the ground truth target labels at the right indexes
            start_index = batch_i * batch_size
            batch_slice = slice(start_index, start_index + min(batch_size, target.shape[0]))
            y_preds[batch_slice] = y_pred
            y_preds_proba[batch_slice] = y_pred_proba
            targets[batch_slice] = target.numpy()

            progress_bar.update(batch_size)

    inference_time = time.time() - inference_start

    # If the evaluation is done during training, print the test accuracy as a percentage
    if train:
        total_correct = np.sum(targets == y_preds)
        print(f"Test accuracy: {total_correct / total_size * 100:.2f}%\n")

    # Else, track the different metrics
    else:
        assert (
            metric_id_prefix is not None and metric_label_prefix is not None
        ), "Please prove metric prefixes when executing the inference."

        # If we evaluate a Concrete ML module in FHE, the inference execution time is also tracked
        if framework == "concrete" and in_fhe:
            progress.measure(
                id=metric_id_prefix + "-execution-time-per-sample",
                label="Execution Time per sample for " + metric_label_prefix,
                value=inference_time / total_size,
            )

        # Compute and report the metrics
        run_and_report_classification_metrics(
            targets, y_preds, metric_id_prefix, metric_label_prefix, use_f1=False
        )

    return y_preds_proba


def evaluate_pre_trained_cnn_model(dataset: str, cnn_class: type, config: dict, cli_args):
    """Evaluate the pre-trained CNN model on the data-set.

    It first evaluates both the Torch and Concrete ML models in the clear bu computing their
    accuracy score on the full data-set. Then, the Concrete ML model's inference is executed on a
    sub-sample in the clear as well as in FHE in order to compute a MSE score between them.

    Args:
        dataset (str): the data-set to consider.
        cnn_class (type): The model's class to train.
        config (dict): The configuration parameters to consider for the model, such as the number
            of bits of quantization to consider during compilation.
        cli_args (): The parsed arguments from the command line to consider.
    """

    # Load the data and split it in train and test subsets
    x_train, x_test, y_train, y_test = load_data(dataset)

    # Extract the number of classes in the data set
    n_classes = len(np.unique(y_train))

    # Create the model with n_classes output classes
    cnn_model = cnn_class(n_classes)

    # Load the pre-trained weights into the model
    cnn_model = load_pre_trained_cnn_model(cnn_model, dataset)

    p_error = config["p_error"]
    n_bits = config["n_bits"]

    if cli_args.verbose:
        print(
            f"Converting and compiling to a quantized module, using n_bits: {n_bits}, "
            f"p_error: {p_error}"
        )

    progress.measure(id="p-error", label="Using p-error", value=p_error)

    # Compile the model for FHE computations, using N_MAX_COMPILE_FHE samples at most in order
    # to avoid a long compilation time
    compilation_start = time.time()
    fhe_module = compile_torch_model(
        cnn_model,
        x_train[0:N_MAX_COMPILE_FHE, ::],
        n_bits=n_bits,
        p_error=p_error,
        verbose=cli_args.verbose,
    )
    compilation_time = time.time() - compilation_start

    progress.measure(id="fhe-compile-time", label="FHE Compile Time", value=compilation_time)

    assert fhe_module.fhe_circuit is not None, "Please compile the FHE module."

    if cli_args.mlir_only:
        print("MLIR:", fhe_module.fhe_circuit.mlir)
        return

    report_compiler_feedback(fhe_module.fhe_circuit)

    # Create a test data loader to supply batches for evaluating the model's different metrics
    test_loader = get_data_loader(x_test, y_test, batch_size=50)

    if cli_args.verbose:
        print("Evaluating the Torch model's inference in the clear:")

    # Evaluate the float model using Torch i order to better compare results
    evaluate_module(
        framework="torch",
        module=cnn_model,
        test_loader=test_loader,
        n_classes=n_classes,
        metric_id_prefix="torch-fp32",
        metric_label_prefix="Torch fp32",
    )

    circuit_bitwidth = fhe_module.fhe_circuit.graph.maximum_integer_bit_width()

    progress.measure(
        id="maximum-circuit-bit-width",
        label="Circuit bit width",
        value=circuit_bitwidth,
    )

    if cli_args.verbose:
        print("\nMax numbers of bits reached during the inference:", circuit_bitwidth)
        print("\nEvaluating the Concrete ML model's quantized clear inference an all test samples:")

    # Evaluate the quantized clear inference using the full data-set
    evaluate_module(
        framework="concrete",
        module=fhe_module,
        test_loader=test_loader,
        n_classes=n_classes,
        metric_id_prefix="quantized-clear",
        metric_label_prefix="Quantized Clear",
    )

    # Build a subset for FHE tests
    fhe_test_loader = get_data_loader(x_test, y_test, samples=cli_args.fhe_samples)

    if cli_args.verbose:
        print("\nEvaluating the Concrete ML model's quantized clear inference on FHE samples:")

    # Evaluate the quantized clear inference using a specific number of FHE samples
    q_y_preds_proba_clear = evaluate_module(
        framework="concrete",
        module=fhe_module,
        test_loader=fhe_test_loader,
        n_classes=n_classes,
        metric_id_prefix="quant-clear-fhe-set",
        metric_label_prefix="Quantized Clear (FHE set)",
    )

    if not cli_args.dont_execute_in_fhe:
        if cli_args.verbose:
            print("\nEvaluating the Concrete ML model's inference in FHE on FHE samples:")

        # Evaluate the FHE inference using a specific number of FHE samples
        q_y_preds_proba_fhe = evaluate_module(
            framework="concrete",
            module=fhe_module,
            test_loader=fhe_test_loader,
            n_classes=n_classes,
            metric_id_prefix="fhe",
            metric_label_prefix="FHE",
            in_fhe=True,
        )

        # Compute the MAE between predicted quantized probabilities from the quantized clear model
        # and the FHE one
        mae_score = mean_absolute_error(q_y_preds_proba_clear, q_y_preds_proba_fhe)

        # Track this MAE metric as well
        progress.measure(
            id="clear-vs-fhe-mae",
            label="MAE score between clear and FHE quantized probabilities",
            value=mae_score,
        )

    else:
        print("\nExecution of the inference in FHE skipped.")


def argument_manager():
    """Parse the command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlir_only", type=int, help="Dump the MLIR graph and stop (no inference executed)."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print more information, such as execution timings."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=random.randint(0, 2**32 - 1),
        help="Set the seed for reproducibility. A new seed is randomly generated by default.",
    )
    parser.add_argument(
        "--datasets",
        choices=CNN_DATASETS,
        type=str,
        nargs="+",
        default=None,
        help="Dataset(s) to use. All data-sets are chosen by default.",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        choices=CNN_STRING_TO_CLASS.keys(),
        help="Chose a CNN to benchmark. All models are chosen by default.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        type=json.loads,
        default=None,
        help="Config(s) to use, such as the n_bits parameter.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Indicate if the model(s) need(s) to be trained again. Else, it uses stored .pt files",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="The number of epochs to use during training.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="The batch_size to use during training.",
    )
    parser.add_argument(
        "--fhe_samples",
        type=int,
        default=1,
        help="The number of FHE samples on which to predict.",
    )
    parser.add_argument(
        "--model_samples",
        type=int,
        default=1,
        help="Number of times each tests should be executed (overwrites PROGRESS_SAMPLES).",
    )
    parser.add_argument(
        "--long_list",
        action="store_true",
        help="List all tasks and stop.",
    )
    parser.add_argument(
        "--short_list",
        action="store_true",
        help="List a task per each models and stop.",
    )
    parser.add_argument(
        "--dont_execute_in_fhe",
        action="store_true",
        help="Don't execute the FHE inference (default is to use should_test_config_in_fhe)",
    )

    args = parser.parse_args()

    if args.datasets is None:
        args.datasets = CNN_DATASETS

    if args.models is None:
        args.models = CNN_CLASSES
    else:
        args.models = [CNN_STRING_TO_CLASS[cnn_name] for cnn_name in args.models]

    return args


def main():
    """Main function to execute."""

    # Retrieve the parameters from the command line
    args = argument_manager()

    # Seed everything using a see
    seed_everything(args.seed)
    print(f"Using --seed {args.seed}")

    # Generate all the tasks to execute in this benchmark file
    all_tasks = list(benchmark_generator(args))

    if args.long_list or args.short_list:
        # Print the short or long lists if asked and stop
        printed_models = set()
        for dataset, cnn_class, config in all_tasks:
            configs = json.dumps(config).replace("'", '"')
            cnn_name = cnn_class.__name__

            # Only print one config per model if printing the short_list
            if not (args.short_list and cnn_name in printed_models):
                print(
                    f"--models {cnn_name} --dataset {dataset} --configs '{configs}'"
                    + f" --train --epochs {args.epochs}" * args.train
                    + f" --fhe_samples {args.fhe_samples}" * (not args.train)
                )
                printed_models.add(cnn_name)
        return

    if args.train:
        # Train each models on each data-sets and stop
        for dataset in args.datasets:
            for cnn_class in args.models:
                train_cnn_model(cnn_class, dataset, args.epochs, args.batch_size)

        return

    # Pylint does not seem to properly understand the following loop and needs to be disabled
    # pylint: disable=undefined-loop-variable
    @progress.track(
        [
            {
                "id": benchmark_name_generator(dataset, cnn_class, config, "_"),
                "name": benchmark_name_generator(dataset, cnn_class, config, " on "),
                "samples": args.model_samples,
                "parameters": {"cnn_class": cnn_class, "dataset": dataset, "config": config},
            }
            for (dataset, cnn_class, config) in all_tasks
        ]
    )
    def perform_deep_learning_benchmark(dataset, cnn_class, config):
        """
        This is the test function called by the py-progress module. It just calls the
        benchmark function with the right parameter combination
        """
        evaluate_pre_trained_cnn_model(dataset, cnn_class, config, args)


if __name__ == "__main__":
    main()
