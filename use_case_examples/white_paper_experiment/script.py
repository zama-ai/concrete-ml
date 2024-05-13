import csv
import os
import random
import time
import warnings
from glob import glob
from typing import Callable, List

import brevitas
import brevitas.nn as qnn
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from utils_experiments import MEAN, STD

from concrete.ml.torch.compile import compile_brevitas_qat_model, compile_torch_model

warnings.filterwarnings("ignore", category=UserWarning)

print("starting")

DEVICE = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Input size, 28x28 pixels, a standard size for MNIST images
INPUT_IMG_SIZE = 28

# Batch size
BATCH_SIZE = 64

# Seed to ensure reproducibility
SEED = 42

FEATURES_MAPS = [
    # The Bravitas.QuantIdentiy layer, only used in the quant NN, aims to quantize the input.
    # ("I",),
    # Convolution layer, with:
    # in_channel=1, out_channels=1, kernel_size=3, stride=1, padding_mode='replicate'
    ("C", 1, 1, 3, 1, "replicate"),
]


# The article presents 3 neural network depths. In this notebook, we focus NN-20 and NN-50
# architectures. The parameter `nb_layers`: controls the depth of the NN.
def LINEAR_LAYERS(nb_layers: int, output_size: int):
    return (  # noqa: W503
        [
            ("R",),
            ("L", INPUT_IMG_SIZE * INPUT_IMG_SIZE, 92),
            ("B", 92),
        ]  # noqa: W503
        + [  # noqa: W503
            ("R",),
            ("L", 92, 92),
            ("B", 92),
        ]
        * (nb_layers - 2)  # noqa: W503
        + [("L", 92, output_size)]  # noqa: W503
    )


class Fp32MNIST(torch.nn.Module):
    """MNIST Torch model."""

    def __init__(self, nb_layers: int, output_size: int = 10):
        super(Fp32MNIST, self).__init__()
        """MNIST Torch model.

        Args:
            nb_layers (int): Number of layers.
            output_size (int): Number of classes.
        """
        self.nb_layers = nb_layers
        self.output_size = output_size

        def make_layers(t):
            if t[0] == "C":
                # Workaround: stride=1, padding_mode='replicate' is replaced by
                # transforms.Pad(1, padding_mode="edge")
                return torch.nn.Conv2d(
                    in_channels=t[1],
                    out_channels=t[2],
                    kernel_size=t[3],
                )
            if t[0] == "L":
                return torch.nn.Linear(in_features=t[1], out_features=t[2])
            if t[0] == "R":
                return torch.nn.ReLU()
            if t[0]:
                return torch.nn.BatchNorm1d(t[1])

            raise NameError(f"'{t}' not defined")

        # QuantIdentity layers are ignored in the floationg point architecture.
        self.features_maps = torch.nn.Sequential(
            *[make_layers(t) for t in FEATURES_MAPS if t[0] != "I"]
        )
        self.linears = torch.nn.Sequential(
            *[
                make_layers(t)
                for t in LINEAR_LAYERS(self.nb_layers, self.output_size)
                if t[0] != "I"
            ]
        )

    def forward(self, x):
        x = self.features_maps(x)
        x = torch.nn.Flatten()(x)
        x = self.linears(x)
        return x


class QuantMNIST(torch.nn.Module):
    """Quantized MNIST network with Brevitas."""

    def __init__(
        self,
        n_bits: int,
        nb_layers: int,
        output_size: int = 10,
        act_quant: brevitas.quant = brevitas.quant.Int8ActPerTensorFloat,
        weight_quant: brevitas.quant = brevitas.quant.Int8WeightPerTensorFloat,
    ):
        """A quantized network with Brevitas.

        Args:
            n_bits (int): Precision bit of quantization.
            nb_layers (int): Number of layers.
            output_size (int): Number of classes
            act_quant (brevitas.quant): Quantization protocol of activations.
            weight_quant (brevitas.quant): Quantization protocol of the weights.
        """
        super(QuantMNIST, self).__init__()

        self.n_bits = n_bits
        self.nb_layers = nb_layers
        self.output_size = output_size

        def tuple2quantlayer(t):
            if t[0] == "R":
                return qnn.QuantReLU(
                    return_quant_tensor=True, bit_width=n_bits, act_quant=act_quant
                )
            if t[0] == "C":
                # Workaround: stride=1, padding_mode='replicate' is replaced by
                # transforms.Pad(1, padding_mode="edge")
                return qnn.QuantConv2d(
                    t[1],
                    t[2],
                    kernel_size=t[3],
                    weight_bit_width=2,
                    weight_quant=weight_quant,
                    return_quant_tensor=True,
                )
            if t[0] == "L":
                return qnn.QuantLinear(
                    in_features=t[1],
                    out_features=t[2],
                    weight_bit_width=n_bits,
                    weight_quant=weight_quant,
                    bias=True,
                    return_quant_tensor=True,
                )
            if t[0] == "I":
                identity_quant = t[1] if len(t) == 2 else n_bits
                return qnn.QuantIdentity(
                    bit_width=identity_quant, act_quant=act_quant, return_quant_tensor=True
                )
            if t[0] == "B":
                return torch.nn.BatchNorm1d(t[1])

        self.features_maps = torch.nn.Sequential(
            *[tuple2quantlayer(t) for t in FEATURES_MAPS if t[0]]
        )

        # self.identity1 and self.identity2 are used to encapsulate the `torch.flatten`.
        self.identity1 = qnn.QuantIdentity(
            bit_width=n_bits, act_quant=act_quant, return_quant_tensor=True
        )
        self.identity2 = qnn.QuantIdentity(
            bit_width=n_bits, act_quant=act_quant, return_quant_tensor=True
        )

        # QuantIdentity layers are taken into account
        self.linears = torch.nn.Sequential(
            *[tuple2quantlayer(t) for t in LINEAR_LAYERS(self.nb_layers, self.output_size)]
        )

    def forward(self, x):
        x = self.features_maps(x)
        x = self.identity1(x)
        x = torch.flatten(x, 1)
        x = self.identity2(x)
        x = self.linears(x)
        return x.value


def run_benchmark(
    model: Callable,
    compile_function: Callable,
    compile_type: str,
    data_calibration,
    data_loader,
    n_bits: int,
):
    nb_layers = model.nb_layers

    history = []
    filename = f"./benchmark/history_{nb_layers=}.csv"

    headers = [
        "compile_type",
        "number_of_layers",
        "QAT/PTQ_n_bits",
        "threshold_n_bits",
        "threshold_method",
        "max_bits",
        "mean_FP32_accuracy",
        "mean_disable_accuracy",
        "mean_simulate_accuracy",
        "FHE_timing",
        "INPUT_COMPRESSION",
        "machine",
    ]

    thresholds = [
        {"n_bits": 6, "method": "APPROXIMATE"},
    ]

    if not os.path.isfile(filename):
        with open(filename, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(headers)

    with open(filename, "a", newline="") as file:
        writer = csv.writer(file)

        for j, threshold in enumerate(thresholds):

            fhe_timing = -1
            history_fp32_predictions = []
            history_disable_predictions = []
            history_simulat_predictions = []

            try:
                q_module = compile_function(
                    model.to(DEVICE),
                    torch_inputset=data_calibration,
                    n_bits=n_bits if compile_type == "PTQ" else None,
                    rounding_threshold_bits=threshold,
                )

                max_bits = q_module.fhe_circuit.graph.maximum_integer_bit_width()
                print(f"{j=}: {max_bits=} after compilation")

                for i, (data, labels) in enumerate(data_loader):

                    fp32_predictions = model(data).cpu().detach()
                    history_fp32_predictions.extend(fp32_predictions.argmax(1) == labels)

                    data, labels = data.detach().cpu().numpy(), labels.detach().cpu().numpy()

                    disable_predictions = q_module.forward(data, fhe="disable")
                    history_disable_predictions.extend(disable_predictions.argmax(1) == labels)

                    simulat_predictions = q_module.forward(data, fhe="simulate")
                    history_simulat_predictions.extend(simulat_predictions.argmax(1) == labels)

                    if i == 0:
                        start_time = time.time()
                        q_module.forward(data[0, None], fhe="execute")
                        fhe_timing = (time.time() - start_time) / 60.0

                row = [
                    compile_type,
                    nb_layers,
                    n_bits,
                    threshold["n_bits"] if threshold is not None else threshold,
                    threshold["method"] if threshold is not None else threshold,
                    max_bits,
                    np.mean(history_fp32_predictions),
                    np.mean(history_disable_predictions),
                    np.mean(history_simulat_predictions),
                    fhe_timing,
                    os.environ.get("USE_INPUT_COMPRESSION", "1"),
                    "HP7C",
                ]

            except BaseException:
                row = [
                    compile_type,
                    nb_layers,
                    n_bits,
                    threshold["n_bits"] if threshold is not None else threshold,
                    threshold["method"] if threshold is not None else threshold,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                    -1,
                ]

            writer.writerow(row)
            print(f"{j=}: {row=}")
            history.append(dict(zip(headers, row)))

    return history


def ptq_benchmark(
    nb_layers: int,
    data_calibration: datasets,
    data_loader: DataLoader,
    n_bits: List = [5, 6, 7],
    model_dir: str = "./checkpoints/MNIST/",
):
    # Test all fp32 pre-trained models, saved in the following path
    state_dict_paths = glob(f"{model_dir}/NLP_{nb_layers}/fp32/*fp32_state_dict.pt")
    assert len(state_dict_paths) > 0, f"There are no fp32 pre-trained models inside {model_dir}."

    for state_dict_path in state_dict_paths:
        checkpoint = torch.load(state_dict_path, map_location=torch.device("cpu"))
        fp32_model = Fp32MNIST(nb_layers=nb_layers).to(DEVICE)
        fp32_model.load_state_dict(checkpoint)
        print(f"FP32 model loaded in: '{state_dict_path}'")

        # In PTQ, we have to specify the quantization bit precision
        for bit in tqdm(n_bits):
            run_benchmark(
                fp32_model,
                compile_function=compile_torch_model,
                compile_type="PTQ",
                data_calibration=data_calibration,
                data_loader=data_loader,
                n_bits=bit,
            )


def qat_benchmark(
    nb_layers: int,
    data_calibration: datasets,
    data_loader: DataLoader,
    model_dir: str = "./checkpoints/MNIST",
):
    # Test all quantized pre-trained models, saved in the following path
    state_dict_paths = glob(f"{model_dir}/NLP_{nb_layers}/quant_*/*_state_dict.pt")
    assert len(state_dict_paths) > 0, f"There are no pretrained models in {model_dir}"

    for state_dict_path in state_dict_paths:
        checkpoint = torch.load(state_dict_path, map_location=torch.device("cpu"))

        bit = int(state_dict_path.split("/")[-2].split("=")[-1])
        quant_model = QuantMNIST(n_bits=bit, nb_layers=nb_layers).to(DEVICE)
        quant_model.load_state_dict(checkpoint)
        print(f"Quantized model loaded in: '{state_dict_path}'")

        run_benchmark(
            quant_model,
            compile_function=compile_brevitas_qat_model,
            compile_type="QAT",
            data_calibration=data_calibration,
            data_loader=data_loader,
            n_bits=bit,
        )


if __name__ == "__main__":

    g = torch.Generator()
    g.manual_seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    train_transform = transforms.Compose(
        [  # Workaround: stride=1, padding_mode='replicate' is replaced by
            # transforms.Pad(1, padding_mode="edge")
            transforms.Pad(1, padding_mode="edge"),
            transforms.ToTensor(),
            transforms.Normalize((MEAN,), (STD,)),
            transforms.GaussianBlur(kernel_size=(3, 3)),
        ]
    )
    test_transform = transforms.Compose(
        [  # Workaround: stride=1, padding_mode='replicate' is replaced by
            # transforms.Pad(1, padding_mode="edge")
            transforms.Pad(1, padding_mode="edge"),
            transforms.ToTensor(),
            transforms.Normalize((MEAN,), (STD,)),
        ]
    )

    train_dataset = datasets.MNIST(
        download=True, root="./data", train=True, transform=train_transform
    )

    test_dataset = datasets.MNIST(
        download=True, root="./data", train=False, transform=test_transform
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        generator=g,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        generator=g,
    )

    data_calibration = next(iter(train_loader))[0]

    print(data_calibration.shape)

    qat_benchmark(nb_layers=50, data_loader=test_loader, data_calibration=data_calibration)
    qat_benchmark(nb_layers=20, data_loader=test_loader, data_calibration=data_calibration)
    ptq_benchmark(nb_layers=20, data_loader=test_loader, data_calibration=data_calibration)
    ptq_benchmark(nb_layers=50, data_loader=test_loader, data_calibration=data_calibration)
