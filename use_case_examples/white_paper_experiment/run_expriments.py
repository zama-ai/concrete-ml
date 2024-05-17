import argparse
import random
import time
import warnings

import numpy as np
import torch
from torchvision import datasets, transforms
from utils_experiments import MEAN, STD, torch_inference

from concrete.ml.torch.compile import compile_torch_model

warnings.filterwarnings("ignore", category=UserWarning)

print("starting")

DEVICE = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Input size, 28x28 pixels, a standard size for MNIST images
INPUT_IMG_SIZE = 28

# Batch size
BATCH_SIZE = 64

# Seed to ensure reproducibility
SEED = 42

# Wether the experiments are run on PC or other machines, like HP7C on AWS
MACHINE = "PC"

# The timing and the accuracy recorded in the article
PAPER_NOTES_20 = [20, 115.52, 0.97]
PAPER_NOTES_50 = [50, 233.55, 0.93]

## Architecture

FEATURES_MAPS = [
    # Convolution layer, with:
    # in_channel=1, out_channels=1, kernel_size=3, stride=1, padding_mode='replicate'
    ("C", 1, 1, 3, 1, "replicate"),
]


# The article presents 3 neural network depths. In this notebook, we focus NN-20 and NN-50
# architectures. The parameter `nb_layers`: controls the depth of the NN.
def linear_layers(nb_layers: int, output_size: int):
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
        """MNIST Torch model.

        Args:
            nb_layers (int): Number of layers.
            output_size (int): Number of classes.
        """
        super().__init__()

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
                for t in linear_layers(self.nb_layers, self.output_size)
                if t[0] != "I"
            ]
        )

    def forward(self, x):
        x = self.features_maps(x)
        x = torch.nn.Flatten()(x)
        x = self.linears(x)
        return x


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a model on MNIST.")
    parser.add_argument("--nb_layers", type=int, required=True, help="Number of layers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n_bits", type=int, default=6, help="Quantization bit")

    parser.add_argument("--verbose", type=bool, default=True, help="Verbose")
    parser.add_argument("--rounding_threshold_bits", type=int, default=6, help="Rounding precision")
    parser.add_argument(
        "--machine", type=str, default="PC", help="On which machine the experiments are run"
    )

    args = parser.parse_args()

    g = torch.Generator()
    g.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # 1. Load data-set
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

    # 2. Instanciate the model with pre-trained weights
    fp32_mnist = Fp32MNIST(nb_layers=args.nb_layers).to(DEVICE)

    checkpoint = torch.load(
        f"./checkpoints/MNIST/NLP_{args.nb_layers}/fp32/MNIST_fp32_state_dict.pt",
        map_location=DEVICE,
    )
    fp32_mnist.load_state_dict(checkpoint)

    acc_train = torch_inference(fp32_mnist, train_loader, device=DEVICE)
    acc_test = torch_inference(fp32_mnist, test_loader, device=DEVICE)

    if args.verbose:
        print(
            f"FP32 {fp32_mnist.nb_layers}-layer MNIST network:\n"
            f"{acc_train:.3%} for the training set and {acc_test:.3%} for the test set"
        )

    # 3. Compile the model
    # The model is compiled through 'compile_torch_model' method
    q_module = compile_torch_model(
        fp32_mnist.to(DEVICE),
        torch_inputset=data_calibration,
        # Quantization bits precision
        n_bits=6,
        rounding_threshold_bits=5,
    )

    ma_bitwidth = q_module.fhe_circuit.graph.maximum_integer_bit_width()

    if args.verbose:
        print(f"Maximum bits in the circuit: {ma_bitwidth} after compilation")

    # 4. Evaluate the model
    fhe_timing = []
    y_predictions = []
    fhe_samples = 3

    # The model is evaluated through all the test data-set in 'simulation' mode
    for i, (data, labels) in enumerate(test_loader):

        data, labels = data.detach().cpu().numpy(), labels.detach().cpu().numpy()
        simulat_predictions = q_module.forward(data, fhe="simulate")
        y_predictions.extend(simulat_predictions.argmax(1) == labels)

        # Then, only 3 random samples are token to measure the timing in FHE mode
        if i <= fhe_samples:
            start_time = time.time()
            q_module.forward(data[0, None], fhe="execute")
            fhe_timing.append((time.time() - start_time) / 60.0)

    best_timing = np.min(fhe_timing)

    paper_notes = PAPER_NOTES_20 if args.nb_layers == 20 else PAPER_NOTES_50
    if args.verbose:
        print(
            f"Accuracy in simulated mode : {np.mean(y_predictions):.3%} for the test set\n"
            f"Timing in FHE: {best_timing:.3f} per sample."
        )

        print(
            f"On {MACHINE} device:\n"
            f"Compared to the data from the paper, which recorded for NN-{paper_notes[0]} an "
            f"inference time of {paper_notes[1]} seconds.\nWe observe a significant gain in this  "
            f"new Concrete ML version. The inference time has been reduced to "
            f"{best_timing:.2f} seconds.\nThis represents a reduction factor of "
            f"{round(paper_notes[1] / best_timing)}.\n"
        )
