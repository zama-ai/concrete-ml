import argparse
import pathlib

import numpy as np
import torch
from concrete.fhe import Configuration, ParameterSelectionStrategy
from models import cnv_2w2a
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm
from trainer import accuracy

from concrete.ml.torch.compile import compile_brevitas_qat_model


def evaluate(torch_model, cml_model, device):

    # Transform pipeline that normalizes the data between -1 and 1
    transform_to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2.0 * x - 1.0),
        ]
    )

    # Import and load the CIFAR test dataset (following bnn_pynq_train.py)
    test_set = CIFAR10(root=".datasets/", train=False, download=True, transform=transform_to_tensor)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=0)

    torch_top_1_batches = []
    torch_top_5_batches = []

    concrete_top_1_batches = []
    concrete_top_5_batches = []

    torch_model = torch_model.to(device)

    for _, data in enumerate(tqdm(test_loader)):

        (input, target) = data

        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Compute Torch output
        torch_output = torch_model(input)

        # Concrete ML inference only handles Numpy inputs
        numpy_input = input.detach().cpu().numpy()

        # Compute Concrete ML output using simulation
        concrete_output_simulated = cml_model.forward(numpy_input, fhe="simulate")

        concrete_output_simulated = torch.tensor(concrete_output_simulated).to(device)

        # Compute Torch top accuracies
        torch_top_1, torch_top_5 = accuracy(torch_output, target, topk=(1, 5))

        torch_top_1_batches.append(torch_top_1.item())
        torch_top_5_batches.append(torch_top_5.item())

        # Compute Concrete ML top accuracies
        concrete_top_1, concrete_top_5 = accuracy(concrete_output_simulated, target, topk=(1, 5))

        concrete_top_1_batches.append(concrete_top_1.item())
        concrete_top_5_batches.append(concrete_top_5.item())

    print("Torch accuracy top1:", np.mean(torch_top_1_batches))
    print("Concrete ML accuracy top1:", np.mean(concrete_top_1_batches))

    print("Torch accuracy top5:", np.mean(torch_top_5_batches))
    print("Concrete ML accuracy top5:", np.mean(concrete_top_5_batches))


def main(args):
    rounding_threshold_bits_list = args.rounding_threshold_bits

    model = cnv_2w2a(False)

    # Add MPS (for macOS with Apple Silicon or AMD GPUs) support when error is fixed. For now, we
    # observe a decrease in torch's top1 accuracy when using MPS devices
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/3953
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device in use:", device)

    # Find relative path to this file
    dir_path = pathlib.Path(__file__).parent.absolute()

    # Load checkpoint
    checkpoint = torch.load(
        dir_path / "./experiments/CNV_2W2A_2W2A_20221114_131345/checkpoints/best.tar",
        map_location=device,
    )

    # Load weights
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    # Create a representative input-set that will be used used for both computing quantization
    # parameters and compiling the model
    input_set = torch.randn(1, 3, 32, 32)

    # Eval mode
    model.eval()

    # Multi-parameter strategy is used in order to speed-up the FHE executions
    cfg = Configuration(
        verbose=True,
        show_optimizer=args.show_optimizer,
        parameter_selection_strategy=ParameterSelectionStrategy.MULTI,
    )

    for rounding_threshold_bits in rounding_threshold_bits_list:
        print(f"Testing network with {rounding_threshold_bits} rounding bits")

        # Compile the quantized model
        quantized_numpy_module = compile_brevitas_qat_model(
            model,
            input_set,
            n_bits={"model_inputs": 8, "model_outputs": 8},
            configuration=cfg,
            rounding_threshold_bits=rounding_threshold_bits,
        )

        # Print max bit-width in the circuit
        print(
            "Max bit-width in the circuit: ",
            quantized_numpy_module.fhe_circuit.graph.maximum_integer_bit_width(),
        )

        # Evaluate torch and Concrete ML model
        evaluate(model, quantized_numpy_module, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rounding_threshold_bits",
        nargs="+",
        type=int,
        default=[None],
        help="Number of bits to target with rounding.",
    )
    parser.add_argument(
        "--show_optimizer",
        action="store_true",
        help="Display optimizer parameters after compiling the model.",
    )

    args = parser.parse_args()
    main(args)
