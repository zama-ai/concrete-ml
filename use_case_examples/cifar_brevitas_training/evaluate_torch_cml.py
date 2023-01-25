import pathlib

import numpy as np
import torch
from concrete.numpy.compilation.configuration import Configuration
from models import cnv_2w2a
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from trainer import accuracy

from concrete.ml.torch.compile import compile_brevitas_qat_model


def cml_inference(quantized_numpy_module, x_numpy):
    predictions = np.zeros(shape=(x_numpy.shape[0], 10))
    for idx, x in enumerate(x_numpy):
        x_q = np.expand_dims(x, 0)
        circuit_prediction = quantized_numpy_module.forward_fhe.encrypt_run_decrypt(x_q)
        predictions[idx] = circuit_prediction
    return predictions


def evaluate(torch_model, cml_model):
    # Import the CIFAR data (following bnn_pynq_train.py)

    transform_to_tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: 2.0 * x - 1.0),
        ]  # Normalizes data between -1 and +1s
    )

    builder = CIFAR10

    test_set = builder(root=".datasets/", train=False, download=True, transform=transform_to_tensor)

    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device in use:", device)
    top1_torch = []
    top5_torch = []

    top1_cml = []
    top5_cml = []

    # Model to device
    torch_model = torch_model.to(device)
    for i, data in enumerate(test_loader):

        (input, target) = data

        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # Compute torch output
        output = torch_model(input)

        # Compute cml output
        numpy_input = input.detach().cpu().numpy()

        q_numpy_input = cml_model.quantize_input(numpy_input)
        y_cml_vl = cml_inference(cml_model, q_numpy_input)
        y_cml_vl = cml_model.dequantize_output(y_cml_vl)

        # y_cml_vl to torch to device
        y_cml_vl = torch.tensor(y_cml_vl).to(device)

        # Compute torch loss
        pred = output.data.argmax(1, keepdim=True)
        correct = pred.eq(target.data.view_as(pred)).sum()
        prec1 = 100.0 * correct.float() / input.size(0)

        _, prec5 = accuracy(output, target, topk=(1, 5))

        top1_torch.append(prec1.item())
        top5_torch.append(prec5.item())

        # Compute cml loss
        pred = y_cml_vl.data.argmax(1, keepdim=True)
        correct = pred.eq(target.data.view_as(pred)).sum()
        prec1 = 100.0 * correct.float() / input.size(0)

        _, prec5 = accuracy(y_cml_vl, target, topk=(1, 5))

        top1_cml.append(prec1.item())
        top5_cml.append(prec5.item())

    print("Torch accuracy top1:", np.mean(top1_torch))
    print("CML accuracy top1:", np.mean(top1_cml))

    print("Torch accuracy top5:", np.mean(top5_torch))
    print("CML accuracy top5:", np.mean(top5_cml))


def main():
    model = cnv_2w2a(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Find relative path to this file
    dir_path = pathlib.Path(__file__).parent.absolute()

    # Load checkpoint
    checkpoint = torch.load(
        dir_path / "./experiments/CNV_2W2A_2W2A_20221114_131345/checkpoints/best.tar",
        map_location=device,
    )

    # Load weights
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    # Get some random data with the right shape
    x = torch.randn(1, 3, 32, 32)

    # Eval mode
    model.eval()

    # Compile with CML using the simulation mode (VL)
    cfg = Configuration(
        dump_artifacts_on_unexpected_failures=False,
        enable_unsafe_features=True,  # This is for our tests only, never use that in prod
    )
    quantized_numpy_module = compile_brevitas_qat_model(
        model,  # our torch model
        x,  # a representative inputset to be used for both quantization and compilation
        n_bits={"model_inputs": 8, "op_inputs": 2, "op_weights": 2, "model_outputs": 8},
        use_virtual_lib=True,  # important to run in simulated FHE
        configuration=cfg,
    )

    # Print max bitwidth in the circuit
    print(
        "Max bitwidth in the circuit: ",
        quantized_numpy_module.fhe_circuit.graph.maximum_integer_bit_width(),
    )

    # Evaluate torch and CML model
    evaluate(model, quantized_numpy_module)


if __name__ == "__main__":
    main()
