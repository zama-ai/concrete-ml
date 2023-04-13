"""Load torch model, compiles it to FHE and exports it"""
import sys
import time
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from concrete.fhe import Configuration
from model import CNV

from concrete.ml.deployment import FHEModelDev
from concrete.ml.torch.compile import compile_brevitas_qat_model


def main():
    # Load model
    model = CNV(num_classes=10, weight_bit_width=2, act_bit_width=2, in_bit_width=3, in_ch=3)
    loaded = torch.load(Path(__file__).parent / "8_bit_model.pt")
    model.load_state_dict(loaded["model_state_dict"])
    model = model.eval()

    IMAGE_TRANSFORM = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # Load data
    try:
        train_set = torchvision.datasets.CIFAR10(
            root=".data/",
            train=True,
            download=False,
            transform=IMAGE_TRANSFORM,
            target_transform=None,
        )
    except:
        train_set = torchvision.datasets.CIFAR10(
            root=".data/",
            train=True,
            download=True,
            transform=IMAGE_TRANSFORM,
            target_transform=None,
        )

    num_samples = 1000
    train_sub_set = torch.stack(
        [train_set[index][0] for index in range(min(num_samples, len(train_set)))]
    )

    # Pre-processing -> images -> feature maps
    with torch.no_grad():
        train_features_sub_set = model.clear_module(train_sub_set)

    configuration = Configuration(
        show_graph=False,
        show_mlir=False,
        show_optimizer=False,
        p_error=None,
        global_p_error=None,
    )

    # Compile the model
    compilation_onnx_path = "compilation_model.onnx"
    print("Compiling the model ...")
    start_compile = time.time()
    quantized_numpy_module = compile_brevitas_qat_model(
        torch_model=model.encrypted_module,  # our model
        torch_inputset=train_features_sub_set,  # a representative input-set to be used for both quantization and compilation
        configuration=configuration,
        p_error=0.05,
        output_onnx_file=compilation_onnx_path,
        n_bits=8,
    )
    end_compile = time.time()
    print(f"Compilation finished in {end_compile - start_compile:.2f} seconds")

    # Key generation
    print("Generating keys ...")
    start_keygen = time.time()
    quantized_numpy_module.fhe_circuit.keygen()
    end_keygen = time.time()
    print(f"Keygen finished in {end_keygen - start_keygen:.2f} seconds")

    print("size_of_inputs", quantized_numpy_module.fhe_circuit.size_of_inputs)
    print("bootstrap_keys", quantized_numpy_module.fhe_circuit.size_of_bootstrap_keys)
    print("keyswitches", quantized_numpy_module.fhe_circuit.size_of_keyswitch_keys)

    dev = FHEModelDev(path_dir="./dev", model=quantized_numpy_module)
    dev.save()


if __name__ == "__main__":
    main()
