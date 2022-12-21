#!/usr/bin/env python
# coding: utf-8
import time
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from concrete.numpy import Configuration
from model import CNV

from concrete.ml.torch.compile import compile_brevitas_qat_model


def main():
    model = CNV(num_classes=10, weight_bit_width=2, act_bit_width=2, in_bit_width=3, in_ch=3)
    loaded = torch.load(Path(__file__).parent / "8_bit_model.pt")
    model.load_state_dict(loaded["model_state_dict"])
    model = model.eval()
    IMAGE_TRANSFORM = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

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
        p_error=None,  # To avoid any confusion: we are always using kwarg p_error
        global_p_error=None,  # To avoid any confusion: we are always using kwarg global_p_error
    )

    # Compile the model
    compilation_onnx_path = "compilation_model.onnx"
    print("Compiling the model ...")
    start_compile = time.time()
    quantized_numpy_module = compile_brevitas_qat_model(
        torch_model=model.encrypted_module,  # our model
        torch_inputset=train_features_sub_set,  # a representative inputset to be used for both quantization and compilation
        n_bits=8,
        configuration=configuration,
        p_error=0.05,
        output_onnx_file=compilation_onnx_path,
    )
    end_compile = time.time()
    print(f"Compilation finished in {end_compile - start_compile:.2f} seconds")

    # Key generation
    print("Generating keys ...")
    start_keygen = time.time()
    quantized_numpy_module.fhe_circuit.keygen()
    end_keygen = time.time()
    print(f"Keygen finished in {end_keygen - start_keygen:.2f} seconds")

    # Inference part
    print("Infering ...")
    start_inference = time.time()
    img, _ = train_set[0]
    # Clear extraction of the feature maps
    with torch.no_grad():
        feature_maps = model.clear_module(img[None, :])
    # Quantization of the feature maps
    quantized_feature_maps = quantized_numpy_module.quantize_input(feature_maps.numpy())
    # Encryption of the feature maps
    encryped_feature_maps = quantized_numpy_module.fhe_circuit.encrypt(quantized_feature_maps)
    # FHE computation
    start_run = time.time()
    encrypted_output = quantized_numpy_module.fhe_circuit.run(encryped_feature_maps)
    end_run = time.time()
    # Decryption of the output
    quantized_output = quantized_numpy_module.fhe_circuit.decrypt(encrypted_output)
    # Dequantization of the output
    output = quantized_numpy_module.dequantize_output(quantized_output)
    end_inference = time.time()
    print("Output:", output)
    print(f"FHE computation finished in {end_run - start_run:.2f} seconds")
    print(f"Full inference finished in {end_inference - start_inference:.2f} seconds")


if __name__ == "__main__":
    main()
