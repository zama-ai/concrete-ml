#!/usr/bin/env python
# coding: utf-8
import os
import time
from pathlib import Path
from typing import List

import torch
import torchvision
import torchvision.transforms as transforms
from concrete.fhe import Circuit, Configuration
from model import CNV

from concrete.ml.deployment.fhe_client_server import FHEModelDev
from concrete.ml.torch.compile import compile_brevitas_qat_model

NUM_SAMPLES = int(os.environ.get("NUM_SAMPLES", 400))


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

    # Create a representative input-set that will be used used for both computing quantization
    # parameters and compiling the model
    with torch.no_grad():
        train_features_sub_set = model.clear_module(train_sub_set)

    # Multi-parameter strategy is used in order to speed-up the FHE executions
    configuration = Configuration(show_optimizer=True)

    compilation_onnx_path = "compilation_model.onnx"
    print("Compiling the model ...")
    start_compile = time.time()

    # Compile the quantized model
    quantized_numpy_module = compile_brevitas_qat_model(
        torch_model=model.encrypted_module,
        torch_inputset=train_features_sub_set,
        configuration=configuration,
        p_error=0.05,
        output_onnx_file=compilation_onnx_path,
    )
    end_compile = time.time()
    print(f"Compilation finished in {end_compile - start_compile:.2f} seconds")

    # Save the graph and mlir
    print("Saving graph and mlir to disk.")
    open("cifar10.graph", "w").write(str(quantized_numpy_module.fhe_circuit))
    open("cifar10.mlir", "w").write(quantized_numpy_module.fhe_circuit.mlir)

    dev = FHEModelDev(path_dir="./client_server", model=quantized_numpy_module)
    dev.save()

    # Key generation
    print("Generating keys ...")
    start_keygen = time.time()
    assert isinstance(quantized_numpy_module.fhe_circuit, Circuit)
    quantized_numpy_module.fhe_circuit.keygen()
    end_keygen = time.time()
    print(f"Keygen finished in {end_keygen - start_keygen:.2f} seconds")

    # Initialize file
    inference_file = Path("inference_results.csv")
    open(inference_file, "w", encoding="utf-8").close()

    # Inference part
    columns: List[str] = []
    for image_index in range(NUM_SAMPLES):
        print("Infering ...")
        img, label = train_set[image_index]  # Get the image

        # Clear extraction of the feature maps
        feature_extraction_start = time.time()
        with torch.no_grad():
            feature_maps = model.clear_module(img[None, :])
        feature_extraction_end = time.time()
        feature_extraction_time = feature_extraction_end - feature_extraction_start

        # Quantization of the feature maps
        quantization_start = time.time()
        quantized_feature_maps = quantized_numpy_module.quantize_input(feature_maps.numpy())
        quantization_end = time.time()
        quantization_time = quantization_end - quantization_start

        # Encryption of the feature maps
        encryption_start = time.time()
        encryped_feature_maps = quantized_numpy_module.fhe_circuit.encrypt(quantized_feature_maps)
        encryption_end = time.time()
        encryption_time = encryption_end - encryption_start

        # FHE computation
        fhe_start = time.time()
        encrypted_output = quantized_numpy_module.fhe_circuit.run(encryped_feature_maps)
        fhe_end = time.time()
        fhe_time = fhe_end - fhe_start

        # Decryption of the output
        decryption_start = time.time()
        quantized_output = quantized_numpy_module.fhe_circuit.decrypt(encrypted_output)
        decryption_end = time.time()
        decryption_time = decryption_end - decryption_start

        # De-quantization of the output
        dequantization_start = time.time()
        output = quantized_numpy_module.dequantize_output(quantized_output)
        dequantization_end = time.time()
        dequantization_time = dequantization_end - dequantization_start

        inference_time = dequantization_end - feature_extraction_start

        # Torch reference
        torch_start = time.time()
        with torch.no_grad():
            torch_output = model.encrypted_module(feature_maps).numpy()
        torch_end = time.time()
        torch_time = torch_end - torch_start

        # Dump everything in a csv
        to_dump = {
            "image_index": image_index,
            # Timings
            "feature_extraction_time": feature_extraction_time,
            "quantization_time": quantization_time,
            "encryption_time": encryption_time,
            "fhe_time": fhe_time,
            "decryption_time": decryption_time,
            "dequantization_time": dequantization_time,
            "inference_time": inference_time,
            "torch_time": torch_time,
            "label": label,
        }

        for prediction_index, prediction in enumerate(quantized_output[0]):
            to_dump[f"quantized_prediction_{prediction_index}"] = prediction
        for prediction_index, prediction in enumerate(output[0]):
            to_dump[f"prediction_{prediction_index}"] = prediction
        for prediction_index, prediction in enumerate(torch_output[0]):
            to_dump[f"torch_prediction_{prediction_index}"] = prediction

        # Write to file
        with open(inference_file, "a", encoding="utf-8") as file:
            if image_index == 0:
                columns = list(to_dump.keys())
                file.write(",".join(columns) + "\n")
            file.write(",".join(str(to_dump[column]) for column in columns) + "\n")

        print("Output:", output)
        print(f"FHE computation finished in {fhe_time:.2f} seconds")
        print(f"Full inference finished in {inference_time:.2f} seconds")


if __name__ == "__main__":
    main()
