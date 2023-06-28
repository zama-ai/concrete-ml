# Ternary VGG on CIFAR10 with Fully Homomorphic encryption

## Description

In this directory we provide Python code for training, from scratch, a VGG-like neural network using Brevitas on CIFAR-10. We also give a script to run the neural network in the Fully Homomorphic Encryption (FHE) settings.

Original files can be found in the [Brevitas](https://github.com/Xilinx/brevitas/) repository. The model in the `models/` folder has a few modifications from the original to make it compatible with Concrete ML:

- `MaxPool` layers have been replaced by `AvgPool` layers. This is mainly because max pooling is a costly operation in FHE which we want to avoid for less FHE costly operations such as average pooling.
- Quantization is applied after each AvgPool as this is needed for Concrete ML to capture the quantization parameter. A QuantIdentity Brevitas layer achieves this.

## Installation

To use this code, you need to have Python 3.8 and install the following dependencies:

<!--pytest-codeblocks:skip-->

```bash
pip install -r requirements.txt
```

## Training and inference

The files in this section are almost as identical to the original. Here we train a VGG-like neural network using an example available on [Brevitas Github repository](https://github.com/Xilinx/brevitas/blob/8c3d9de0113528cf6693c6474a13d802a66682c6/src/brevitas_examples/bnn_pynq/).

To train the neural network:

<!--pytest-codeblocks:skip-->

```bash
python3 bnn_pynq_train.py --data ./data --experiments ./experiments
```

To evaluate the trained model:

<!--pytest-codeblocks:skip-->

```bash
python3 bnn_pynq_train.py --evaluate --resume ./experiments/CNV_2W2A_2W2A_20221114_131345/checkpoints/best.tar
```

### Simulation in Concrete ML

In Concrete ML, you can test your model before running it in FHE such that you don't have to pay the cost of FHE runtime during development.

You can launch this evaluation as follows:

<!--pytest-codeblocks:skip-->

```bash
python3 evaluate_torch_cml.py
```

### Rounding for Improved Performance

In Concrete, a rounding operator is available which removes a specific number of least significant bits to reach a lower desired bit-width. This results in significant performance improvement. The default rounding threshold is set at 6 bits but can be changed to suit your needs.

<!--pytest-codeblocks:skip-->

```bash
python3 evaluate_torch_cml.py --rounding_threshold_bits 8
```

Testing with different rounding_threshold_bits values can help you understand the impact on the final accuracy:

<!--pytest-codeblocks:skip-->

```bash
python3 evaluate_torch_cml.py --rounding_threshold_bits 1 2 3 4 5 6 7 8
```

Using rounding with 6 bits for all accumulators provides a significant speedup for FHE, with only a 1.3% loss in accuracy compared to the original model. More details can be found in the Accuracy and Performance section below.

## Fully Homomorphic Encryption (FHE)

Once you're satisfied with the model's performance, you can compile it to the FHE settings.

<!--pytest-codeblocks:skip-->

```bash
python3 evaluation_one_example_fhe.py
```

Here, an image from the CIFAR10 data-set is randomly chosen and preprocessed. The data is then quantized, encrypted and then given to the FHE circuit that evaluates the encrypted image. The result, encrypted as well, is then decrypted and compared vs. the expected output coming from the clear inference.

## Hardware Used for the Experiment

Experiments were conducted on an m6i.metal machine offering 128 CPU cores and 512GB of memory. The choice of hardware can significantly influence the execution time and performance of the model.

## Accuracy and performance

| Runtime                | Rounding | Accuracy |
| ---------------------- | -------- | -------- |
| VGG Torch              | None     | 88.7     |
| VGG FHE (simulation\*) | None     | 88.7     |
| VGG FHE (simulation\*) | 8 bits   | 88.3     |
| VGG FHE (simulation\*) | 7 bits   | 88.3     |
| VGG FHE (simulation\*) | 6 bits   | 87.5     |
| VGG FHE (simulation\*) | 5 bits   | 84.9     |
| VGG FHE                | 6 bits   | 87.5\*\* |

We ran the FHE inference over 10 examples and achieved 100% similar predictions between the simulation and FHE. The overall accuracy for the entire data-set is expected to match the simulation. The original model with a maximum of 13 bits of precision ran in around 9 hours on the specified hardware. Using the rounding approach, the final model ran in **31 minutes**, providing a speedup factor of 18x while preserving accuracy. This significant performance improvement demonstrates the benefits of the rounding operator in the FHE setting.

\* Simulation is used to evaluate the accuracy in the clear for faster debugging.
\*\* We ran the FHE inference over 10 examples and got 100% similar predictions between the simulation and FHE. The overall accuracy for the entire data-set is expected to match the simulation.
