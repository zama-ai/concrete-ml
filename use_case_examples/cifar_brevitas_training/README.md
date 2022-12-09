# Ternary VGG on CIFAR10 with Fully Homomorphic encryption

## Description

In this directory, i.e. `cifar_brevitas_training` we provide Python code for training and evaluating a VGG-like neural network using Brevitas and give a script to run the neural network in the Fully Homomorphic Encryption (FHE) settings.

Original files can be found in the [Brevitas](https://github.com/Xilinx/brevitas/) repository. The model in the `models/` folder has a few modifications from the original to make it compatible with Concrete-ML:

- `MaxPool` layers have been replaced by `AvgPool` layers. This is mainly because max pooling is a costly operation in FHE which we want to avoid for less FHE costly operations such as average pooling.
- Quantization is applied after each AvgPool as this is needed for Concrete-ML to capture the quantization parameter. A QuantizedIdenty Brevitas layer does it.
- The `x.view(x.shape[0], -1)` has been replaced by `torch.flatten(x, 1)` which offers equivalent operation but is compatible with Concrete-ML
- `x = 2.0 * x - torch.tensor([1.0], device=x.device)` line has been removed and applied during the pre-processing as this is a transformation on the raw data (normalization between -1 and +1).

## Installation

To use this code, you will need to have Python 3.8 and the following dependencies installed:

- concrete-ml
- torchvision

You can install these dependencies using pip and the requirements.txt file available in this directory as follows:

<!--pytest-codeblocks:skip-->

```bash
pip install -r requirements.txt
```

# Usage

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

## Fully Homomorphic Encryption (FHE)

Three files are available to test the Torch neural network in the FHE settings.

### Simulation in Concrete-ML

In Concrete-ML, you can test your model before running it in FHE such that you don't have to pay the cost of FHE runtime during development.

You can launch this evaluation as follows:

<!--pytest-codeblocks:skip-->

```bash
python3 evaluate_torch_cml.py
```

It evaluates the model with Torch and Concrete-ML in simulation mode (a representation of FHE circuit running in the clear) to compare the results.

### Fully Homomorphic Encryption

Once the model has been proposed to have a correct performance, compilation to the FHE settings can be done.

<!--pytest-codeblocks:skip-->

```bash
python3 evaluation_one_example_fhe.py
```

Here, a picture from the CIFAR10 data-set is randomly chosen and preprocessed. The data is then quantized, encrypted and then given to the FHE circuit that evaluates the encrypted image. The result, encrypted as well, is then decrypted and compared vs. the expected output coming from the clear inference.

Warning: this execution can be quite costly.

# Accuracy

| Runtime                | Accuracy |
| ---------------------- | -------- |
| VGG Torch              | 88.9     |
| VGG Concrete-ML        | 88.7     |
| VGG FHE (simulation\*) | 88.7     |

FIXME (https://github.com/zama-ai/concrete-ml-internal/issues/2350): add actual FHE accuracy and performance.

\* The simulation is done using Virtual Library (VL) that simulates the FHE evaluation in the clear for faster debugging.
