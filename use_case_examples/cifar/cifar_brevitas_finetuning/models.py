import brevitas
import brevitas.nn as qnn
import torch
import torch.nn as nn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat

""" In this models.py we provide the code for the PyTorch and Brevitas networks."""

# This architecture is inspired by the original VGG-11 network available in
# PyTorch.hub (https://pytorch.org/hub/pytorch_vision_vgg/)

# Each tuple refers to a PyTorch or Brevitas layer:
# I: QuantIdentity layer, only required for the Brevitas network. Mainly used to quantize
# the input data or to encapsulate a PyTorch layer inside the Brevitas model.
# C: Convolutional layer.
# P: Pooling layer, we replaced the original `MaxPool2d` in VGG-11 by a `AvgPool2d` layer.
# Because in the current version of Concrete ML `MaxPool2d` isn't available yet.
# R: ReLU activation.
FEATURES_MAPS = [
    ("I",),
    ("C", 3, 64, 3, 1, 1),
    ("R",),
    ("P", 2, 1, 0, 1, False),  # Original values in VGG-11: k=2, s=1.
    ("I",),
    ("C", 64, 128, 3, 1, 1),
    ("R",),
    ("P", 2, 2, 0, 1, False),
    ("I",),
    ("C", 128, 256, 3, 1, 1),
    ("R",),
    ("C", 256, 256, 3, 1, 1),
    ("R",),
    ("P", 2, 2, 0, 1, False),
    ("I",),
    ("C", 256, 512, 3, 1, 1),
    ("R",),
    ("C", 512, 512, 3, 1, 1),
    ("R",),
    ("P", 1, 1, 0, 1, False),  # Original values in VGG-11: k=2, s=1.
    ("I",),
    ("C", 512, 512, 3, 1, 1),
    ("R",),
    ("C", 512, 512, 3, 1, 1),
    ("R",),
    ("P", 7, 1, 0, 1, False),  # Original values in VGG-11: k=2, s=1.
    ("I",),
]


class Fp32VGG11(nn.Module):
    def __init__(self, output_size: int):
        super(Fp32VGG11, self).__init__()
        """ Torch model.
     
        Args:
            output_size (int): Number of classes.
        """
        self.output_size = output_size

        def make_layers(t):

            if t[0] == "P":
                return nn.AvgPool2d(kernel_size=t[1], stride=t[2], padding=t[3], ceil_mode=t[5])
            elif t[0] == "C":
                return nn.Conv2d(t[1], t[2], kernel_size=t[3], stride=t[4], padding=t[5])
            elif t[0] == "L":
                return nn.Linear(in_features=t[1], out_features=t[2])
            elif t[0] == "R":
                return nn.ReLU()
            else:
                raise NameError(f"{t} not defined")

        # For the PyTorch model, we don't take into account the `QuantIdentity` layers.
        # Because, it is a Brevitas layer.
        self.features = nn.Sequential(*[make_layers(t) for t in FEATURES_MAPS if t[0] != "I"])

        # The original values in VGG-11 is output_size=(7, 7).
        # We reduced the the kernel size from 7 to 1 to further reduce the image's size.
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # Removing all the classification layers of the original VGG-11 to speed up computation.
        self.final_layer = nn.Linear(in_features=512 * 1 * 1, out_features=output_size)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        # Replace `x.view(x.shape[0], -1)` by `torch.flatten(x, 1)` which is equivalent.
        # But is compatible with Concrete ML.
        x = nn.Flatten()(x)
        x = self.final_layer(x)
        return x


class QuantVGG11(nn.Module):
    def __init__(
        self,
        bit: int,
        output_size: int = 3,
        act_quant: brevitas.quant = Int8ActPerTensorFloat,
        weight_quant: brevitas.quant = Int8WeightPerTensorFloat,
    ):
        """A quantized network with Brevitas.

        Args:
            bit (int): Bit of quantization.
            output_size (int): Number of classes.
            act_quant (brevitas.quant): Quantization protocol of activations.
            weight_quant (brevitas.quant): Quantization protocol of the weights.

        """
        super(QuantVGG11, self).__init__()
        self.bit = bit

        def tuple2quantlayer(t):
            if t[0] == "R":
                return qnn.QuantReLU(return_quant_tensor=True, bit_width=bit, act_quant=act_quant)
            if t[0] == "P":
                return nn.AvgPool2d(kernel_size=t[1], stride=t[2], padding=t[3], ceil_mode=t[5])
            if t[0] == "C":
                return qnn.QuantConv2d(
                    t[1],
                    t[2],
                    kernel_size=t[3],
                    stride=t[4],
                    padding=t[5],
                    weight_bit_width=bit,
                    weight_quant=weight_quant,
                    return_quant_tensor=True,
                )
            if t[0] == "L":
                return qnn.QuantLinear(
                    in_features=t[1],
                    out_features=t[2],
                    weight_bit_width=bit,
                    weight_quant=weight_quant,
                    bias=True,
                    return_quant_tensor=True,
                )
            if t[0] == "I":
                # According to the literature, the first layer holds the most information
                # about the input data. So, it is possible to quantize the input using more
                # precision bit-width than the rest of the network.
                identity_quant = t[1] if len(t) == 2 else bit
                return qnn.QuantIdentity(
                    bit_width=identity_quant, act_quant=act_quant, return_quant_tensor=True
                )

        # The very first layer is a `QuantIdentity` layer, which is very important
        # to ensure that the input data is also quantized.
        self.features = nn.Sequential(*[tuple2quantlayer(t) for t in FEATURES_MAPS])

        # self.identity1 and self.identity2 are used to encapsulate the `torch.flatten`.
        self.identity1 = qnn.QuantIdentity(
            bit_width=bit, act_quant=act_quant, return_quant_tensor=True
        )

        self.identity2 = qnn.QuantIdentity(
            bit_width=bit, act_quant=act_quant, return_quant_tensor=True
        )

        # Fully connected linear layer.
        self.final_layer = qnn.QuantLinear(
            in_features=512 * 1 * 1,
            out_features=output_size,
            weight_quant=weight_quant,
            weight_bit_width=bit,
            bias=True,
            return_quant_tensor=True,
        )

    def forward(self, x):
        x = self.features(x)
        x = self.identity1(x)
        # As `torch.flatten` is a PyTorch layer, you must place it between two `QuantIdentity`
        # layers to ensure that all intermediate values of the network are properly quantized.
        x = torch.flatten(x, 1)
        # Replace `x.view(x.shape[0], -1)` by `torch.flatten(x, 1)` which is an equivalent
        # But is compatible with Concrete ML.
        x = self.identity2(x)
        x = self.final_layer(x)
        return x.value
