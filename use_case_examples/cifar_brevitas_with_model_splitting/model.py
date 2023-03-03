# https://github.com/Xilinx/brevitas/blob/8c3d9de0113528cf6693c6474a13d802a66682c6/src/brevitas_examples/bnn_pynq/models/CNV.py
import torch
from clear_module import ClearModule
from encrypted_module import EncryptedModule

CNV_OUT_CH_POOL = [
    (64, False),
    (64, True),
    (128, False),
    (128, True),
    (256, False),
    (256, False),
]

INTERMEDIATE_FC_FEATURES = [(256, 512), (512, 512)]
LAST_FC_IN_FEATURES = 512
LAST_FC_PER_OUT_CH_SCALING = False
POOL_SIZE = 2
KERNEL_SIZE = 3
EPSILON_VALUE = 0.5
SPLIT_INDEX = 1


# The model combining both modules
class CNV(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        weight_bit_width: int,
        act_bit_width: int,
        in_bit_width: int,
        in_ch: int,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.weight_bit_width = weight_bit_width
        self.act_bit_width = act_bit_width
        self.in_bit_width = in_bit_width
        self.in_ch = in_ch

        self.clear_module = ClearModule(in_ch=in_ch, out_bit_width=in_bit_width)
        self.encrypted_module = EncryptedModule(
            num_classes=num_classes,
            weight_bit_width=weight_bit_width,
            act_bit_width=act_bit_width,
            in_bit_width=in_bit_width,
            in_ch=self.clear_module.out_ch,
        )

    def forward(self, x):
        x = self.clear_module(x)
        return self.encrypted_module(x)
