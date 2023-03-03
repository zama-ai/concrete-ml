import torch
from brevitas.nn import QuantIdentity
from brevitas_utils import CommonActQuant
from constants import CNV_OUT_CH_POOL, KERNEL_SIZE, SPLIT_INDEX
from torch.nn import AvgPool2d, BatchNorm2d, Conv2d


# First layers of the model
# They will in-fine run in clear on the client side
# It can use float or quantization
class ClearModule(torch.nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_bit_width: int,
    ):
        super().__init__()

        self.in_ch = in_ch
        out_ch = in_ch

        self.conv_features = torch.nn.ModuleList()
        for out_ch, is_pool_enabled in CNV_OUT_CH_POOL[:SPLIT_INDEX]:
            self.conv_features.append(
                Conv2d(
                    kernel_size=KERNEL_SIZE,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    bias=True,
                )
            )
            in_ch = out_ch
            self.conv_features.append(BatchNorm2d(in_ch, eps=1e-4))
            if is_pool_enabled:
                self.conv_features.append(AvgPool2d(kernel_size=2))
        self.out_ch = out_ch

        self.conv_features.append(
            QuantIdentity(
                act_quant=CommonActQuant,
                return_quant_tensor=False,
                bit_width=out_bit_width,
                min_val=-1.0,
                max_val=1.0 - 2.0 ** (-7),
                narrow_range=True,
            )
        )

    def forward(self, x):
        for mod in self.conv_features:
            x = mod(x)
        return x
