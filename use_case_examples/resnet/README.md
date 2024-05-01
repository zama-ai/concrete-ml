# ResNet in FHE

## Overview

`resnet.py` is taken from torchvision https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py.

The main modification is the replacement of the adaptive average pooling layer with a standard average pooling layer.

```diff
-        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
+        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
```

Concrete ML does not support `AdaptiveAvgPool2d` yet.

`resnet_fhe.ipynb` is a notebook that demonstrates how to use Concrete ML to compile and run a ResNet model in FHE along with some figures to show the accuracy vs different bit-width.

