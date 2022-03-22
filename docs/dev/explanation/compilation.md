```{note}
FIXME: Arthur to do
```

# Compilation Pipeline In Depth

## Overview of the torch compilation process

Compiling a torch Module is pretty straightforward.

The torch Module is first converted to a Numpy equivalent we call `NumpyModule` if all the layers in the torch Module are supported.

Then the module is quantized post-training to be compatible with our compiler which only works on integers. The post training quantization uses the provided dataset for calibration.

The dataset is then quantized to be usable for compilation with the QuantizedModule.

The QuantizedModule is compiled yielding an executable FHECircuit.

Here is the visual representation of the different steps:

![Torch compilation flow](../../_static/compilation-pipeline/torch_to_numpy_flow.svg)
