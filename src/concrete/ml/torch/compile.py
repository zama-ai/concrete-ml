"""torch compilation function."""

from typing import Optional, Tuple, Union

import numpy
import torch
from concrete.common.compilation import CompilationArtifacts, CompilationConfiguration

from ..quantization import PostTrainingAffineQuantization, QuantizedArray, QuantizedModule
from . import NumpyModule

Tensor = Union[torch.Tensor, numpy.ndarray]
Dataset = Union[Tensor, Tuple[Tensor, ...]]


def convert_torch_tensor_or_numpy_array_to_numpy_array(
    torch_tensor_or_numpy_array: Tensor,
) -> numpy.ndarray:
    """Convert a torch tensor or a numpy array to a numpy array.

    Args:
        torch_tensor_or_numpy_array (Tensor): the value that is either
            a torch tensor or a numpy array.

    Returns:
        numpy.ndarray: the value converted to a numpy array.
    """
    return (
        torch_tensor_or_numpy_array
        if isinstance(torch_tensor_or_numpy_array, numpy.ndarray)
        else torch_tensor_or_numpy_array.cpu().numpy()
    )


def compile_torch_model(
    torch_model: torch.nn.Module,
    torch_inputset: Dataset,
    compilation_configuration: Optional[CompilationConfiguration] = None,
    compilation_artifacts: Optional[CompilationArtifacts] = None,
    show_mlir: bool = False,
    n_bits=7,
    use_virtual_lib: bool = False,
) -> QuantizedModule:
    """Take a model in torch, turn it to numpy, transform weights to integer.

    Later, we'll compile the integer model.

    Args:
        torch_model (torch.nn.Module): the model to quantize,
        torch_inputset (Dataset): the inputset, can contain either torch
            tensors or numpy.ndarray, only datasets with a single input are supported for now.
        compilation_configuration (CompilationConfiguration): Configuration object to use
            during compilation
        compilation_artifacts (CompilationArtifacts): Artifacts object to fill
            during compilation
        show_mlir (bool): if set, the MLIR produced by the converter and which is going
            to be sent to the compiler backend is shown on the screen, e.g., for debugging or demo
        n_bits: the number of bits for the quantization
        use_virtual_lib (bool): set to use the so called virtual lib simulating FHE computation.
            Defaults to False.


    Returns:
        QuantizedModule: The resulting compiled QuantizedModule.
    """

    inputset_as_numpy_tuple = (
        tuple(convert_torch_tensor_or_numpy_array_to_numpy_array(val) for val in torch_inputset)
        if isinstance(torch_inputset, tuple)
        else (convert_torch_tensor_or_numpy_array_to_numpy_array(torch_inputset),)
    )

    dummy_input_for_tracing = tuple(
        torch.from_numpy(val).float() for val in inputset_as_numpy_tuple
    )

    # Create corresponding numpy model
    numpy_model = NumpyModule(torch_model, dummy_input_for_tracing)

    # Quantize with post-training static method, to have a model with integer weights
    post_training_quant = PostTrainingAffineQuantization(n_bits, numpy_model, is_signed=True)
    quantized_module = post_training_quant.quantize_module(*inputset_as_numpy_tuple)

    # Quantize input
    quantized_numpy_inputset = tuple(QuantizedArray(n_bits, val) for val in inputset_as_numpy_tuple)

    quantized_module.compile(
        quantized_numpy_inputset,
        compilation_configuration,
        compilation_artifacts,
        show_mlir,
        use_virtual_lib,
    )

    return quantized_module
