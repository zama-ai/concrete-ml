"""torch compilation function."""

from typing import Optional, Tuple, Union

import numpy
import onnx
import torch
from concrete.numpy import MAXIMUM_BIT_WIDTH
from concrete.numpy.compilation.artifacts import DebugArtifacts
from concrete.numpy.compilation.configuration import Configuration

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


def _compile_torch_or_onnx_model(
    model: Union[torch.nn.Module, onnx.ModelProto],
    torch_inputset: Dataset,
    configuration: Optional[Configuration] = None,
    compilation_artifacts: Optional[DebugArtifacts] = None,
    show_mlir: bool = False,
    n_bits=MAXIMUM_BIT_WIDTH,
    use_virtual_lib: bool = False,
) -> QuantizedModule:
    """Compile a torch module or ONNX into an FHE equivalent.

    Take a model in torch or ONNX, turn it to numpy, quantize its inputs / weights / outputs and
    finally compile it with Concrete-Numpy

    Args:
        model (Union[torch.nn.Module, onnx.ModelProto]): the model to quantize, either in torch or
            in ONNX
        torch_inputset (Dataset): the inputset, can contain either torch
            tensors or numpy.ndarray, only datasets with a single input are supported for now.
        configuration (Configuration): Configuration object to use
            during compilation
        compilation_artifacts (DebugArtifacts): Artifacts object to fill
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

    # Tracing needs to be done with the batch size of 1 since we compile our models to FHE with
    # this batch size. The input set contains many examples, to determine a representative bitwidth,
    # but for tracing we only take a single one. We need the ONNX tracing batch size to match
    # the batch size during FHE inference which can only be 1 for the moment.
    # FIXME: if it's possible to use batch size > 1 in FHE, update this function
    # see https://github.com/zama-ai/concrete-ml-internal/issues/758
    dummy_input_for_tracing = tuple(
        torch.from_numpy(val[[0], ::]).float() for val in inputset_as_numpy_tuple
    )

    # Create corresponding numpy model
    numpy_model = NumpyModule(model, dummy_input_for_tracing)

    # Quantize with post-training static method, to have a model with integer weights
    post_training_quant = PostTrainingAffineQuantization(n_bits, numpy_model, is_signed=True)
    quantized_module = post_training_quant.quantize_module(*inputset_as_numpy_tuple)

    # Quantize input
    quantized_numpy_inputset = tuple(QuantizedArray(n_bits, val) for val in inputset_as_numpy_tuple)

    quantized_module.compile(
        quantized_numpy_inputset,
        configuration,
        compilation_artifacts,
        show_mlir=show_mlir,
        use_virtual_lib=use_virtual_lib,
    )

    return quantized_module


def compile_torch_model(
    torch_model: torch.nn.Module,
    torch_inputset: Dataset,
    configuration: Optional[Configuration] = None,
    compilation_artifacts: Optional[DebugArtifacts] = None,
    show_mlir: bool = False,
    n_bits=MAXIMUM_BIT_WIDTH,
    use_virtual_lib: bool = False,
) -> QuantizedModule:
    """Compile a torch module into an FHE equivalent.

    Take a model in torch, turn it to numpy, quantize its inputs / weights / outputs and finally
    compile it with Concrete-Numpy

    Args:
        torch_model (torch.nn.Module): the model to quantize
        torch_inputset (Dataset): the inputset, can contain either torch
            tensors or numpy.ndarray, only datasets with a single input are supported for now.
        configuration (Configuration): Configuration object to use
            during compilation
        compilation_artifacts (DebugArtifacts): Artifacts object to fill
            during compilation
        show_mlir (bool): if set, the MLIR produced by the converter and which is going
            to be sent to the compiler backend is shown on the screen, e.g., for debugging or demo
        n_bits: the number of bits for the quantization
        use_virtual_lib (bool): set to use the so called virtual lib simulating FHE computation.
            Defaults to False.

    Returns:
        QuantizedModule: The resulting compiled QuantizedModule.
    """
    return _compile_torch_or_onnx_model(
        torch_model,
        torch_inputset,
        configuration=configuration,
        compilation_artifacts=compilation_artifacts,
        show_mlir=show_mlir,
        n_bits=n_bits,
        use_virtual_lib=use_virtual_lib,
    )


def compile_onnx_model(
    onnx_model: onnx.ModelProto,
    torch_inputset: Dataset,
    configuration: Optional[Configuration] = None,
    compilation_artifacts: Optional[DebugArtifacts] = None,
    show_mlir: bool = False,
    n_bits=MAXIMUM_BIT_WIDTH,
    use_virtual_lib: bool = False,
) -> QuantizedModule:
    """Compile a torch module into an FHE equivalent.

    Take a model in torch, turn it to numpy, quantize its inputs / weights / outputs and finally
    compile it with Concrete-Numpy

    Args:
        onnx_model (onnx.ModelProto): the model to quantize
        torch_inputset (Dataset): the inputset, can contain either torch
            tensors or numpy.ndarray, only datasets with a single input are supported for now.
        configuration (Configuration): Configuration object to use
            during compilation
        compilation_artifacts (DebugArtifacts): Artifacts object to fill
            during compilation
        show_mlir (bool): if set, the MLIR produced by the converter and which is going
            to be sent to the compiler backend is shown on the screen, e.g., for debugging or demo
        n_bits: the number of bits for the quantization
        use_virtual_lib (bool): set to use the so called virtual lib simulating FHE computation.
            Defaults to False.

    Returns:
        QuantizedModule: The resulting compiled QuantizedModule.
    """
    return _compile_torch_or_onnx_model(
        onnx_model,
        torch_inputset,
        configuration=configuration,
        compilation_artifacts=compilation_artifacts,
        show_mlir=show_mlir,
        n_bits=n_bits,
        use_virtual_lib=use_virtual_lib,
    )
