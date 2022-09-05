"""torch compilation function."""

import tempfile
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy
import onnx
import torch
from brevitas.export import BrevitasONNXManager
from concrete.numpy import MAXIMUM_TLU_BIT_WIDTH
from concrete.numpy.compilation.artifacts import DebugArtifacts
from concrete.numpy.compilation.configuration import Configuration

from ..common.debugging import assert_true
from ..common.utils import DEFAULT_P_ERROR_PBS, get_onnx_opset_version
from ..onnx.convert import OPSET_VERSION_FOR_ONNX_EXPORT
from ..quantization import PostTrainingAffineQuantization, PostTrainingQATImporter, QuantizedModule
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
    import_qat: bool = False,
    configuration: Optional[Configuration] = None,
    compilation_artifacts: Optional[DebugArtifacts] = None,
    show_mlir: bool = False,
    n_bits=MAXIMUM_TLU_BIT_WIDTH,
    use_virtual_lib: bool = False,
    p_error: Optional[float] = DEFAULT_P_ERROR_PBS,
) -> QuantizedModule:
    """Compile a torch module or ONNX into an FHE equivalent.

    Take a model in torch or ONNX, turn it to numpy, quantize its inputs / weights / outputs and
    finally compile it with Concrete-Numpy

    Args:
        model (Union[torch.nn.Module, onnx.ModelProto]): the model to quantize, either in torch or
            in ONNX
        torch_inputset (Dataset): the inputset, can contain either torch
            tensors or numpy.ndarray, only datasets with a single input are supported for now.
        import_qat (bool): Flag to signal that the network being imported contains quantizers in
            in its computation graph and that Concrete ML should not requantize it.
        configuration (Configuration): Configuration object to use
            during compilation
        compilation_artifacts (DebugArtifacts): Artifacts object to fill
            during compilation
        show_mlir (bool): if set, the MLIR produced by the converter and which is going
            to be sent to the compiler backend is shown on the screen, e.g., for debugging or demo
        n_bits: the number of bits for the quantization
        use_virtual_lib (bool): set to use the so called virtual lib simulating FHE computation.
            Defaults to False
        p_error (Optional[float]): probability of error of a PBS

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
    onnx_model = numpy_model.onnx_model

    # Quantize with post-training static method, to have a model with integer weights
    post_training_quant: Union[PostTrainingAffineQuantization, PostTrainingQATImporter]
    if import_qat:
        post_training_quant = PostTrainingQATImporter(n_bits, numpy_model, is_signed=True)
    else:
        post_training_quant = PostTrainingAffineQuantization(n_bits, numpy_model, is_signed=True)
    quantized_module = post_training_quant.quantize_module(*inputset_as_numpy_tuple)

    quantized_numpy_inputset = quantized_module.quantize_input(*inputset_as_numpy_tuple)

    quantized_module.compile(
        quantized_numpy_inputset,
        configuration,
        compilation_artifacts,
        show_mlir=show_mlir,
        use_virtual_lib=use_virtual_lib,
        p_error=p_error,
    )

    quantized_module.onnx_model = onnx_model

    return quantized_module


def compile_torch_model(
    torch_model: torch.nn.Module,
    torch_inputset: Dataset,
    import_qat: bool = False,
    configuration: Optional[Configuration] = None,
    compilation_artifacts: Optional[DebugArtifacts] = None,
    show_mlir: bool = False,
    n_bits=MAXIMUM_TLU_BIT_WIDTH,
    use_virtual_lib: bool = False,
    p_error: Optional[float] = DEFAULT_P_ERROR_PBS,
) -> QuantizedModule:
    """Compile a torch module into an FHE equivalent.

    Take a model in torch, turn it to numpy, quantize its inputs / weights / outputs and finally
    compile it with Concrete-Numpy

    Args:
        torch_model (torch.nn.Module): the model to quantize
        torch_inputset (Dataset): the inputset, can contain either torch
            tensors or numpy.ndarray, only datasets with a single input are supported for now.
        import_qat (bool): Set to True to import a network that contains quantizers and was
            trained using quantization aware training
        configuration (Configuration): Configuration object to use
            during compilation
        compilation_artifacts (DebugArtifacts): Artifacts object to fill
            during compilation
        show_mlir (bool): if set, the MLIR produced by the converter and which is going
            to be sent to the compiler backend is shown on the screen, e.g., for debugging or demo
        n_bits: the number of bits for the quantization
        use_virtual_lib (bool): set to use the so called virtual lib simulating FHE computation.
            Defaults to False
        p_error (Optional[float]): probability of error of a PBS

    Returns:
        QuantizedModule: The resulting compiled QuantizedModule.
    """
    return _compile_torch_or_onnx_model(
        torch_model,
        torch_inputset,
        import_qat,
        configuration=configuration,
        compilation_artifacts=compilation_artifacts,
        show_mlir=show_mlir,
        n_bits=n_bits,
        use_virtual_lib=use_virtual_lib,
        p_error=p_error,
    )


def compile_onnx_model(
    onnx_model: onnx.ModelProto,
    torch_inputset: Dataset,
    import_qat: bool = False,
    configuration: Optional[Configuration] = None,
    compilation_artifacts: Optional[DebugArtifacts] = None,
    show_mlir: bool = False,
    n_bits=MAXIMUM_TLU_BIT_WIDTH,
    use_virtual_lib: bool = False,
    p_error: Optional[float] = DEFAULT_P_ERROR_PBS,
) -> QuantizedModule:
    """Compile a torch module into an FHE equivalent.

    Take a model in torch, turn it to numpy, quantize its inputs / weights / outputs and finally
    compile it with Concrete-Numpy

    Args:
        onnx_model (onnx.ModelProto): the model to quantize
        torch_inputset (Dataset): the inputset, can contain either torch
            tensors or numpy.ndarray, only datasets with a single input are supported for now.
        import_qat (bool): Flag to signal that the network being imported contains quantizers in
            in its computation graph and that Concrete ML should not requantize it.
        configuration (Configuration): Configuration object to use
            during compilation
        compilation_artifacts (DebugArtifacts): Artifacts object to fill
            during compilation
        show_mlir (bool): if set, the MLIR produced by the converter and which is going
            to be sent to the compiler backend is shown on the screen, e.g., for debugging or demo
        n_bits: the number of bits for the quantization
        use_virtual_lib (bool): set to use the so called virtual lib simulating FHE computation.
            Defaults to False.
        p_error (Optional[float]): probability of error of a PBS

    Returns:
        QuantizedModule: The resulting compiled QuantizedModule.
    """

    onnx_model_opset_version = get_onnx_opset_version(onnx_model)
    assert_true(
        onnx_model_opset_version == OPSET_VERSION_FOR_ONNX_EXPORT,
        f"ONNX version must be {OPSET_VERSION_FOR_ONNX_EXPORT} "
        f"but it is {onnx_model_opset_version}",
    )

    return _compile_torch_or_onnx_model(
        onnx_model,
        torch_inputset,
        import_qat,
        configuration=configuration,
        compilation_artifacts=compilation_artifacts,
        show_mlir=show_mlir,
        n_bits=n_bits,
        use_virtual_lib=use_virtual_lib,
        p_error=p_error,
    )


def compile_brevitas_qat_model(
    torch_model: torch.nn.Module,
    torch_inputset: Dataset,
    n_bits: Union[int, dict],
    configuration: Optional[Configuration] = None,
    compilation_artifacts: Optional[DebugArtifacts] = None,
    show_mlir: bool = False,
    use_virtual_lib: bool = False,
    p_error: Optional[float] = DEFAULT_P_ERROR_PBS,
    output_onnx_file: Optional[str] = None,
) -> QuantizedModule:
    """Compile a Brevitas Quantization Aware Training model.

    The torch_model parameter is a subclass of torch.nn.Module that uses quantized
    operations from brevitas.qnn. The model is trained before calling this function. This
    function compiles the trained model to FHE.

    Args:
        torch_model (torch.nn.Module): the model to quantize
        torch_inputset (Dataset): the inputset, can contain either torch
            tensors or numpy.ndarray, only datasets with a single input are supported for now.
        n_bits (Union[int,dict]): the number of bits for the quantization
        configuration (Configuration): Configuration object to use
            during compilation
        compilation_artifacts (DebugArtifacts): Artifacts object to fill
            during compilation
        show_mlir (bool): if set, the MLIR produced by the converter and which is going
            to be sent to the compiler backend is shown on the screen, e.g., for debugging or demo
        use_virtual_lib (bool): set to use the so called virtual lib simulating FHE computation,
            defaults to False.
        p_error (Optional[float]): probability of error of a PBS
        output_onnx_file (str): temporary file to store ONNX model. If None a temporary file
            is generated

    Returns:
        QuantizedModule: The resulting compiled QuantizedModule.
    """

    inputset_as_numpy_tuple = (
        tuple(convert_torch_tensor_or_numpy_array_to_numpy_array(val) for val in torch_inputset)
        if isinstance(torch_inputset, tuple)
        else (convert_torch_tensor_or_numpy_array_to_numpy_array(torch_inputset),)
    )

    dummy_input_for_tracing = tuple(
        torch.from_numpy(val[[0], ::]).float() for val in inputset_as_numpy_tuple
    )

    output_onnx_file_path = Path(
        tempfile.mkstemp(suffix=".onnx")[1]
        if (use_tempfile := (output_onnx_file is None))
        else output_onnx_file
    )

    # Brevitas to ONNX
    onnx_model = BrevitasONNXManager.export(
        torch_model,
        input_shape=dummy_input_for_tracing[0].shape,
        export_path=str(output_onnx_file_path),
        keep_initializers_as_inputs=False,
        opset_version=OPSET_VERSION_FOR_ONNX_EXPORT,
    )

    # Compile using the ONNX conversion flow, in QAT mode
    q_module_vl = compile_onnx_model(
        onnx_model,
        torch_inputset,
        n_bits=n_bits,
        import_qat=True,
        compilation_artifacts=compilation_artifacts,
        show_mlir=show_mlir,
        use_virtual_lib=use_virtual_lib,
        configuration=configuration,
        p_error=p_error,
    )

    # Remove the tempfile if we used one
    if use_tempfile:
        output_onnx_file_path.unlink(missing_ok=True)

    return q_module_vl
