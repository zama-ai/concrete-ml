"""torch compilation function."""

import tempfile
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy
import onnx
import torch
from brevitas.export.onnx.qonnx.manager import QONNXManager as BrevitasONNXManager
from brevitas.nn.quant_layer import QuantInputOutputLayer as QNNMixingLayer
from brevitas.nn.quant_layer import QuantNonLinearActLayer as QNNUnivariateLayer
from concrete.fhe import ParameterSelectionStrategy
from concrete.fhe.compilation.artifacts import DebugArtifacts
from concrete.fhe.compilation.configuration import Configuration

from ..common.debugging import assert_false, assert_true
from ..common.utils import (
    MAX_BITWIDTH_BACKWARD_COMPATIBLE,
    check_there_is_no_p_error_options_in_configuration,
    get_onnx_opset_version,
    manage_parameters_for_pbs_errors,
    to_tuple,
)
from ..onnx.convert import OPSET_VERSION_FOR_ONNX_EXPORT
from ..onnx.onnx_utils import remove_initializer_from_input
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


def build_quantized_module(
    model: Union[torch.nn.Module, onnx.ModelProto],
    torch_inputset: Dataset,
    import_qat: bool = False,
    n_bits=MAX_BITWIDTH_BACKWARD_COMPATIBLE,
    rounding_threshold_bits: Optional[int] = None,
) -> QuantizedModule:
    """Build a quantized module from a Torch or ONNX model.

    Take a model in torch or ONNX, turn it to numpy, quantize its inputs / weights / outputs and
    retrieve the associated quantized module.

    Args:
        model (Union[torch.nn.Module, onnx.ModelProto]): The model to quantize, either in torch or
            in ONNX.
        torch_inputset (Dataset): the calibration input-set, can contain either torch
            tensors or numpy.ndarray
        import_qat (bool): Flag to signal that the network being imported contains quantizers in
            in its computation graph and that Concrete ML should not re-quantize it
        n_bits: the number of bits for the quantization
        rounding_threshold_bits (int): if not None, every accumulators in the model are rounded down
            to the given bits of precision

    Returns:
        QuantizedModule: The resulting QuantizedModule.
    """
    inputset_as_numpy_tuple = tuple(
        convert_torch_tensor_or_numpy_array_to_numpy_array(val) for val in to_tuple(torch_inputset)
    )

    # Tracing needs to be done with the batch size of 1 since we compile our models to FHE with
    # this batch size. The input set contains many examples, to determine a representative
    # bit-width, but for tracing we only take a single one. We need the ONNX tracing batch size to
    # match the batch size during FHE inference which can only be 1 for the moment.
    dummy_input_for_tracing = tuple(
        torch.from_numpy(val[[0], ::]).float() for val in inputset_as_numpy_tuple
    )

    # Create corresponding numpy model
    numpy_model = NumpyModule(model, dummy_input_for_tracing)

    # Quantize with post-training static method, to have a model with integer weights
    post_training = PostTrainingQATImporter if import_qat else PostTrainingAffineQuantization
    post_training_quant = post_training(n_bits, numpy_model, rounding_threshold_bits)

    # Build the quantized module
    # TODO: mismatch here. We traced with dummy_input_for_tracing which made some operator
    # only work over shape of (1, ., .). For example, some reshape have newshape hardcoded based
    # on the inputset we sent in the NumpyModule.
    quantized_module = post_training_quant.quantize_module(*inputset_as_numpy_tuple)

    return quantized_module


# pylint: disable-next=too-many-arguments
def _compile_torch_or_onnx_model(
    model: Union[torch.nn.Module, onnx.ModelProto],
    torch_inputset: Dataset,
    import_qat: bool = False,
    configuration: Optional[Configuration] = None,
    artifacts: Optional[DebugArtifacts] = None,
    show_mlir: bool = False,
    n_bits=MAX_BITWIDTH_BACKWARD_COMPATIBLE,
    rounding_threshold_bits: Optional[int] = None,
    p_error: Optional[float] = None,
    global_p_error: Optional[float] = None,
    verbose: bool = False,
) -> QuantizedModule:
    """Compile a torch module or ONNX into an FHE equivalent.

    Take a model in torch or ONNX, turn it to numpy, quantize its inputs / weights / outputs and
    finally compile it with Concrete

    Args:
        model (Union[torch.nn.Module, onnx.ModelProto]): the model to quantize, either in torch or
            in ONNX
        torch_inputset (Dataset): the calibration input-set, can contain either torch
            tensors or numpy.ndarray
        import_qat (bool): Flag to signal that the network being imported contains quantizers in
            in its computation graph and that Concrete ML should not re-quantize it
        configuration (Configuration): Configuration object to use during compilation
        artifacts (DebugArtifacts): Artifacts object to fill during compilation
        show_mlir (bool): if set, the MLIR produced by the converter and which is going
            to be sent to the compiler backend is shown on the screen, e.g., for debugging or demo
        n_bits: the number of bits for the quantization
        rounding_threshold_bits (int): if not None, every accumulators in the model are rounded down
            to the given bits of precision
        p_error (Optional[float]): probability of error of a single PBS
        global_p_error (Optional[float]): probability of error of the full circuit. In FHE
            simulation `global_p_error` is set to 0
        verbose (bool): whether to show compilation information

    Returns:
        QuantizedModule: The resulting compiled QuantizedModule.
    """

    inputset_as_numpy_tuple = tuple(
        convert_torch_tensor_or_numpy_array_to_numpy_array(val) for val in to_tuple(torch_inputset)
    )

    # Build the quantized module
    quantized_module = build_quantized_module(
        model=model,
        torch_inputset=inputset_as_numpy_tuple,
        import_qat=import_qat,
        n_bits=n_bits,
        rounding_threshold_bits=rounding_threshold_bits,
    )

    # Check that p_error or global_p_error is not set in both the configuration and in the direct
    # parameters
    check_there_is_no_p_error_options_in_configuration(configuration)

    if (
        rounding_threshold_bits is not None
        and configuration is not None
        and configuration.parameter_selection_strategy != ParameterSelectionStrategy.MULTI
    ):
        warnings.warn(
            "It is recommended to set the optimization strategy to multi-parameter when using "
            "rounding as it should provide better performance.",
            stacklevel=2,
        )

    # Find the right way to set parameters for compiler, depending on the way we want to default
    p_error, global_p_error = manage_parameters_for_pbs_errors(p_error, global_p_error)

    quantized_module.compile(
        inputset_as_numpy_tuple,
        configuration,
        artifacts,
        show_mlir=show_mlir,
        p_error=p_error,
        global_p_error=global_p_error,
        verbose=verbose,
    )

    return quantized_module


# pylint: disable-next=too-many-arguments
def compile_torch_model(
    torch_model: torch.nn.Module,
    torch_inputset: Dataset,
    import_qat: bool = False,
    configuration: Optional[Configuration] = None,
    artifacts: Optional[DebugArtifacts] = None,
    show_mlir: bool = False,
    n_bits=MAX_BITWIDTH_BACKWARD_COMPATIBLE,
    rounding_threshold_bits: Optional[int] = None,
    p_error: Optional[float] = None,
    global_p_error: Optional[float] = None,
    verbose: bool = False,
) -> QuantizedModule:
    """Compile a torch module into an FHE equivalent.

    Take a model in torch, turn it to numpy, quantize its inputs / weights / outputs and finally
    compile it with Concrete

    Args:
        torch_model (torch.nn.Module): the model to quantize
        torch_inputset (Dataset): the calibration input-set, can contain either torch
            tensors or numpy.ndarray.
        import_qat (bool): Set to True to import a network that contains quantizers and was
            trained using quantization aware training
        configuration (Configuration): Configuration object to use
            during compilation
        artifacts (DebugArtifacts): Artifacts object to fill
            during compilation
        show_mlir (bool): if set, the MLIR produced by the converter and which is going
            to be sent to the compiler backend is shown on the screen, e.g., for debugging or demo
        n_bits: the number of bits for the quantization
        rounding_threshold_bits (int): if not None, every accumulators in the model are rounded down
            to the given bits of precision
        p_error (Optional[float]): probability of error of a single PBS
        global_p_error (Optional[float]): probability of error of the full circuit. In FHE
            simulation `global_p_error` is set to 0
        verbose (bool): whether to show compilation information

    Returns:
        QuantizedModule: The resulting compiled QuantizedModule.
    """
    assert_true(
        isinstance(torch_model, torch.nn.Module),
        "The compile_torch_model function must be called on a torch.nn.Module",
    )

    has_any_qnn_layers = any(
        isinstance(layer, (QNNMixingLayer, QNNUnivariateLayer)) for layer in torch_model.modules()
    )

    assert_false(
        has_any_qnn_layers,
        "The compile_torch_model was called on a torch.nn.Module that contains "
        "Brevitas quantized layers. These models must be imported "
        "using compile_brevitas_qat_model instead.",
    )

    return _compile_torch_or_onnx_model(
        torch_model,
        torch_inputset,
        import_qat,
        configuration=configuration,
        artifacts=artifacts,
        show_mlir=show_mlir,
        n_bits=n_bits,
        rounding_threshold_bits=rounding_threshold_bits,
        p_error=p_error,
        global_p_error=global_p_error,
        verbose=verbose,
    )


# pylint: disable-next=too-many-arguments
def compile_onnx_model(
    onnx_model: onnx.ModelProto,
    torch_inputset: Dataset,
    import_qat: bool = False,
    configuration: Optional[Configuration] = None,
    artifacts: Optional[DebugArtifacts] = None,
    show_mlir: bool = False,
    n_bits=MAX_BITWIDTH_BACKWARD_COMPATIBLE,
    rounding_threshold_bits: Optional[int] = None,
    p_error: Optional[float] = None,
    global_p_error: Optional[float] = None,
    verbose: bool = False,
) -> QuantizedModule:
    """Compile a torch module into an FHE equivalent.

    Take a model in torch, turn it to numpy, quantize its inputs / weights / outputs and finally
    compile it with Concrete-Python

    Args:
        onnx_model (onnx.ModelProto): the model to quantize
        torch_inputset (Dataset): the calibration input-set, can contain either torch
            tensors or numpy.ndarray.
        import_qat (bool): Flag to signal that the network being imported contains quantizers in
            in its computation graph and that Concrete ML should not re-quantize it.
        configuration (Configuration): Configuration object to use
            during compilation
        artifacts (DebugArtifacts): Artifacts object to fill
            during compilation
        show_mlir (bool): if set, the MLIR produced by the converter and which is going
            to be sent to the compiler backend is shown on the screen, e.g., for debugging or demo
        n_bits: the number of bits for the quantization
        rounding_threshold_bits (int): if not None, every accumulators in the model are rounded down
            to the given bits of precision
        p_error (Optional[float]): probability of error of a single PBS
        global_p_error (Optional[float]): probability of error of the full circuit. In FHE
            simulation `global_p_error` is set to 0
        verbose (bool): whether to show compilation information

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
        artifacts=artifacts,
        show_mlir=show_mlir,
        n_bits=n_bits,
        rounding_threshold_bits=rounding_threshold_bits,
        p_error=p_error,
        global_p_error=global_p_error,
        verbose=verbose,
    )


# pylint: disable-next=too-many-arguments
def compile_brevitas_qat_model(
    torch_model: torch.nn.Module,
    torch_inputset: Dataset,
    n_bits: Optional[Union[int, dict]] = None,
    configuration: Optional[Configuration] = None,
    artifacts: Optional[DebugArtifacts] = None,
    show_mlir: bool = False,
    rounding_threshold_bits: Optional[int] = None,
    p_error: Optional[float] = None,
    global_p_error: Optional[float] = None,
    output_onnx_file: Union[Path, str] = None,
    verbose: bool = False,
) -> QuantizedModule:
    """Compile a Brevitas Quantization Aware Training model.

    The torch_model parameter is a subclass of torch.nn.Module that uses quantized
    operations from brevitas.qnn. The model is trained before calling this function. This
    function compiles the trained model to FHE.

    Args:
        torch_model (torch.nn.Module): the model to quantize
        torch_inputset (Dataset): the calibration input-set, can contain either torch
            tensors or numpy.ndarray.
        n_bits (Optional[Union[int, dict]): the number of bits for the quantization. By default,
            for most models, a value of None should be given, which instructs Concrete ML to use the
            bit-widths configured using Brevitas quantization options. For some networks, that
            perform a non-linear operation on an input on an output, if None is given, a default
            value of 8 bits is used for the input/output quantization. For such models the user can
            also specify a dictionary with model_inputs/model_outputs keys to override
            the 8-bit default or a single integer for both values.
        configuration (Configuration): Configuration object to use
            during compilation
        artifacts (DebugArtifacts): Artifacts object to fill
            during compilation
        show_mlir (bool): if set, the MLIR produced by the converter and which is going
            to be sent to the compiler backend is shown on the screen, e.g., for debugging or demo
        rounding_threshold_bits (int): if not None, every accumulators in the model are rounded down
            to the given bits of precision
        p_error (Optional[float]): probability of error of a single PBS
        global_p_error (Optional[float]): probability of error of the full circuit. In FHE
            simulation `global_p_error` is set to 0
        output_onnx_file (str): temporary file to store ONNX model. If None a temporary file
            is generated
        verbose (bool): whether to show compilation information

    Returns:
        QuantizedModule: The resulting compiled QuantizedModule.
    """

    inputset_as_numpy_tuple = tuple(
        convert_torch_tensor_or_numpy_array_to_numpy_array(val) for val in to_tuple(torch_inputset)
    )

    dummy_input_for_tracing = tuple(
        torch.from_numpy(val[[0], ::]).float() for val in inputset_as_numpy_tuple
    )

    output_onnx_file_path = Path(
        tempfile.mkstemp(suffix=".onnx")[1] if output_onnx_file is None else output_onnx_file
    )

    use_tempfile: bool = output_onnx_file is None

    assert_true(
        isinstance(torch_model, torch.nn.Module),
        "The compile_brevitas_qat_model function must be called on a torch.nn.Module",
    )

    has_any_qnn_layers = any(
        isinstance(layer, (QNNMixingLayer, QNNUnivariateLayer)) for layer in torch_model.modules()
    )

    assert_true(
        has_any_qnn_layers,
        "The compile_brevitas_qat_model was called on a torch.nn.Module that contains "
        "no Brevitas quantized layers, consider using compile_torch_model instead",
    )

    # Brevitas to ONNX
    exporter = BrevitasONNXManager()
    # Here we add a "eliminate_nop_pad" optimization step for onnxoptimizer
    # https://github.com/onnx/optimizer/blob/master/onnxoptimizer/passes/eliminate_nop_pad.h#L5
    # It deletes 0-values padding.
    # This is needed because AvgPool2d adds a 0-Pad operation that then breaks the compilation
    # A list of steps that can be added can be found in the following link
    # https://github.com/onnx/optimizer/blob/master/onnxoptimizer/pass_registry.h
    # In the export function, the `args` parameter is used instead of the `input_shape` one in
    # order to be able to handle multi-inputs models
    exporter.onnx_passes.append("eliminate_nop_pad")
    exporter.onnx_passes.append("fuse_pad_into_conv")
    onnx_model = exporter.export(
        torch_model,
        args=dummy_input_for_tracing,
        export_path=str(output_onnx_file_path),
        keep_initializers_as_inputs=False,
        opset_version=OPSET_VERSION_FOR_ONNX_EXPORT,
    )
    onnx_model = remove_initializer_from_input(onnx_model)

    if n_bits is None:
        n_bits = {
            "model_inputs": 8,
            "op_weights": 8,
            "op_inputs": 8,
            "model_outputs": 8,
        }
    elif isinstance(n_bits, int):
        n_bits = {
            "model_inputs": n_bits,
            "op_weights": n_bits,
            "op_inputs": n_bits,
            "model_outputs": n_bits,
        }
    elif isinstance(n_bits, dict):
        assert_true(
            set(n_bits.keys()) == {"model_inputs", "model_outputs"},
            "When importing a Brevitas QAT network, n_bits should only contain the following keys: "
            f'"model_inputs", "model_outputs". Instead, got {n_bits.keys()}',
        )

        n_bits = {
            "model_inputs": n_bits["model_inputs"],
            "op_weights": n_bits["model_inputs"],
            "op_inputs": n_bits["model_inputs"],
            "model_outputs": n_bits["model_outputs"],
        }
    assert_true(
        n_bits is None or isinstance(n_bits, (int, dict)),
        "The n_bits parameter must be either a dictionary, an integer or None",
    )

    # Compile using the ONNX conversion flow, in QAT mode
    q_module = compile_onnx_model(
        onnx_model,
        torch_inputset,
        n_bits=n_bits,
        import_qat=True,
        artifacts=artifacts,
        show_mlir=show_mlir,
        rounding_threshold_bits=rounding_threshold_bits,
        configuration=configuration,
        p_error=p_error,
        global_p_error=global_p_error,
        verbose=verbose,
    )

    # Remove the tempfile if we used one
    if use_tempfile:
        output_onnx_file_path.unlink()

    return q_module
