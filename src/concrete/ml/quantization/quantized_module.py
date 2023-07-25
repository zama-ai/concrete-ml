"""QuantizedModule API."""
import copy
import re
from functools import partial
from typing import Any, Dict, Generator, Iterable, List, Optional, TextIO, Tuple, Union

import numpy
import onnx
from concrete.fhe.compilation.artifacts import DebugArtifacts
from concrete.fhe.compilation.circuit import Circuit
from concrete.fhe.compilation.compiler import Compiler
from concrete.fhe.compilation.configuration import Configuration

from ..common.debugging import assert_true
from ..common.serialization.dumpers import dump, dumps
from ..common.utils import (
    SUPPORTED_FLOAT_TYPES,
    SUPPORTED_INT_TYPES,
    USE_OLD_VL,
    FheMode,
    all_values_are_floats,
    all_values_are_integers,
    all_values_are_of_dtype,
    check_there_is_no_p_error_options_in_configuration,
    generate_proxy_function,
    manage_parameters_for_pbs_errors,
    set_multi_parameter_in_configuration,
    to_tuple,
)
from .base_quantized_op import ONNXOpInputOutputType, QuantizedOp
from .quantizers import QuantizedArray, UniformQuantizer


def _raise_qat_import_error(bad_qat_ops: List[Tuple[str, str]]):
    """Raise a descriptive error if any invalid ops are present in the ONNX graph.

    Args:
        bad_qat_ops (List[Tuple[str, str]]): list of tensor names and operation types

    Raises:
        ValueError: if there were any invalid, non-quantized, tensors as inputs to non-fusable ops
    """

    raise ValueError(
        "Error occurred during quantization aware training (QAT) import: "
        "The following tensors were expected to be quantized, but the values "
        "found during calibration do not appear to be quantized. \n\n"
        + "\n".join(
            map(
                lambda info: f"* Tensor {info[0]}, input of an {info[1]} operation",
                bad_qat_ops,
            )
        )
        + "\n\nCould not determine a unique scale for the quantization! "
        "Please check the ONNX graph of this model."
    )


def _get_inputset_generator(q_inputs: Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]) -> Generator:
    """Create an input set generator with proper dimensions.

    Args:
        q_inputs (Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]): The quantized inputs.

    Returns:
        Generator: The input set generator with proper dimensions.
    """
    q_inputs = to_tuple(q_inputs)

    assert len(q_inputs) > 0, "The input-set cannot be empty"

    if len(q_inputs) > 1:
        return (
            tuple(numpy.expand_dims(q_input[idx], 0) for q_input in q_inputs)
            for idx in range(q_inputs[0].shape[0])
        )

    # Else, there's only a single input (q_inputs, )
    return (numpy.expand_dims(q_input, 0) for q_input in q_inputs[0])


class QuantizedModule:
    """Inference for a quantized model."""

    ordered_module_input_names: Tuple[str, ...]
    ordered_module_output_names: Tuple[str, ...]
    quant_layers_dict: Dict[str, Tuple[Tuple[str, ...], QuantizedOp]]
    input_quantizers: List[UniformQuantizer]
    output_quantizers: List[UniformQuantizer]
    fhe_circuit: Union[None, Circuit]

    def __init__(
        self,
        ordered_module_input_names: Iterable[str] = None,
        ordered_module_output_names: Iterable[str] = None,
        quant_layers_dict: Dict[str, Tuple[Tuple[str, ...], QuantizedOp]] = None,
        onnx_model: onnx.ModelProto = None,
    ):
        # Set base attributes for API consistency. This could be avoided if an abstract base class
        # is created for both Concrete ML models and QuantizedModule
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2899
        self.fhe_circuit = None
        self._is_compiled = False
        self.input_quantizers = []
        self.output_quantizers = []
        self._onnx_model = onnx_model
        self._post_processing_params: Dict[str, Any] = {}

        # If any of the arguments are not provided, skip the init
        if not all([ordered_module_input_names, ordered_module_output_names, quant_layers_dict]):
            return

        # for mypy
        assert isinstance(ordered_module_input_names, Iterable)
        assert isinstance(ordered_module_output_names, Iterable)
        assert all([ordered_module_input_names, ordered_module_output_names, quant_layers_dict])
        self.ordered_module_input_names = tuple(ordered_module_input_names)
        self.ordered_module_output_names = tuple(ordered_module_output_names)

        num_outputs = len(self.ordered_module_output_names)
        assert_true(
            (num_outputs) == 1,
            f"{QuantizedModule.__class__.__name__} only supports a single output for now, "
            f"got {num_outputs}",
        )

        assert quant_layers_dict is not None
        self.quant_layers_dict = copy.deepcopy(quant_layers_dict)
        self.output_quantizers = self._set_output_quantizers()

    def dump_dict(self) -> Dict:
        """Dump itself to a dict.

        Returns:
            metadata (Dict): Dict of serialized objects.
        """
        metadata: Dict[str, Any] = {}

        metadata["_is_compiled"] = self._is_compiled
        metadata["input_quantizers"] = self.input_quantizers
        metadata["output_quantizers"] = self.output_quantizers
        metadata["_onnx_model"] = self._onnx_model
        metadata["_post_processing_params"] = self._post_processing_params
        metadata["ordered_module_input_names"] = self.ordered_module_input_names
        metadata["ordered_module_output_names"] = self.ordered_module_output_names
        metadata["quant_layers_dict"] = self.quant_layers_dict

        return metadata

    @staticmethod
    def load_dict(metadata: Dict):
        """Load itself from a string.

        Args:
            metadata (Dict): Dict of serialized objects.

        Returns:
            QuantizedModule: The loaded object.
        """

        # Instantiate the module
        obj = QuantizedModule()

        # pylint: disable=protected-access
        obj._is_compiled = metadata["_is_compiled"]
        obj.input_quantizers = metadata["input_quantizers"]
        obj.output_quantizers = metadata["output_quantizers"]
        obj._onnx_model = metadata["_onnx_model"]
        obj.ordered_module_input_names = metadata["ordered_module_input_names"]
        obj.ordered_module_output_names = metadata["ordered_module_output_names"]
        obj.quant_layers_dict = metadata["quant_layers_dict"]
        # pylint: enable=protected-access

        return obj

    def dumps(self) -> str:
        """Dump itself to a string.

        Returns:
            metadata (str): String of the serialized object.
        """
        return dumps(self)

    def dump(self, file: TextIO) -> None:
        """Dump itself to a file.

        Args:
            file (TextIO): The file to dump the serialized object into.
        """
        dump(self, file)

    @property
    def is_compiled(self) -> bool:
        """Indicate if the model is compiled.

        Returns:
            bool: If the model is compiled.
        """
        return self._is_compiled

    def check_model_is_compiled(self):
        """Check if the quantized module is compiled.

        Raises:
            AttributeError: If the quantized module is not compiled.
        """
        if not self.is_compiled:
            raise AttributeError(
                "The quantized module is not compiled. Please run compile(...) first before "
                "executing it in FHE."
            )

    @property
    def post_processing_params(self) -> Dict[str, Any]:
        """Get the post-processing parameters.

        Returns:
            Dict[str, Any]: the post-processing parameters
        """
        return self._post_processing_params

    @post_processing_params.setter
    def post_processing_params(self, post_processing_params: Dict[str, Any]):
        """Set the post-processing parameters.

        Args:
            post_processing_params (dict): the post-processing parameters
        """
        self._post_processing_params = post_processing_params

    # pylint: disable-next=no-self-use
    def post_processing(self, values: numpy.ndarray) -> numpy.ndarray:
        """Apply post-processing to the de-quantized values.

        For quantized modules, there is no post-processing step but the method is kept to make the
        API consistent for the client-server API.

        Args:
            values (numpy.ndarray): The de-quantized values to post-process.

        Returns:
            numpy.ndarray: The post-processed values.
        """
        return values

    def _set_output_quantizers(self) -> List[UniformQuantizer]:
        """Get the output quantizers.

        Returns:
            List[UniformQuantizer]: List of output quantizers.
        """
        output_layers = (
            self.quant_layers_dict[output_name][1]
            for output_name in self.ordered_module_output_names
        )
        output_quantizers = list(
            QuantizedArray(
                output_layer.n_bits,
                values=None,
                value_is_float=False,
                stats=output_layer.output_quant_stats,
                params=output_layer.output_quant_params,
            ).quantizer
            for output_layer in output_layers
        )
        return output_quantizers

    @property
    def onnx_model(self):
        """Get the ONNX model.

        .. # noqa: DAR201

        Returns:
           _onnx_model (onnx.ModelProto): the ONNX model
        """
        return self._onnx_model

    def __call__(self, *x: numpy.ndarray):
        """Define the QuantizedModule's call method.

        This method is a forward method executed in the clear.

        Args:
            *x (numpy.ndarray): Input float values to consider.

        Returns:
            numpy.ndarray: Predictions of the quantized model, in floating points.
        """
        return self.forward(*x)

    def forward(
        self,
        *x: numpy.ndarray,
        fhe: Union[FheMode, str] = FheMode.DISABLE,
        debug: bool = False,
    ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, Optional[Dict[Any, Any]]]]:
        """Forward pass with numpy function only on floating points.

        This method executes the forward pass in the clear, with simulation or in FHE. Input values
        are expected to be floating points, as the method handles the quantization step. The
        returned values are floating points as well.

        Args:
            *x (numpy.ndarray): Input float values to consider.
            fhe (Union[FheMode, str]): The mode to use for prediction. Can be FheMode.DISABLE for
                Concrete ML Python inference, FheMode.SIMULATE for FHE simulation and
                FheMode.EXECUTE for actual FHE execution. Can also be the string representation of
                any of these values. Default to FheMode.DISABLE.
            debug (bool): In debug mode, returns quantized intermediary values of the computation.
                This is useful when a model's intermediary values in Concrete ML need to be
                compared with the intermediary values obtained in pytorch/onnx. When set, the
                second return value is a dictionary containing ONNX operation names as keys and,
                as values, their input QuantizedArray or ndarray. The use can thus extract the
                quantized or float values of quantized inputs. This feature is only available in
                FheMode.DISABLE mode. Default to False.

        Returns:
            numpy.ndarray: Predictions of the quantized model, in floating points.
        """
        assert_true(
            FheMode.is_valid(fhe),
            "`fhe` mode is not supported. Expected one of 'disable' (resp. FheMode.DISABLE), "
            "'simulate' (resp. FheMode.SIMULATE) or 'execute' (resp. FheMode.EXECUTE). Got "
            f"{fhe}",
        )

        # Make sure that the inputs are floating points
        assert_true(
            all_values_are_floats(*x),
            "Input values are expected to be floating points of dtype "
            f"{SUPPORTED_FLOAT_TYPES}. Do not quantize the inputs manually as it is handled "
            "within this method.",
            TypeError,
        )

        # Quantized the input values
        q_x = to_tuple(self.quantize_input(*x))

        if debug and fhe == "disable":
            debug_value_tracker: Optional[
                Dict[str, Dict[Union[int, str], Optional[ONNXOpInputOutputType]]]
            ] = {}
            for (_, layer) in self.quant_layers_dict.values():
                layer.debug_value_tracker = debug_value_tracker
            result = self.quantized_forward(*q_x, fhe="disable")
            for (_, layer) in self.quant_layers_dict.values():
                layer.debug_value_tracker = None
            return result, debug_value_tracker

        q_y_pred = self.quantized_forward(*q_x, fhe=fhe)

        # De-quantize the output predicted values
        y_pred = self.dequantize_output(q_y_pred)

        return y_pred

    def quantized_forward(
        self, *q_x: numpy.ndarray, fhe: Union[FheMode, str] = FheMode.DISABLE
    ) -> numpy.ndarray:
        """Forward function for the FHE circuit.

        Args:
            *q_x (numpy.ndarray): Input integer values to consider.
            fhe (Union[FheMode, str]): The mode to use for prediction. Can be FheMode.DISABLE for
                Concrete ML Python inference, FheMode.SIMULATE for FHE simulation and
                FheMode.EXECUTE for actual FHE execution. Can also be the string representation of
                any of these values. Default to FheMode.DISABLE.

        Returns:
            (numpy.ndarray): Predictions of the quantized model, with integer values.

        """
        # Make sure that the inputs are integers
        assert_true(
            all_values_are_integers(*q_x),
            f"Input values are expected to be integers of dtype {SUPPORTED_INT_TYPES}. "
            "Please quantize the inputs manually as it is not handled within this method.",
            TypeError,
        )

        n_inputs = len(self.input_quantizers)
        n_values = len(q_x)
        assert_true(
            n_values == n_inputs,
            f"Got {n_values} inputs, expected {n_inputs}. Either the quantized module has not been "
            "properly initialized or the input data has been changed since its initialization.",
            ValueError,
        )

        if fhe == "disable":
            return self._clear_forward(*q_x)

        simulate = fhe == "simulate"
        return self._fhe_forward(*q_x, simulate=simulate)

    def _clear_forward(self, *q_x: numpy.ndarray) -> numpy.ndarray:
        """Forward function for the FHE circuit executed in the clear.

        Args:
            *q_x (numpy.ndarray): Input integer values to consider.

        Returns:
            (numpy.ndarray): Predictions of the quantized model, with integer values.

        """

        q_inputs = [
            QuantizedArray(
                self.input_quantizers[idx].n_bits,
                q_x[idx],
                value_is_float=False,
                options=self.input_quantizers[idx].quant_options,
                stats=self.input_quantizers[idx].quant_stats,
                params=self.input_quantizers[idx].quant_params,
            )
            for idx in range(len(self.input_quantizers))
        ]

        # Init layer_results with the inputs
        layer_results: Dict[str, ONNXOpInputOutputType] = dict(
            zip(self.ordered_module_input_names, q_inputs)
        )

        bad_qat_ops: List[Tuple[str, str]] = []
        for output_name, (input_names, layer) in self.quant_layers_dict.items():
            inputs = (layer_results.get(input_name, None) for input_name in input_names)

            error_tracker: List[int] = []
            layer.error_tracker = error_tracker
            output = layer(*inputs)
            layer.error_tracker = None

            if len(error_tracker) > 0:
                # The error message contains the ONNX tensor name that
                # triggered this error
                for input_idx in error_tracker:
                    bad_qat_ops.append((input_names[input_idx], layer.__class__.op_type()))

            layer_results[output_name] = output

        if len(bad_qat_ops) > 0:
            _raise_qat_import_error(bad_qat_ops)

        output_quantized_arrays = tuple(
            layer_results[output_name] for output_name in self.ordered_module_output_names
        )

        assert_true(len(output_quantized_arrays) == 1)

        # The output of a graph must be a QuantizedArray
        assert isinstance(output_quantized_arrays[0], QuantizedArray)

        return output_quantized_arrays[0].qvalues

    def _fhe_forward(self, *q_x: numpy.ndarray, simulate: bool = True) -> numpy.ndarray:
        """Forward function executed in FHE or with simulation.

        Args:
            *q_x (numpy.ndarray): Input integer values to consider.
            simulate (bool): Whether the function should be run in FHE or in simulation mode.
                Default to True.

        Returns:
            (numpy.ndarray): Predictions of the quantized model, with integer values.

        """

        # Make sure that the module is compiled
        assert_true(
            self.fhe_circuit is not None,
            "The quantized module is not compiled. Please run compile(...) first before "
            "executing it in FHE.",
        )

        results_cnp_circuit_list = []
        for i in range(q_x[0].shape[0]):

            # Extract example i from every element in the tuple q_x
            q_input = tuple(q_x[input][[i]] for input in range(len(q_x)))

            # For mypy
            assert self.fhe_circuit is not None

            # If the inference should be executed using simulation
            if simulate:

                # If the old simulation method should be used
                if USE_OLD_VL:
                    predict_method = partial(
                        self.fhe_circuit.graph, p_error=self.fhe_circuit.p_error
                    )

                # Else, use the official simulation method
                else:
                    predict_method = self.fhe_circuit.simulate  # pragma: no cover

            # Else, use the FHE execution method
            else:
                predict_method = self.fhe_circuit.encrypt_run_decrypt

            # Execute the forward pass in FHE or with simulation
            q_result = predict_method(*q_input)

            results_cnp_circuit_list.append(q_result)

        results_cnp_circuit = numpy.concatenate(results_cnp_circuit_list, axis=0)

        return results_cnp_circuit

    def quantize_input(self, *x: numpy.ndarray) -> Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]:
        """Take the inputs in fp32 and quantize it using the learned quantization parameters.

        Args:
            x (numpy.ndarray): Floating point x.

        Returns:
            Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]: Quantized (numpy.int64) x.
        """
        n_inputs = len(self.input_quantizers)
        n_values = len(x)
        assert_true(
            n_values == n_inputs,
            f"Got {n_values} inputs, expected {n_inputs}. Either the quantized module has not been "
            "properly initialized or the input data has been changed since its initialization.",
            ValueError,
        )

        q_x = tuple(self.input_quantizers[idx].quant(x[idx]) for idx in range(len(x)))

        # Make sure all inputs are quantized to int64
        assert all_values_are_of_dtype(*q_x, dtypes="int64"), "Inputs were not quantized to int64"

        return q_x[0] if len(q_x) == 1 else q_x

    def dequantize_output(self, q_y_preds: numpy.ndarray) -> numpy.ndarray:
        """Take the last layer q_out and use its de-quant function.

        Args:
            q_y_preds (numpy.ndarray): Quantized output values of the last layer.

        Returns:
            numpy.ndarray: De-quantized output values of the last layer.
        """
        y_preds = tuple(
            output_quantizer.dequant(q_y_preds) for output_quantizer in self.output_quantizers
        )

        assert_true(len(y_preds) == 1)

        return y_preds[0]

    def set_inputs_quantization_parameters(self, *input_q_params: UniformQuantizer):
        """Set the quantization parameters for the module's inputs.

        Args:
            *input_q_params (UniformQuantizer): The quantizer(s) for the module.
        """
        n_inputs = len(self.ordered_module_input_names)
        n_values = len(input_q_params)
        assert_true(
            n_values == n_inputs,
            f"Got {n_values} inputs, expected {n_inputs}. Either the quantized module has not been "
            "properly initialized or the input data has been changed since its initialization.",
            ValueError,
        )

        self.input_quantizers.clear()
        self.input_quantizers.extend(copy.deepcopy(q_params) for q_params in input_q_params)

    def compile(
        self,
        inputs: Union[Tuple[numpy.ndarray, ...], numpy.ndarray],
        configuration: Optional[Configuration] = None,
        artifacts: Optional[DebugArtifacts] = None,
        show_mlir: bool = False,
        p_error: Optional[float] = None,
        global_p_error: Optional[float] = None,
        verbose: bool = False,
    ) -> Circuit:
        """Compile the module's forward function.

        Args:
            inputs (numpy.ndarray): A representative set of input values used for building
                cryptographic parameters.
            configuration (Optional[Configuration]): Options to use for compilation. Default
                to None.
            artifacts (Optional[DebugArtifacts]): Artifacts information about the
                compilation process to store for debugging.
            show_mlir (bool): Indicate if the MLIR graph should be printed during compilation.
            p_error (Optional[float]): Probability of error of a single PBS. A p_error value cannot
                be given if a global_p_error value is already set. Default to None, which sets this
                error to a default value.
            global_p_error (Optional[float]): Probability of error of the full circuit. A
                global_p_error value cannot be given if a p_error value is already set. This feature
                is not supported during simulation, meaning the probability is
                currently set to 0. Default to None, which sets this
                error to a default value.
            verbose (bool): Indicate if compilation information should be printed
                during compilation. Default to False.

        Returns:
            Circuit: The compiled Circuit.
        """
        inputs = to_tuple(inputs)

        ref_len = inputs[0].shape[0]
        assert_true(
            all(input.shape[0] == ref_len for input in inputs),
            "Mismatched dataset lengths",
        )

        assert not numpy.any([numpy.issubdtype(input.dtype, numpy.integer) for input in inputs]), (
            "Inputs used for compiling a QuantizedModule should only be floating points and not"
            "already-quantized values."
        )

        # Concrete does not support variable *args-style functions, so compile a proxy
        # function dynamically with a suitable number of arguments
        forward_proxy, orig_args_to_proxy_func_args = generate_proxy_function(
            self._clear_forward, self.ordered_module_input_names
        )

        compiler = Compiler(
            forward_proxy,
            {arg_name: "encrypted" for arg_name in orig_args_to_proxy_func_args.values()},
        )

        # Quantize the inputs
        q_inputs = self.quantize_input(*inputs)

        # Generate the input-set with proper dimensions
        inputset = _get_inputset_generator(q_inputs)

        # Check that p_error or global_p_error is not set in both the configuration and in the
        # direct parameters
        check_there_is_no_p_error_options_in_configuration(configuration)

        # Find the right way to set parameters for compiler, depending on the way we want to default
        p_error, global_p_error = manage_parameters_for_pbs_errors(p_error, global_p_error)

        # Remove this function once the default strategy is set to multi-parameter in Concrete
        # Python
        # TODO: https://github.com/zama-ai/concrete-ml-internal/issues/3860
        configuration = set_multi_parameter_in_configuration(configuration)

        # Jit compiler is now deprecated and will soon be removed, it is thus forced to False
        # by default
        self.fhe_circuit = compiler.compile(
            inputset,
            configuration=configuration,
            artifacts=artifacts,
            show_mlir=show_mlir,
            p_error=p_error,
            global_p_error=global_p_error,
            verbose=verbose,
            single_precision=False,
            fhe_simulation=False,
            fhe_execution=True,
            jit=False,
        )

        # CRT simulation is not supported yet
        # TODO: https://github.com/zama-ai/concrete-ml-internal/issues/3841
        if not USE_OLD_VL:
            self.fhe_circuit.enable_fhe_simulation()  # pragma: no cover

        self._is_compiled = True

        return self.fhe_circuit

    def bitwidth_and_range_report(
        self,
    ) -> Optional[Dict[str, Dict[str, Union[Tuple[int, ...], int]]]]:
        """Report the ranges and bit-widths for layers that mix encrypted integer values.

        Returns:
            op_names_to_report (Dict): a dictionary with operation names as keys. For each
                operation, (e.g., conv/gemm/add/avgpool ops), a range and a bit-width are returned.
                The range contains the min/max values encountered when computing the operation and
                the bit-width gives the number of bits needed to represent this range.
        """

        if self.fhe_circuit is None:
            return None

        op_names_to_report: Dict[str, Dict[str, Union[Tuple[int, ...], int]]] = {}
        for (_, op_inst) in self.quant_layers_dict.values():
            # Get the value range of this tag and all its subtags
            # The potential tags for this op start with the op instance name
            # and are, sometimes, followed by a subtag starting with a period:
            # ex: abc, abc.cde, abc.cde.fgh
            # so first craft a regex to match all such tags.
            pattern = re.compile(re.escape(op_inst.op_instance_name) + "(\\..*)?")
            value_range = self.fhe_circuit.graph.integer_range(pattern)
            bitwidth = self.fhe_circuit.graph.maximum_integer_bit_width(pattern)

            # Only store the range and bit-width if there are valid ones,
            # as some ops (fusable ones) do not have tags
            if value_range is not None and bitwidth >= 0:
                op_names_to_report[op_inst.op_instance_name] = {
                    "range": value_range,
                    "bitwidth": bitwidth,
                }

        return op_names_to_report
