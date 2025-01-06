"""QuantizedModule API."""

import copy
import os
import re
from functools import partial
from typing import Any, Dict, Generator, Iterable, List, Optional, Sequence, TextIO, Tuple, Union

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
    check_compilation_device_is_valid_and_is_cuda,
    check_there_is_no_p_error_options_in_configuration,
    generate_proxy_function,
    manage_parameters_for_pbs_errors,
    to_tuple,
)
from ..torch.numpy_module import NumpyModule
from .base_quantized_op import ONNXOpInputOutputType, QuantizedOp
from .quantized_ops import QuantizedReduceSum
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
                lambda info: f"* Tensor {info[0]}, input of a {info[1]} operation",
                bad_qat_ops,
            )
        )
        + "\n\nAre you missing a QuantIdentity layer in your Brevitas model? "
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


# pylint: disable=too-many-instance-attributes
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
        ordered_module_input_names: Optional[Iterable[str]] = None,
        ordered_module_output_names: Optional[Iterable[str]] = None,
        quant_layers_dict: Optional[Dict[str, Tuple[Tuple[str, ...], QuantizedOp]]] = None,
        onnx_model: Optional[onnx.ModelProto] = None,
        onnx_preprocessing: Optional[onnx.ModelProto] = None,
    ):

        all_or_none_params = [
            ordered_module_input_names,
            ordered_module_output_names,
            quant_layers_dict,
        ]
        if not (
            all(v is None or v == {} for v in all_or_none_params)
            or not any(v is None or v == {} for v in all_or_none_params)
        ):
            raise ValueError(
                "Please either set all three 'ordered_module_input_names', "
                "'ordered_module_output_names' and 'quant_layers_dict' or none of them."
            )

        self.ordered_module_input_names = (
            tuple(ordered_module_input_names) if ordered_module_input_names else ()
        )
        self.ordered_module_output_names = (
            tuple(ordered_module_output_names) if ordered_module_output_names else ()
        )
        self.quant_layers_dict = (
            copy.deepcopy(quant_layers_dict) if quant_layers_dict is not None else {}
        )

        # Set base attributes for API consistency. This could be avoided if an abstract base class
        # is created for both Concrete ML models and QuantizedModule
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/2899
        self.input_quantizers: List[UniformQuantizer] = []
        self.output_quantizers: List[UniformQuantizer] = []
        self.fhe_circuit: Optional[Circuit] = None
        self._is_compiled = False
        self._compiled_for_cuda = False
        self._onnx_model = onnx_model
        self._post_processing_params: Dict[str, Any] = {}

        # Initialize output quantizers based on quant_layers_dict
        if self.quant_layers_dict:
            self.output_quantizers = self._set_output_quantizers()
        else:
            self.output_quantizers = []

        # Input-output quantizer mapping for composition is not enabled at initialization
        self._composition_mapping: Optional[Dict] = None

        # Initialize _preprocessing_module
        # The onnx graph is used to pre-process the inputs before FHE execution
        self._preprocessing_module = NumpyModule(onnx_preprocessing) if onnx_preprocessing else None

        # Ensure that there is no preprocessing step
        if self._preprocessing_module is not None:
            assert_true(self._preprocessing_module.onnx_preprocessing is None)

    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4127
    def set_reduce_sum_copy(self):
        """Set reduce sum to copy or not the inputs.

        Due to bit-width propagation in the compilation we need, in some situations,
        to copy the inputs with a PBS to avoid it.
        """
        assert self.quant_layers_dict is not None
        for _, quantized_op in self.quant_layers_dict.values():
            if isinstance(quantized_op, QuantizedReduceSum):
                quantized_op.copy_inputs = True

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
        metadata["onnx_preprocessing"] = (
            self._preprocessing_module.onnx_model if self._preprocessing_module else None
        )

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
        obj._preprocessing_module = (
            NumpyModule(metadata["onnx_preprocessing"]) if metadata["onnx_preprocessing"] else None
        )

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

    def post_processing(
        self, *values: numpy.ndarray
    ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]:
        """Apply post-processing to the de-quantized values.

        For quantized modules, there is no post-processing step but the method is kept to make the
        API consistent for the client-server API.

        Args:
            values (numpy.ndarray): The de-quantized values to post-process.

        Returns:
            Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]: The post-processed values.
        """
        return values[0] if len(values) == 1 else values

    def pre_processing(self, *values: numpy.ndarray) -> Tuple[numpy.ndarray, ...]:
        """Apply pre-processing to the input values.

        Args:
            values (numpy.ndarray): The input values to pre-process.

        Returns:
            Tuple[numpy.ndarray, ...]: The pre-processed values.
        """

        if self._preprocessing_module is not None:
            return to_tuple(self._preprocessing_module(*values))

        return values

    def _set_output_quantizers(self) -> List[UniformQuantizer]:
        """Get the output quantizers.

        Returns:
            List[UniformQuantizer]: List of output quantizers.
        """
        output_layers = list(
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

    # Remove this once we handle the re-quantization step in post-training only
    # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4472
    def _add_requant_for_composition(self, composition_mapping: Optional[Dict]):
        """Trigger a re-quantization step for outputs using an input-output mapping for quantizers.

        Args:
            composition_mapping (Optional[Dict]): Dictionary that maps output positions with input
                positions in the case of composable circuits. Setting this parameter triggers a
                re-quantization step at the end of the FHE circuit. This makes sure outputs are
                de-quantized using their output quantizer and then re-quantized using their
                associated input quantizer. Default to None.

        Raises:
            ValueError: If the mapping is not properly constructed: it must be a dictionary of
                positive integers, mapping output positions to input positions, where positions
                must not be greater than the model's number of outputs/inputs.
        """
        if not isinstance(composition_mapping, Dict):
            raise ValueError(
                "Parameter 'composition_mapping' mus be a dictionary. Got "
                f"{type(composition_mapping)}"
            )

        max_output_pos = len(self.output_quantizers) - 1
        max_input_pos = len(self.input_quantizers) - 1

        for output_position, input_position in composition_mapping.items():
            if not isinstance(output_position, int) or output_position < 0:
                raise ValueError(
                    "Output positions (keys) must be positive integers. Got "
                    f"{type(output_position)}"
                )

            if output_position > max_output_pos:
                raise ValueError(
                    "Output positions (keys) must not be greater than the model's number of "
                    f"outputs. Expected position '{max_output_pos}' at most, but got "
                    f"'{output_position}'"
                )

            if not isinstance(input_position, int) or input_position < 0:
                raise ValueError(
                    "Input positions (values) must be positive integers. Got "
                    f"{type(input_position)}"
                )

            if input_position > max_input_pos:
                raise ValueError(
                    "Input positions (values) must not be greater than the model's number of "
                    f"inputs. Expected position '{max_input_pos}' at most, but got "
                    f"'{input_position}'"
                )

        self._composition_mapping = composition_mapping

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
    ) -> Union[
        numpy.ndarray,
        Tuple[numpy.ndarray, ...],
        Tuple[
            Union[Tuple[numpy.ndarray, ...], numpy.ndarray],
            Dict[str, Dict[Union[int, str], ONNXOpInputOutputType]],
        ],
    ]:
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

        # Pre-process input
        x = self.pre_processing(*x)

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
            debug_value_tracker: Dict[
                str, Dict[Union[int, str], Optional[ONNXOpInputOutputType]]
            ] = {}
            for _, layer in self.quant_layers_dict.values():
                layer.debug_value_tracker = debug_value_tracker
            q_y_pred = self.quantized_forward(*q_x, fhe="disable")
            for _, layer in self.quant_layers_dict.values():
                layer.debug_value_tracker = None
            # De-quantize the output predicted values
            y_pred = self.dequantize_output(*to_tuple(q_y_pred))
            return y_pred, debug_value_tracker

        q_y_pred = self.quantized_forward(*q_x, fhe=fhe)

        # De-quantize the output predicted values
        y_pred = self.dequantize_output(*to_tuple(q_y_pred))
        return y_pred

    def quantized_forward(
        self, *q_x: numpy.ndarray, fhe: Union[FheMode, str] = FheMode.DISABLE
    ) -> Union[Tuple[numpy.ndarray, ...], numpy.ndarray]:
        """Forward function for the FHE circuit.

        Args:
            *q_x (numpy.ndarray): Input integer values to consider.
            fhe (Union[FheMode, str]): The mode to use for prediction. Can be FheMode.DISABLE for
                Concrete ML Python inference, FheMode.SIMULATE for FHE simulation and
                FheMode.EXECUTE for actual FHE execution. Can also be the string representation of
                any of these values. Default to FheMode.DISABLE.

        Returns:
            (Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]): Predictions of the quantized model,
            with integer values.

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

    def _clear_forward(
        self, *q_x: numpy.ndarray
    ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]:
        """Forward function for the FHE circuit executed in the clear.

        Args:
            *q_x (numpy.ndarray): Input integer values to consider.

        Returns:
            (Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]): Predictions of the quantized model,
                with integer values.

        Raises:
            ValueError: If composition is enabled and that mapped input-output shapes do not match.
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
                    bad_qat_ops.append((input_names[input_idx], str(layer.__class__.op_type())))

            layer_results[output_name] = output

        if len(bad_qat_ops) > 0:
            _raise_qat_import_error(bad_qat_ops)

        output_quantized_arrays = tuple(
            layer_results[output_name] for output_name in self.ordered_module_output_names
        )

        # The output of a graph must be a QuantizedArray
        assert all(isinstance(elt, QuantizedArray) for elt in output_quantized_arrays)

        q_results = tuple(
            elt.qvalues for elt in output_quantized_arrays if isinstance(elt, QuantizedArray)
        )

        # Remove this once we handle the re-quantization step in post-training only
        # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4472
        if self._composition_mapping is not None:
            mismatch_shapes = list(
                f"Output {output_i}: {q_results[output_i].shape} "
                f"-> Input {input_i}: {q_x[input_i].shape}"
                for output_i, input_i in self._composition_mapping.items()
            )

            if not all(
                q_x[input_i].shape == q_results[output_i].shape
                for output_i, input_i in self._composition_mapping.items()
            ):
                raise ValueError(
                    "A shape mismatch has been found between inputs and outputs when composing the "
                    "forward pass. Please check the given composition mapping. Got "
                    f"{self._composition_mapping}, which gives the following shape mapping:\n"
                    + "\n".join(mismatch_shapes)
                )

            # Only add a re-quantization step to outputs that appear in the composition mapping.
            # This is because some outputs might not be used as inputs when composing a circuit
            q_results = tuple(
                (
                    self.input_quantizers[self._composition_mapping[i]].quant(
                        self.output_quantizers[i].dequant(q_result)
                    )
                    if i in self._composition_mapping
                    else q_result
                )
                for i, q_result in enumerate(q_results)
            )

        # Check that the number of outputs properly matches the number of output quantizers. This is
        # to make sure that no processing like, for example, composition mapping has altered the
        # number of outputs
        assert len(q_results) == len(self.output_quantizers), (
            "The number of outputs does not match the number of output quantizers. Got "
            f"{len(q_results)=} != {len(self.output_quantizers)=} "
        )

        if len(q_results) == 1:
            return q_results[0]

        return q_results

    def _fhe_forward(
        self, *q_x: numpy.ndarray, simulate: bool = True
    ) -> Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]:
        """Forward function executed in FHE or with simulation.

        Args:
            *q_x (numpy.ndarray): Input integer values to consider.
            simulate (bool): Whether the function should be run in FHE or in simulation mode.
                Default to True.

        Returns:
            (Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]): Predictions of the quantized model,
                with integer values.

        """

        # Make sure that the module is compiled
        assert_true(
            self.fhe_circuit is not None,
            "The quantized module is not compiled. Please run compile(...) first before "
            "executing it in FHE.",
        )
        q_result_by_output: List[List[numpy.ndarray]] = [[] for _ in self.output_quantizers]
        for i in range(q_x[0].shape[0]):

            # Extract example i from every element in the tuple q_x
            q_input = tuple(q_x[input][[i]] for input in range(len(q_x)))

            # For mypy
            assert self.fhe_circuit is not None

            # If the inference should be executed using simulation
            if simulate:
                is_crt_encoding = self.fhe_circuit.statistics["packing_key_switch_count"] != 0

                # If the virtual library method should be used
                # For now, use the virtual library when simulating
                # circuits that use CRT  encoding because the official simulation is too slow
                # FIXME: https://github.com/zama-ai/concrete-ml-internal/issues/4391
                if USE_OLD_VL or is_crt_encoding:
                    predict_method = partial(
                        self.fhe_circuit.graph, p_error=self.fhe_circuit.p_error
                    )  # pragma: no cover

                # Else, use the official simulation method
                else:
                    predict_method = self.fhe_circuit.simulate

            # Else, use the FHE execution method
            else:
                predict_method = self.fhe_circuit.encrypt_run_decrypt

            # Execute the forward pass in FHE or with simulation
            q_result = to_tuple(predict_method(*q_input))

            assert len(q_result) == len(q_result_by_output), (
                "Number of outputs does not match the number of output quantizers.\n"
                f"{len(q_result)=}!={len(self.output_quantizers)=}"
            )
            for elt_index, elt in enumerate(q_result):
                q_result_by_output[elt_index].append(elt)

        q_results: Tuple[numpy.ndarray, ...] = tuple(
            numpy.concatenate(elt, axis=0) for elt in q_result_by_output
        )
        if len(q_results) == 1:
            return q_results[0]
        return q_results

    def quantize_input(
        self, *x: Optional[numpy.ndarray], dtype: numpy.typing.DTypeLike = numpy.int64
    ) -> Union[numpy.ndarray, Tuple[Optional[numpy.ndarray], ...]]:
        """Take the inputs in fp32 and quantize it using the learned quantization parameters.

        Args:
            x (Optional[numpy.ndarray]): Floating point x or None.
            dtype (numpy.typing.DTypeLike): optional user-specified datatype for the output


        Returns:
            Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]: Quantized (numpy.int64) x, or None if
                the corresponding input is None.
        """
        n_inputs = len(self.input_quantizers)
        n_values = len(x)

        assert_true(
            n_values == n_inputs,
            f"Got {n_values} inputs, expected {n_inputs}. Either the quantized module has not been "
            "properly initialized or the input data has been changed since its initialization.",
            ValueError,
        )

        assert not all(x_i is None for x_i in x), "Please provide at least one input to quantize."

        # Ignore [arg-type] check from mypy as it is not able to see that the input to `quant`
        # cannot be None
        q_x = tuple(
            (
                self.input_quantizers[idx].quant(x[idx], dtype)  # type: ignore[arg-type]
                if x[idx] is not None
                else None
            )
            for idx in range(len(x))
        )

        # Make sure all inputs are quantized to int64
        assert all_values_are_of_dtype(
            *q_x, dtypes=numpy.dtype(dtype).name, allow_none=True
        ), "Inputs were not quantized to int64"

        if len(q_x) == 1:
            assert q_x[0] is not None

            return q_x[0]

        return q_x

    def dequantize_output(
        self, *q_y_preds: numpy.ndarray
    ) -> Union[numpy.ndarray, Tuple[Union[numpy.ndarray], ...]]:
        """Take the last layer q_out and use its de-quant function.

        Args:
            q_y_preds (numpy.ndarray): Quantized outputs values.

        Returns:
            Union[numpy.ndarray, Tuple[numpy.ndarray, ...]]: De-quantized output values of
                the last layer.
        """
        # Make sure that we have as many predictions as quantizers
        assert len(q_y_preds) == len(self.output_quantizers), (
            f"{len(q_y_preds)=} != {len(self.output_quantizers)=} "
            "but the number of outputs to de-quantize should match the number of output quantizers."
        )

        y_preds = tuple(
            numpy.array(output_quantizer.dequant(q_y_pred))
            for q_y_pred, output_quantizer in zip(q_y_preds, self.output_quantizers)
        )

        if len(y_preds) == 1:
            return y_preds[0]

        return y_preds

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
        inputs_encryption_status: Optional[Sequence[str]] = None,
        device: str = "cpu",
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
            inputs_encryption_status (Optional[Sequence[str]]): encryption status ('clear',
                'encrypted') for each input.
            device: FHE compilation device, can be either 'cpu' or 'cuda'.

        Returns:
            Circuit: The compiled Circuit.

        Raises:
            ValueError: if inputs_encryption_status does not match with the
                parameters of the quantized module
        """
        inputs = to_tuple(inputs)

        # Apply pre_processing
        inputs = self.pre_processing(*inputs)

        ref_len = inputs[0].shape[0]
        assert_true(
            all(input.shape[0] == ref_len for input in inputs),
            "Mismatched dataset lengths",
        )

        assert not numpy.any(
            numpy.array([numpy.issubdtype(input.dtype, numpy.integer) for input in inputs])
        ), (
            "Inputs used for compiling a QuantizedModule should only be floating points and not "
            "already-quantized values."
        )

        # Concrete does not support variable *args-style functions, so compile a proxy
        # function dynamically with a suitable number of arguments
        forward_proxy, orig_args_to_proxy_func_args = generate_proxy_function(
            self._clear_forward, self.ordered_module_input_names
        )

        if inputs_encryption_status is None:
            inputs_encryption_status = tuple(
                "encrypted" for _ in orig_args_to_proxy_func_args.values()
            )
        else:
            if len(inputs_encryption_status) < len(orig_args_to_proxy_func_args.values()):
                raise ValueError(
                    f"Missing arguments from '{inputs_encryption_status}', expected "
                    f"{len(orig_args_to_proxy_func_args.values())} arguments."
                )
            if len(inputs_encryption_status) > len(orig_args_to_proxy_func_args.values()):
                raise ValueError(
                    f"Too many arguments in '{inputs_encryption_status}', expected "
                    f"{len(orig_args_to_proxy_func_args.values())} arguments."
                )
            if not all(value in {"clear", "encrypted"} for value in inputs_encryption_status):
                raise ValueError(
                    f"Unexpected status from '{inputs_encryption_status}',"
                    " expected 'clear' or 'encrypted'."
                )
            if not any(value == "encrypted" for value in inputs_encryption_status):
                raise ValueError(
                    f"At least one input should be encrypted but got {inputs_encryption_status}"
                )

        assert inputs_encryption_status is not None  # For mypy
        inputs_encryption_status_dict = dict(
            zip(orig_args_to_proxy_func_args.values(), inputs_encryption_status)
        )

        compiler = Compiler(
            forward_proxy,
            parameter_encryption_statuses=inputs_encryption_status_dict,
        )

        # Quantize the inputs
        q_inputs = self.quantize_input(*inputs)

        # Make sure all inputs are quantized to int64 and are not None
        assert all_values_are_of_dtype(
            *to_tuple(q_inputs), dtypes="int64", allow_none=False
        ), "Inputs were not quantized to int64"

        # Generate the input-set with proper dimensions
        # Ignore [arg-type] check from mypy as it is not able to see that no values in `q_inputs`
        # is None
        inputset = _get_inputset_generator(q_inputs)  # type: ignore[arg-type]

        # Check that p_error or global_p_error is not set in both the configuration and in the
        # direct parameters
        check_there_is_no_p_error_options_in_configuration(configuration)

        # Find the right way to set parameters for compiler, depending on the way we want to default
        p_error, global_p_error = manage_parameters_for_pbs_errors(p_error, global_p_error)

        use_gpu = check_compilation_device_is_valid_and_is_cuda(device)

        # Enable input ciphertext compression
        enable_input_compression = os.environ.get("USE_INPUT_COMPRESSION", "1") == "1"
        enable_key_compression = os.environ.get("USE_KEY_COMPRESSION", "1") == "1"

        self.fhe_circuit = compiler.compile(
            inputset,
            configuration=configuration,
            artifacts=artifacts,
            show_mlir=show_mlir,
            p_error=p_error,
            global_p_error=global_p_error,
            verbose=verbose,
            single_precision=False,
            use_gpu=use_gpu,
            compress_input_ciphertexts=enable_input_compression,
            compress_evaluation_keys=enable_key_compression,
        )

        self._is_compiled = True
        self._compiled_for_cuda = use_gpu

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
        for _, op_inst in self.quant_layers_dict.values():
            # Get the value range of this tag and all its subtags
            # The potential tags for this op start with the op instance name
            # and are, sometimes, followed by a subtag starting with a period:
            # ex: abc, abc.cde, abc.cde.fgh
            # so first craft a regex to match all such tags.
            pattern = re.compile(re.escape(op_inst.op_instance_name) + "(\\..*)?")
            value_range = self.fhe_circuit.graph.integer_range(pattern)
            bitwidth = self.fhe_circuit.graph.maximum_integer_bit_width(
                pattern,
            )

            # Only store the range and bit-width if there are valid ones,
            # as some ops (fusable ones) do not have tags
            if value_range is not None and bitwidth >= 0:
                op_names_to_report[op_inst.op_instance_name] = {
                    "range": value_range,
                    "bitwidth": bitwidth,
                }

        return op_names_to_report
