"""Post Training Quantization methods."""

from abc import abstractmethod
from typing import Dict, List, Optional, Set, Tuple, Type, Union, cast

import numpy
import onnx
from onnx import numpy_helper

from ..common.debugging import assert_true
from ..onnx.onnx_utils import ONNX_OPS_TO_NUMPY_IMPL, get_attribute, get_op_type
from ..onnx.ops_impl import RawOpOutput
from ..torch.numpy_module import NumpyModule
from .base_quantized_op import (
    DEFAULT_MODEL_BITS,
    ONNX_OPS_TO_QUANTIZED_IMPL,
    ONNXOpInputOutputType,
    QuantizedMixingOp,
    QuantizedOp,
)
from .quantized_module import QuantizedModule
from .quantized_ops import QuantizedBrevitasQuant
from .quantizers import QuantizationOptions, QuantizedArray, UniformQuantizer


def get_n_bits_dict(n_bits: Union[int, Dict[str, int]]) -> Dict[str, int]:
    """Convert the n_bits parameter into a proper dictionary.

    Args:
        n_bits (int, Dict[str, int]): number of bits for quantization, can be a single value or
            a dictionary with the following keys :
            - "op_inputs" and "op_weights" (mandatory)
            - "model_inputs" and "model_outputs" (optional, default to 5 bits).
            When using a single integer for n_bits, its value is assigned to "op_inputs" and
            "op_weights" bits. The maximum between this value and a default value (5) is then
            assigned to the number of "model_inputs" "model_outputs". This default value is a
            compromise between model accuracy and runtime performance in FHE. "model_outputs" gives
            the precision of the final network's outputs, while "model_inputs" gives the precision
            of the network's inputs. "op_inputs" and "op_weights" both control the quantization for
            inputs and weights of all layers.

    Returns:
        n_bits_dict (Dict[str, int]): A dictionary properly representing the number of bits to use
            for quantization.
    """

    assert_true(
        isinstance(n_bits, int)
        or (
            isinstance(n_bits, Dict)
            and set(n_bits.keys()).issubset(
                {"model_inputs", "op_weights", "model_outputs", "op_inputs"}
            )
            and {"op_weights", "op_inputs"}.issubset(set(n_bits.keys()))
        ),
        "Invalid n_bits, either pass an integer or a dictionary containing integer values for "
        "the following keys:\n"
        "- `op_weights` and `op_inputs` (mandatory)\n"
        f"- `model_outputs` and `model_inputs` (optional, default to {DEFAULT_MODEL_BITS} "
        "bits)",
    )

    # If a single integer is passed, we use a default value for the model's input and
    # output bits
    if isinstance(n_bits, int):
        n_bits_dict = {
            "model_inputs": max(DEFAULT_MODEL_BITS, n_bits),
            "op_weights": n_bits,
            "op_inputs": n_bits,
            "model_outputs": max(DEFAULT_MODEL_BITS, n_bits),
        }

    # If model_inputs or model_outputs are not given, we consider a default value
    elif isinstance(n_bits, Dict):
        n_bits_dict = {
            "model_inputs": DEFAULT_MODEL_BITS,
            "model_outputs": max(DEFAULT_MODEL_BITS, n_bits["op_inputs"]),
        }

        n_bits_dict.update(n_bits)

    assert_true(
        n_bits_dict["model_outputs"] >= n_bits_dict["op_inputs"],
        "Using fewer bits to represent the model_outputs than the op inputs is not "
        f"recommended. Got model_outputs: {n_bits_dict['model_outputs']} and op_inputs: "
        f"{n_bits_dict['op_inputs']}",
    )

    return n_bits_dict


class ONNXConverter:
    """Base ONNX to Concrete ML computation graph conversion class.

    This class provides a method to parse an ONNX graph and apply several transformations. First,
    it creates QuantizedOps for each ONNX graph op. These quantized ops have calibrated
    quantizers that are useful when the operators work on integer data or when the output of
    the ops is the output of the encrypted program. For operators that compute in float and will
    be merged to TLUs, these quantizers are not used. Second, this converter creates quantized
    tensors for initializer and weights stored in the graph.

    This class should be sub-classed to provide specific calibration and quantization options
    depending on the usage (Post-training quantization vs Quantization Aware training).

    Arguments:
        n_bits (int, Dict[str, int]): number of bits for quantization, can be a single value or
            a dictionary with the following keys :
            - "op_inputs" and "op_weights" (mandatory)
            - "model_inputs" and "model_outputs" (optional, default to 5 bits).
            When using a single integer for n_bits, its value is assigned to "op_inputs" and
            "op_weights" bits. The maximum between this value and a default value (5) is then
            assigned to the number of "model_inputs" "model_outputs". This default value is a
            compromise between model accuracy and runtime performance in FHE. "model_outputs" gives
            the precision of the final network's outputs, while "model_inputs" gives the precision
            of the network's inputs. "op_inputs" and "op_weights" both control the quantization for
            inputs and weights of all layers.
        numpy_model (NumpyModule): Model in numpy.
        rounding_threshold_bits (int): if not None, every accumulators in the model are rounded down
            to the given bits of precision
    """

    quant_ops_dict: Dict[str, Tuple[Tuple[str, ...], QuantizedOp]]
    n_bits: Dict[str, int]
    quant_params: Dict[str, numpy.ndarray]
    numpy_model: NumpyModule
    rounding_threshold_bits: Optional[int]

    def __init__(
        self,
        n_bits: Union[int, Dict],
        numpy_model: NumpyModule,
        rounding_threshold_bits: Optional[int] = None,
    ):
        self.quant_ops_dict = {}

        self.n_bits = get_n_bits_dict(n_bits)
        self.quant_params = {}
        self.numpy_model = numpy_model
        self.rounding_threshold_bits = rounding_threshold_bits

    @property
    def n_bits_model_outputs(self):
        """Get the number of bits to use for the quantization of the last layer's output.

        Returns:
            n_bits (int): number of bits for output quantization
        """
        return self.n_bits["model_outputs"]

    @property
    def n_bits_model_inputs(self):
        """Get the number of bits to use for the quantization of the first layer's output.

        Returns:
            n_bits (int): number of bits for input quantization
        """
        return self.n_bits["model_inputs"]

    @property
    def n_bits_op_weights(self):
        """Get the number of bits to use for the quantization of any constants (usually weights).

        Returns:
            n_bits (int): number of bits for quantizing constants used by operators
        """
        return self.n_bits["op_weights"]

    @property
    def n_bits_op_inputs(self):
        """Get the number of bits to use for the quantization of any operators' inputs.

        Returns:
            n_bits (int): number of bits for the quantization of the operators' inputs
        """
        return self.n_bits["op_inputs"]

    @abstractmethod
    def _process_layer(
        self,
        quantized_op: QuantizedOp,
        *calibration_data: numpy.ndarray,
        quantizers: List[Optional[UniformQuantizer]],
    ) -> Tuple[numpy.ndarray, Optional[UniformQuantizer]]:
        """Configure a graph operation according to model conversion mode.

        Args:
            quantized_op (QuantizedOp): Quantized graph operator instance
            *calibration_data (numpy.ndarray): tuple of network input tensors to be used for
                calibration
            quantizers (List[Optional[UniformQuantizer]]): a list of quantizers that
                should produce the quantized values used in calibration. If none are given,
                the calibration will generate the quantized values with the layer's input
                calibration options.

        Returns:
            numpy.ndarray: calibration data for the following operators
        """

    def _calibrate_layers_activation(
        self,
        calibrate_quantized: bool,
        quantized_op: QuantizedOp,
        *calibration_data: numpy.ndarray,
        quantizers: List[Optional[UniformQuantizer]],
    ) -> Tuple[numpy.ndarray, Optional[UniformQuantizer]]:
        """Calibrate the QuantizedOp with the previous layer's output calibration data.

        Args:
            calibrate_quantized (bool): determines if we use de-quantized values (True) or
                raw values (False) during calibration.
            quantized_op (QuantizedOp): the quantized operator for the current layer.
            *calibration_data: numpy.ndarray: the previous layer's calibration data.
            quantizers (List[Optional[UniformQuantizer]]): a list of quantizers that
                should produce the quantized values used in calibration. If none are given,
                the calibration will generate the quantized values with the layer's input
                calibration options.

        Returns:
            numpy.ndarray: the output of the newly calibrated layer.
        """
        # Calibrate the output of the layer
        raw_result = quantized_op.calibrate(*calibration_data)

        # Some operators need to quantize their inputs using model_outputs instead of op_inputs in
        # order to reduce the impact of quantization.
        if quantized_op.quantize_inputs_with_model_outputs_precision:
            n_bits = self.n_bits_model_outputs
        else:
            n_bits = self.n_bits_op_inputs

        # Create new calibration data (output of the previous layer)
        # Use the op's input options (thus behavior in calibration is the same as in compilation)
        q_calibration_data: List[ONNXOpInputOutputType] = []
        for data in calibration_data:
            is_clear_value = isinstance(data, RawOpOutput)
            if is_clear_value or data is None:
                q_calibration_data.append(data)
            else:
                q_calibration_data.append(
                    QuantizedArray(n_bits, data, True, options=quantized_op.input_quant_opts)
                )

        # Override, when necessary, the calibration data with data that is quantized with
        # layer quantizers that are overridden by the QAT graph quantizers
        for idx, data in enumerate(calibration_data):
            if quantizers[idx] is None:
                continue

            quantizer = quantizers[idx]
            assert quantizer is not None

            q_calibration_data[idx] = QuantizedArray(
                quantizer.n_bits,
                data,
                True,
                options=quantizer.quant_options,
                stats=quantizer.quant_stats,
                params=quantizer.quant_params,
            )

        # Enable rounding calibration if used has set a rounding_threshold_bits
        calibrate_attr = (
            {"calibrate_rounding": True} if isinstance(quantized_op, QuantizedMixingOp) else {}
        )

        # Add calibrate_attr to list of attr for q_impl method
        q_impl_attr = quantized_op.attrs.copy()
        q_impl_attr.update(calibrate_attr)

        # De-quantize to have the value in clear and ready for next calibration
        quant_result = quantized_op.q_impl(*q_calibration_data, **q_impl_attr)
        if quantized_op.produces_graph_output:

            assert isinstance(quant_result, QuantizedArray), (
                "The PyTorch module can not return a raw value, "
                "such as a clear constant or the shape of a tensor."
            )

            assert quantized_op.output_quant_stats is not None
            assert quantized_op.output_quant_params is not None
            quantized_op.output_quant_stats.copy_stats(quant_result.quantizer.quant_stats)
            quantized_op.output_quant_params.copy_params(quant_result.quantizer.quant_params)

        assert_true(
            not quantized_op.produces_raw_output or isinstance(quant_result, RawOpOutput),
            "QuantizedOps whose ONNX numpy implementation is marked as producing raw output"
            " must return an instance of RawOpOutput. \n"
            f" ** Offending Op: {quantized_op.__class__.__name__}",
        )
        # For PTQ, the calibration is performed on quantized data. But
        # raw operation output (RawOpOutput) data should not be quantized
        if calibrate_quantized and not isinstance(quant_result, RawOpOutput):
            assert isinstance(quant_result, QuantizedArray)
            return (
                quant_result.dequant(),
                quant_result.quantizer if isinstance(quant_result, QuantizedArray) else None,
            )

        # For QAT, the calibration is performed on raw data, performing
        # calibration on quantized that would confound inferred QAT and PTQ.
        return (
            raw_result,
            quant_result.quantizer if isinstance(quant_result, QuantizedArray) else None,
        )

    @abstractmethod
    def _process_initializer(self, n_bits: int, values: numpy.ndarray) -> QuantizedArray:
        """Transform a constant tensor according to the model conversion mode.

        The values supplied are floating point values that will be quantized.

        Arguments:
            n_bits (int): Number of bits to quantize the weight values
            values (numpy.ndarray): Float values that initialize this tensor

        Returns:
            QuantizedArray: a quantized tensor with integer values on n_bits bits
        """

    @abstractmethod
    def _get_input_quant_opts(
        self,
        values: Tuple[ONNXOpInputOutputType, ...],
        quantized_op_class: Type["QuantizedOp"],
    ) -> QuantizationOptions:
        """Construct a quantization options set for the input of a layer.

        Args:
            values (Tuple[ONNXOpInputOutputType, ...]): calibration data for this op
            quantized_op_class (Type["QuantizedOp"]): The quantized operator's class

        Returns:
            QuantizationOptions: quantization options set, specific to the network conversion method
        """

    def _quantize_params(self):
        """Transform all floating points initializers to integers."""
        graph: onnx.GraphProto = self.numpy_model.onnx_model.graph
        inits = graph.initializer
        self.quant_params.update(
            (
                onnx_init.name,
                numpy_helper.to_array(onnx_init),
            )
            for onnx_init in inits
        )

    # pylint: disable-next=too-many-branches,too-many-statements
    def _quantize_layers(self, *input_calibration_data: numpy.ndarray):
        """Compute parameters for post-training quantization and generate quantized ops.

        Does a forward pass over a batch of data and compute all
        quantization parameters for activations and layers. Moreover, this function determines
        the compilation mode of the quantized ops: on integers or in floating point.

        Args:
            *input_calibration_data (numpy.ndarray): Data that will be used to compute the bounds,
                scales and zero point values for every quantized object.
        """
        # pylint: disable=too-many-locals

        graph = self.numpy_model.onnx_model.graph

        # Get the list of output tensor names
        graph_output_names = [o.name for o in graph.output]

        node_results: Dict[str, ONNXOpInputOutputType] = dict(
            {
                graph_input.name: input_value
                for graph_input, input_value in zip(graph.input, input_calibration_data)
            },
            **self.quant_params,
        )
        node_override_quantizer: Dict[str, Optional[UniformQuantizer]] = {}

        constants: Set[str] = set(self.quant_params.keys())

        # We need to determine, for each op, whether it only performs univariate computations.
        # A univariate computation is one which depends on a single scalar integer encrypted input
        # which is only multiplied or added to constants or to itself, or a nonlinear function is
        # applied to it.
        # Here a scalar integer encrypted input is a single element in an encrypted tensor.

        # To determine what integer inputs are required for an op, we need to keep track of them
        # through the graph computation. It is not possible to simply check the ONNX input nodes
        # of an op, as they could be tensors produced by ops that do floating point computations
        # with TLUs.

        # We first define which ops perform 'non-fusable' computations and in which settings.
        # Some examples are: gemm & conv, which add together scalars - different elements (cells) of
        # their input encrypted tensors. Another case is Add which, when adding two different
        # encrypted integer inputs, cannot be fused. However, Add can be fused when it adds
        # the results computed from a unique integer tensor. Such as the function f(x) = x + x / 2.

        # We keep track, for each tensor, from which integer tensor(s) it is produced. We also
        # consider that input tensors 'produce' themselves. When an operation can be fused, its
        # input integer tensor names are simply forwarded to the next op.

        # When an op cannot be fused, it produces a new integer encrypted tensor.

        # All tensor names are taken from the ONNX tensor names.

        # First, input tensors produce themselves
        tensor_int_producers: Dict[str, Set[str]] = {
            graph_input.name: {graph_input.name} for graph_input in graph.input
        }

        for node in graph.node:
            op_type = get_op_type(node)

            attributes = {attribute.name: get_attribute(attribute) for attribute in node.attribute}

            # For now only single output nodes
            assert_true(len(node.output) == 1)

            output_name = node.output[0]
            if op_type == "Constant":
                constant_values = ONNX_OPS_TO_NUMPY_IMPL["Constant"](**attributes)[0]
                node_results[output_name] = constant_values
                constants.add(output_name)
                continue

            quantized_op_class = ONNX_OPS_TO_QUANTIZED_IMPL[op_type]

            # Add rounding_threshold_bits to the attributes if available in quantized_op_class
            if issubclass(quantized_op_class, QuantizedMixingOp):
                attributes.update({"rounding_threshold_bits": self.rounding_threshold_bits})

            # All inputs, allow optional constants (they become None)
            curr_inputs = {
                input_name: node_results.get(input_name, None) for input_name in node.input
            }

            # Constant inputs
            curr_cst_inputs: Dict[int, ONNXOpInputOutputType] = {}
            for input_idx, (input_name, value) in enumerate(curr_inputs.items()):
                if not (input_name in self.quant_params or input_name in constants):
                    continue

                if quantized_op_class.must_quantize_input(input_idx):
                    if isinstance(value, QuantizedArray):
                        curr_cst_inputs[input_idx] = value
                    else:
                        # Initializers are ndarray or scalar
                        assert value is not None
                        assert isinstance(value, numpy.ndarray) or numpy.isscalar(value)
                        curr_cst_inputs[input_idx] = self._process_initializer(
                            self.n_bits_op_weights, value
                        )
                else:
                    # Initializers are ndarray or scalar
                    assert isinstance(value, numpy.ndarray) or numpy.isscalar(value)
                    curr_cst_inputs[input_idx] = value

            has_variable_inputs = (len(curr_inputs) - len(curr_cst_inputs)) > 0

            variable_input_names = [
                input_name for input_name in curr_inputs if input_name not in constants
            ]
            curr_calibration_data = tuple(
                curr_inputs[input_name] for input_name in variable_input_names
            )

            # For mypy
            assert_true(
                all(val is None or isinstance(val, numpy.ndarray) for val in curr_calibration_data)
            )
            curr_calibration_data = cast(Tuple[numpy.ndarray], curr_calibration_data)

            # Find the unique integer producers of the current's op output tensor
            node_integer_inputs = set.union(
                *[tensor_int_producers.get(input_node, set()) for input_node in node.input]
            )

            # If we depend on a variable input use the quantized version of the operator
            if has_variable_inputs:

                assert_true(
                    op_type in ONNX_OPS_TO_QUANTIZED_IMPL,
                    f"{op_type} can't be found in {ONNX_OPS_TO_QUANTIZED_IMPL}",
                )

                # Note that the output of a quantized op could be a network output
                # Thus the quantized op outputs are quantized to the network output bit-width
                quantized_op_instance = quantized_op_class(
                    self.n_bits_model_outputs,
                    node.name,
                    node_integer_inputs,
                    curr_cst_inputs,
                    self._get_input_quant_opts(curr_calibration_data, quantized_op_class),
                    **attributes,
                )

                # Determine if this op computes a tensor that is a graph output, i.e., a tensor
                # that will be decrypted and de-quantized in the clear
                quantized_op_instance.produces_graph_output = output_name in graph_output_names

                # Store the output tensor's integer producers
                tensor_int_producers[output_name] = set()
                if not quantized_op_instance.can_fuse():
                    # This tensor is produced by a non fusable op
                    # Thus this tensor is marked as produced by itself
                    tensor_int_producers[output_name].add(output_name)
                else:
                    # If the op that produces this output tensor is fusable
                    # the output tensor's integer producers are the same as the op's inputs'
                    # integer producers (forwarding)
                    tensor_int_producers[output_name] = node_integer_inputs

                # Store the learned quantized layer
                self.quant_ops_dict[output_name] = (
                    tuple(variable_input_names),
                    quantized_op_instance,
                )

                layer_quant = list(
                    node_override_quantizer.get(input_name, None)
                    for input_name in variable_input_names
                )
                output_calibration_data, layer_quantizer = self._process_layer(
                    quantized_op_instance, *curr_calibration_data, quantizers=layer_quant
                )
                node_results[output_name] = output_calibration_data
                node_override_quantizer[output_name] = layer_quantizer

            # Otherwise use the original operator to operate on float values to do constant folding
            else:
                # Get the non quantized values
                # Note that we handle both numpy.array and QuantizedArray here.
                # Inputs to a constant folding step are always float but may produce
                # either numpy.array or QuantizedArray
                real_cst_inputs = (
                    input.values if isinstance(input, QuantizedArray) else input
                    for input in curr_cst_inputs.values()
                )

                # The output of a constant folding op can be either QuantizedArray or numpy.ndarray
                node_output: Tuple[ONNXOpInputOutputType, ...] = ()

                if get_op_type(node) == QuantizedBrevitasQuant.op_type():
                    list_real_cst_inputs = list(real_cst_inputs)
                    quantizer = QuantizedBrevitasQuant(
                        self.n_bits_model_outputs,
                        node.name,
                        node_integer_inputs,
                        {
                            1: list_real_cst_inputs[1],
                            2: list_real_cst_inputs[2],
                            3: list_real_cst_inputs[3],
                        },
                        self._get_input_quant_opts(curr_calibration_data, quantized_op_class),
                        **attributes,
                    )
                    # The values to quantize may be stored in a QuantizedArray (for initializers
                    # and constants)
                    assert isinstance(
                        curr_cst_inputs[0], QuantizedArray
                    ), "Only QuantizedArray constant inputs of a Brevitas quantizer are supported"

                    constant_values_to_quantize = curr_cst_inputs[0]

                    # QuantizedBrevitasQuant takes a single input
                    quantizer.calibrate(constant_values_to_quantize.values)
                    node_output = (quantizer(constant_values_to_quantize),)
                else:
                    node_output = ONNX_OPS_TO_NUMPY_IMPL[op_type](*real_cst_inputs, **attributes)
                num_output = len(node_output)
                assert_true(
                    (num_output) == 1,
                    f"Currently {self.__class__.__name__} can only manage single output operator, "
                    f"got {num_output} for op {op_type}",
                    RuntimeError,
                )
                node_results[output_name] = node_output[0]
                constants.add(output_name)

    def quantize_module(self, *calibration_data: numpy.ndarray) -> QuantizedModule:
        """Quantize numpy module.

        Following https://arxiv.org/abs/1712.05877 guidelines.

        Args:
            *calibration_data (numpy.ndarray):  Data that will be used to compute the bounds,
                                                scales and zero point values for every quantized
                                                object.

        Returns:
            QuantizedModule: Quantized numpy module
        """
        # First transform all parameters to their quantized version
        self._quantize_params()

        self._quantize_layers(*calibration_data)

        # Create quantized module from self.quant_layers_dict
        quantized_module = QuantizedModule(
            ordered_module_input_names=(
                graph_input.name for graph_input in self.numpy_model.onnx_model.graph.input
            ),
            ordered_module_output_names=(
                graph_output.name for graph_output in self.numpy_model.onnx_model.graph.output
            ),
            quant_layers_dict=self.quant_ops_dict,
            onnx_model=self.numpy_model.onnx_model,
        )

        self._process_input_quantizers(quantized_module, calibration_data)

        return quantized_module

    def _process_input_quantizers(
        self, quantized_module: QuantizedModule, calibration_data: Tuple[numpy.ndarray, ...]
    ):  # pylint: disable=too-many-branches
        """Determine the quantizers for a quantized module.

        Args:
            quantized_module (QuantizedModule): the quantized module containing the ops of the model
            calibration_data: calibration data for each input tensor
        """

        # Create several lists:
        # - a list of layers that use each input directly
        #   (i.e., have this input as an integer input)
        # - a list of quantizers that are applied to each input node
        # - a list of inputs that have TLUs, for which these TLUs cannot be removed
        layer_using_input: Dict[int, List[QuantizedOp]] = {}
        quantizers_for_input: Dict[int, List[QuantizedBrevitasQuant]] = {}
        inputs_not_optimizable: List[int] = []

        # Determine, for each input, whether it is used by a single non-fusable layer
        # and, optionally, if it is processed by a QAT quantizer in the process

        # If the input is used by a fusable op directly, or goes through a uniform QAT quantizer
        # then we can use the QAT quantizer's or fusable-op's quantizer in the clear, instead
        # of relying on default quantization parameters in the clear (using model_inputs bits)
        for inp_idx, graph_input in enumerate(self.numpy_model.onnx_model.graph.input):
            layer_using_input[inp_idx] = []
            for _, q_op in quantized_module.quant_layers_dict.values():
                # If this op does not use this input (irrespective of univariate fusable
                # processing it may have gone through), ignore it
                assert isinstance(q_op, QuantizedOp)
                if graph_input.name not in q_op.int_input_names:
                    continue

                # This op uses this input (or a univariate transform of it)
                if not q_op.can_fuse():
                    # If this is a non-fusable op working on integers, store it
                    # If there is no QAT quantizer on this input, we'll use the op's
                    # input quantizer as the clear-input quantizer
                    layer_using_input[inp_idx].append(q_op)
                elif isinstance(q_op, QuantizedBrevitasQuant):
                    # If this is a fusable op that is a QAT quantizer, store it
                    # We'll use this quantizer's parameters in the clear-input quantizer
                    if inp_idx not in quantizers_for_input:
                        quantizers_for_input[inp_idx] = []
                    quantizers_for_input[inp_idx].append(q_op)
                else:
                    # If the input is processed by some other type of univariate layer
                    # we cannot optimize out the TLUs on this op by moving them in the clear
                    inputs_not_optimizable.append(inp_idx)

        # The input quantizers which we can extract from the graph
        # are:
        # - in quantizers_for_input : QAT quantizers applied directly to inputs
        # - in layer_using_input : quantizers taken from non-fusable ops
        # Now set the input quantizers based on the quantizers that can be extracted from the graph
        q_input_list = []
        for inp_idx, val in enumerate(calibration_data):
            # If multiple QAT Quantizers are applied to an input, it can not be optimized
            # to remove the TLU and use a single quantizer in the clear
            has_multiple_qat_quantizers = (
                inp_idx in quantizers_for_input and len(quantizers_for_input[inp_idx]) > 1
            )

            if inp_idx in inputs_not_optimizable or has_multiple_qat_quantizers:
                # If an input is not optimizable, use the "model_inputs" bits set by the user
                q_input_list.append(
                    QuantizedArray(self.n_bits_model_inputs, val, is_signed=True).quantizer
                )
            elif inp_idx in quantizers_for_input:
                # If a QAT quantizer is applied to this input, use its output params that
                # are determined from the ONNX file
                quantizer = quantizers_for_input[inp_idx][0]
                opts = QuantizationOptions(quantizer.output_quant_opts.n_bits)
                opts.copy_opts(quantizer.output_quant_opts)
                # Set the same options as those produced by a QuantizedBrevitasQuant op
                q_input_list.append(
                    UniformQuantizer(
                        opts,
                        quantizer.output_quant_stats,
                        quantizer.output_quant_params,
                    )
                )

                # Propagate the quantization options to non-fusable ops down the line,
                # since these ops were initialized with a default n_bits - not necessarily
                # the n_bits set in the QuantizedBrevitasQuant layer
                for q_op in layer_using_input[inp_idx]:
                    q_op.input_quant_opts.copy_opts(quantizer.output_quant_opts)
            else:
                # If the input is injected directly into a non-fusable op (conv, etc...),
                # use that op's quantization options (ensures matching options that allows
                # the optimization to take place)
                opts = layer_using_input[inp_idx][0].input_quant_opts
                for layer in layer_using_input[inp_idx][1:]:
                    opts_k = layer.input_quant_opts
                    assert_true(
                        opts.is_equal(opts_k),
                        "Multiple quantizers "
                        "applied to the same input must have the same quantization options",
                    )

                q_input_list.append(QuantizedArray(opts.n_bits, val, options=opts).quantizer)

        q_input = tuple(q_input_list)
        quantized_module.set_inputs_quantization_parameters(*q_input)


class PostTrainingAffineQuantization(ONNXConverter):
    """Post-training Affine Quantization.

    Create the quantized version of the passed numpy module.

    Args:
        n_bits (int, Dict):             Number of bits to quantize the model. If an int is passed
                                        for n_bits, the value will be used for activation,
                                        inputs and weights. If a dict is passed, then it should
                                        contain  "model_inputs", "op_inputs", "op_weights" and
                                        "model_outputs" keys with corresponding number of
                                        quantization bits for:
                                        - model_inputs : number of bits for model input
                                        - op_inputs : number of bits to quantize layer input values
                                        - op_weights: learned parameters or constants in the network
                                        - model_outputs: final model output quantization bits
        numpy_model (NumpyModule):      Model in numpy.
        rounding_threshold_bits (int): if not None, every accumulators in the model are rounded down
            to the given bits of precision
        is_signed:                      Whether the weights of the layers can be signed.
                                        Currently, only the weights can be signed.

    Returns:
        QuantizedModule: A quantized version of the numpy model.
    """

    def _process_layer(
        self,
        quantized_op: QuantizedOp,
        *calibration_data: numpy.ndarray,
        quantizers: List[Optional[UniformQuantizer]],
    ) -> Tuple[numpy.ndarray, Optional[UniformQuantizer]]:
        """Configure a graph operation by performing calibration for uniform quantization.

        Args:
            quantized_op (QuantizedOp): Quantized graph operator instance
            *calibration_data (numpy.ndarray): tuple of network input tensors to be used for
                calibration
            quantizers (List[Optional[UniformQuantizer]]): a list of quantizers that
                should produce the quantized values used in calibration. If none are given,
                the calibration will generate the quantized values with the layer's input
                calibration options.

        Returns:
            numpy.ndarray: calibration data for the following operators
        """

        return self._calibrate_layers_activation(
            True, quantized_op, *calibration_data, quantizers=quantizers
        )

    def _process_initializer(self, n_bits: int, values: numpy.ndarray):
        """Quantize a network constant tensor (a weights tensor).

        The values supplied are floating point values that will be quantized.

        Arguments:
            n_bits (int): Number of bits to quantize the weight values
            values (numpy.ndarray): Float values that initialize this tensor

        Returns:
            QuantizedArray: a quantized tensor with integer values on n_bits bits
        """

        if isinstance(values, numpy.ndarray) and numpy.issubdtype(values.dtype, numpy.integer):
            return values.view(RawOpOutput)

        assert isinstance(values, (numpy.ndarray, float))
        is_signed = is_symmetric = self._check_distribution_is_symmetric_around_zero(values)

        return QuantizedArray(
            n_bits,
            values,
            is_signed=is_signed,
            is_symmetric=is_symmetric,
        )

    def _get_input_quant_opts(
        self,
        values: Tuple[ONNXOpInputOutputType, ...],
        quantized_op_class: Type["QuantizedOp"],
    ):
        """Construct a quantization options set for the input of a layer.

        Inputs and activations require signed quantization.

        Args:
            values (Tuple[ONNXOpInputOutputType, ...]): calibration data for this op
            quantized_op_class (Type["QuantizedOp"]): The quantized operator's class

        Returns:
            QuantizationOptions: quantization options set, specific to the network conversion method
        """
        is_signed = any(v.min() < 0 for v in values if isinstance(v, numpy.ndarray)) or any(
            v.values.min() < 0 for v in values if isinstance(v, QuantizedArray)
        )

        # Some operators need to quantize their inputs using model_outputs instead of op_inputs in
        # order to reduce the impact of quantization.
        if quantized_op_class.quantize_inputs_with_model_outputs_precision:
            n_bits = self.n_bits_model_outputs
        else:
            n_bits = self.n_bits_op_inputs

        opts = QuantizationOptions(
            n_bits,
            is_signed=is_signed,
        )
        return opts

    @staticmethod
    def _check_distribution_is_symmetric_around_zero(values: numpy.ndarray) -> bool:
        """Check if the distribution of the values is somewhat symmetric around 0.

        Neural network weights are usually symmetric, while regression coefficients
        are usually non-symmetric

        Symmetric quantization will have a zero zero-point, which avoids the computation
        of a term in the quantized Gemm, leading to lower overall circuit bit-width
        and faster speed. However, symmetric quantization can lose precision if the distribution
        of the original values is not symmetric

        Args:
            values (numpy.ndarray): a sample from the distribution to check

        Returns:
            bool: whether the distribution can be considered symmetric around 0
        """

        vmin, vmax = numpy.percentile(values, 3), numpy.percentile(values, 97)
        ratio_min_max = 0 if numpy.abs(vmax) < 0.001 else numpy.abs(vmin / vmax)

        max_skew = 3
        # We check if the distribution support contains zero, and if
        # the size of the support of the distribution on one side of 0 is
        # not too large (skewed distribution) with respect to the support on the other side.
        return not (
            (vmin > 0 and vmax > 0)
            or (vmin < 0 and vmax < 0)
            or ratio_min_max < 1 / max_skew
            or ratio_min_max > max_skew
        )


class PostTrainingQATImporter(ONNXConverter):
    """Converter of Quantization Aware Training networks.

    This class provides specific configuration for QAT networks during ONNX network conversion
    to Concrete ML computation graphs.
    """

    def _process_layer(
        self,
        quantized_op: QuantizedOp,
        *calibration_data: numpy.ndarray,
        quantizers: List[Optional[UniformQuantizer]],
    ) -> Tuple[numpy.ndarray, Optional[UniformQuantizer]]:
        """Configure a graph operation by calibrating it for Quantization Aware Training.

        Args:
            quantized_op (QuantizedOp): Quantized graph operator instance
            *calibration_data (numpy.ndarray): tuple of network input tensors to be used for
                calibration
            quantizers (List[Optional[UniformQuantizer]]): a list of quantizers that
                should produce the quantized values used in calibration. If none are given,
                the calibration will generate the quantized values with the layer's input
                calibration options.

        Returns:
            numpy.ndarray: calibration data for the following operators
        """

        return self._calibrate_layers_activation(
            False, quantized_op, *calibration_data, quantizers=quantizers
        )

    def _process_initializer(self, n_bits: int, values: numpy.ndarray):
        """Process an already quantized weight tensor.

        The values supplied are in floating point, but are discrete, in the sense
        that the number of unique (possible) values is equal to 2**n_bits. These values,
        w_hat take the form w_hat = alpha * w_int, where w_int are integer values. Therefore, this
        function will retrieve the alpha and w_int that form w_hat which is passed here as `values`.

        Arguments:
            n_bits (int): Number of bits that was used to quantize the weight values
            values (numpy.ndarray): Discrete float values that initialize this tensor

        Returns:
            QuantizedArray: a quantized tensor with integer values on n_bits bits and with alpha as
                the scaling factor.
        """

        # Assume that integer initializer op inputs are raw values that should not be quantized
        # This allows to have patterns such as Add(shape(x)[0], 1). In this case the value `1`
        # is the integer initializer and should not be quantized.
        if isinstance(values, numpy.ndarray) and numpy.issubdtype(values.dtype, numpy.integer):
            return values.view(RawOpOutput)

        assert_true(
            isinstance(values, float)
            or (
                isinstance(values, numpy.ndarray) and numpy.issubdtype(values.dtype, numpy.floating)
            ),
            f"ONNX initializers must be float if not marked as raw arguments, "
            f"got {values.dtype if isinstance(values, numpy.ndarray) else type(values)} "
            f"for initializer {values}",
        )
        return QuantizedArray(n_bits, values, is_signed=True, is_symmetric=False, is_qat=True)

    def _get_input_quant_opts(
        self,
        values: Tuple[ONNXOpInputOutputType, ...],
        quantized_op_class: Type["QuantizedOp"],
    ):
        """Construct a quantization options set for the input of a layer of a QAT network.

        QAT networks require specific quantization for inputs to nodes that perform quantization
        (such as Gemm/Conv/Add).

        Args:
            values (Tuple[ONNXOpInputOutputType, ...]): calibration data for this op
            quantized_op_class (Type["QuantizedOp"]): The quantized operator's class

        Returns:
            QuantizationOptions: quantization options set, specific to the network conversion method
        """

        # Some operators need to quantize their inputs using model_outputs instead of op_inputs in
        # order to reduce the impact of quantization.
        if quantized_op_class.quantize_inputs_with_model_outputs_precision:
            n_bits = self.n_bits_model_outputs
        else:
            n_bits = self.n_bits_op_inputs

        opts = QuantizationOptions(n_bits, is_signed=True, is_qat=True)
        return opts
