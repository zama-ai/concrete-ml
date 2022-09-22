"""Post Training Quantization methods."""

from abc import abstractmethod
from typing import Dict, Set, Tuple, Type, Union, cast

import numpy
import onnx
from onnx import numpy_helper

from ..common.debugging import assert_true
from ..onnx.onnx_utils import ONNX_OPS_TO_NUMPY_IMPL, get_attribute, get_op_name
from ..torch.numpy_module import NumpyModule
from .base_quantized_op import DEFAULT_MODEL_BITS, ONNX_OPS_TO_QUANTIZED_IMPL, QuantizedOp
from .quantized_module import QuantizedModule
from .quantizers import QuantizationOptions, QuantizedArray


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
        y_model (NumpyModule): Model in numpy.
        is_signed (bool): Whether the weights of the layers can be signed. Currently, only the
            weights can be signed.
    """

    quant_ops_dict: Dict[str, Tuple[Tuple[str, ...], QuantizedOp]]
    n_bits: Union[int, Dict]
    quant_params: Dict[str, QuantizedArray]
    numpy_model: NumpyModule
    is_signed: bool

    def __init__(self, n_bits: Union[int, Dict], numpy_model: NumpyModule, is_signed: bool = False):
        self.quant_ops_dict = {}

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

        self.n_bits = n_bits_dict
        self.quant_params = {}
        self.numpy_model = numpy_model
        self.is_signed = is_signed

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
    ) -> numpy.ndarray:
        """Configure a graph operation according to model conversion mode.

        Args:
            quantized_op (QuantizedOp): Quantized graph operator instance
            *calibration_data (numpy.ndarray): tuple of network input tensors to be used for
                calibration

        Returns:
            numpy.ndarray: calibration data for the following operators
        """

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
        self, values: Tuple[numpy.ndarray], quantized_op_class: Type["QuantizedOp"]
    ) -> QuantizationOptions:
        """Construct a quantization options set for the input of a layer.

        Args:
            values (Tuple[numpy.ndarray]): calibration data for this op
            quantized_op_class (Type["QuantizedOp"]): The quantized operator's class

        Returns:
            QuantizationOptions: quantization options set, specific to the network conversion method
        """

    # TODO: https://github.com/zama-ai/concrete-ml-internal/issues/195
    # probably could work by calling layer.quantize_params on each layer just before quantizing them
    # layer to have weight quantization that manages all that by itself.
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

        node_results: Dict[str, Union[numpy.ndarray, QuantizedArray]] = dict(
            {
                graph_input.name: input_value
                for graph_input, input_value in zip(graph.input, input_calibration_data)
            },
            **self.quant_params,
        )

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
        # encrypted integer inputs, can not be fused. However, Add can be fused when it adds
        # the results computed from a unique integer tensor. Such as the function f(x) = x + x / 2.

        # We keep track, for each tensor, from which integer tensor(s) it is produced. We also
        # consider that input tensors 'produce' themselves. When an operation can be fused, its
        # input integer tensor names are simply forwarded to the next op.

        # When an op can not be fused, it produces a new integer encrypted tensor.

        # All tensor names are taken from the ONNX tensor names.

        # First, input tensors produce themselves
        tensor_int_producers: Dict[str, Set[str]] = {
            graph_input.name: {graph_input.name} for graph_input in graph.input
        }

        for node in graph.node:
            op_type = get_op_name(node)

            attributes = {attribute.name: get_attribute(attribute) for attribute in node.attribute}

            # For now only single output nodes
            assert_true(len(node.output) == 1)

            output_name = node.output[0]
            if op_type == "Constant":

                # FIXME: Do not handle constants with QuantizedArray, use numpy.ndarray
                # let the ops quantize their own inputs
                constant_values = ONNX_OPS_TO_NUMPY_IMPL["Constant"](**attributes)[0]
                node_results[output_name] = constant_values
                constants.add(output_name)
                continue

            quantized_op_class = ONNX_OPS_TO_QUANTIZED_IMPL[op_type]

            # All inputs
            curr_inputs = {input_name: node_results[input_name] for input_name in node.input}

            # Constant inputs
            curr_cst_inputs = {}
            for input_idx, (input_name, value) in enumerate(curr_inputs.items()):
                if not (input_name in self.quant_params or input_name in constants):
                    continue

                # Initializers are ndarray
                assert isinstance(value, numpy.ndarray) or numpy.isscalar(value)

                curr_cst_inputs[input_idx] = (
                    value
                    if not quantized_op_class.must_quantize_input(input_idx)
                    else self._process_initializer(self.n_bits_op_weights, value)
                )

            has_variable_inputs = (len(curr_inputs) - len(curr_cst_inputs)) > 0

            # If we depend on a variable input use the quantized version of the operator
            if has_variable_inputs:

                assert_true(
                    op_type in ONNX_OPS_TO_QUANTIZED_IMPL,
                    f"{op_type} can't be found in {ONNX_OPS_TO_QUANTIZED_IMPL}",
                )

                variable_input_names = [
                    input_name for input_name in curr_inputs if input_name not in constants
                ]
                curr_calibration_data = tuple(
                    curr_inputs[input_name] for input_name in variable_input_names
                )

                # For mypy
                assert_true(all(isinstance(val, numpy.ndarray) for val in curr_calibration_data))
                curr_calibration_data = cast(Tuple[numpy.ndarray], curr_calibration_data)

                # Find the unique integer producers of the current's op output tensor
                node_integer_inputs = set.union(
                    *[tensor_int_producers.get(input_node, set()) for input_node in node.input]
                )

                # Note that the output of a quantized op could be a network output
                # Thus the quantized op outputs are quantized to the network output bitwidth
                quantized_op_instance = quantized_op_class(
                    self.n_bits_model_outputs,
                    node_integer_inputs,
                    curr_cst_inputs,
                    self._get_input_quant_opts(curr_calibration_data, quantized_op_class),
                    **attributes,
                )

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

                output_calibration_data = self._process_layer(
                    quantized_op_instance, *curr_calibration_data
                )
                node_results[output_name] = output_calibration_data

            # Otherwise use the original operator to operate on float values to do constant folding
            else:
                # Get the non quantized values
                real_cst_inputs = (
                    input.values if isinstance(input, QuantizedArray) else input
                    for input in curr_cst_inputs.values()
                )

                node_output = ONNX_OPS_TO_NUMPY_IMPL[op_type](*real_cst_inputs, **attributes)
                assert_true(
                    (num_output := len(node_output)) == 1,
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
            (graph_input.name for graph_input in self.numpy_model.onnx_model.graph.input),
            (graph_output.name for graph_output in self.numpy_model.onnx_model.graph.output),
            self.quant_ops_dict,
        )

        q_input = tuple(
            QuantizedArray(self.n_bits_model_inputs, val, is_signed=False).quantizer
            for val in calibration_data
        )

        # Note that the input quantizer can not be signed
        # since Concrete Numpy does not yet support signed inputs
        assert_true(
            all(not q.is_signed for q in q_input),
            "The QuantizedModule input " "quantizer can not be signed",
        )

        quantized_module.set_inputs_quantization_parameters(*q_input)
        return quantized_module


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
        is_signed:                      Whether the weights of the layers can be signed.
                                        Currently, only the weights can be signed.

    Returns:
        QuantizedModule: A quantized version of the numpy model.
    """

    def _process_layer(
        self,
        quantized_op: QuantizedOp,
        *calibration_data: numpy.ndarray,
    ):
        """Configure a graph operation by performing calibration for uniform quantization.

        Args:
            quantized_op (QuantizedOp): Quantized graph operator instance
            *calibration_data (numpy.ndarray): tuple of network input tensors to be used for
                calibration

        Returns:
            numpy.ndarray: calibration data for the following operators
        """

        return self._calibrate_layers_activation(quantized_op, *calibration_data)

    def _calibrate_layers_activation(
        self,
        quantized_op: QuantizedOp,
        *calibration_data: numpy.ndarray,
    ) -> numpy.ndarray:
        """Calibrate the QuantizedOp linked to name with previous calibration data.

        Args:
            quantized_op (QuantizedOp): the quantized operator for the current layer.
            *calibration_data: numpy.ndarray: the previous layer's calibration data.

        Returns:
            numpy.ndarray: the output of the newly calibrated layer.
        """
        # Calibrate the output of the layer
        quantized_op.calibrate(*calibration_data)

        # Some operators need to quantize their inputs using model_outputs instead of op_inputs in
        # order to reduce the impact of quantization.
        if quantized_op.quantize_inputs_with_model_outputs_precision:
            n_bits = self.n_bits_model_outputs
        else:
            n_bits = self.n_bits_op_inputs

        # Create new calibration data (output of the previous layer)
        q_calibration_data = (QuantizedArray(n_bits, data) for data in calibration_data)

        # Dequantize to have the value in clear and ready for next calibration
        return quantized_op(*q_calibration_data).dequant()

    def _process_initializer(self, n_bits: int, values: numpy.ndarray):
        """Quantize a network constant tensor (a weights tensor).

        The values supplied are floating point values that will be quantized.

        Arguments:
            n_bits (int): Number of bits to quantize the weight values
            values (numpy.ndarray): Float values that initialize this tensor

        Returns:
            QuantizedArray: a quantized tensor with integer values on n_bits bits
        """

        is_signed = is_symmetric = self._check_distribution_is_symmetric_around_zero(values)

        return QuantizedArray(
            n_bits,
            values,
            is_signed=is_signed,
            is_symmetric=is_symmetric,
        )

    def _get_input_quant_opts(
        self, values: Tuple[numpy.ndarray], quantized_op_class: Type["QuantizedOp"]
    ):
        """Construct a quantization options set for the input of a layer.

        Inputs and activations require signed quantization.

        Args:
            values (Tuple[numpy.ndarray]): calibration data for this op
            quantized_op_class (Type["QuantizedOp"]): The quantized operator's class

        Returns:
            QuantizationOptions: quantization options set, specific to the network conversion method
        """

        is_signed = any(v.min() < 0 for v in values)

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
        of a term in the quantized Gemm, leading to lower overall circuit bitwidth
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
    ):
        """Configure a graph operation by calibrating it for Quantization Aware Training.

        Args:
            quantized_op (QuantizedOp): Quantized graph operator instance
            *calibration_data (numpy.ndarray): tuple of network input tensors to be used for
                calibration

        Returns:
            numpy.ndarray: calibration data for the following operators
        """

        layer_output = quantized_op.calibrate(*calibration_data)
        return layer_output

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
        return QuantizedArray(n_bits, values, is_signed=True, is_symmetric=False, is_qat=True)

    def _get_input_quant_opts(
        self, values: Tuple[numpy.ndarray], quantized_op_class: Type["QuantizedOp"]
    ):
        """Construct a quantization options set for the input of a layer of a QAT network.

        QAT networks require specific quantization for inputs to nodes that perform quantization
        (such as Gemm/Conv/Add).

        Args:
            values (Tuple[numpy.ndarray]): calibration data for this op
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
