"""Post Training Quantization methods."""

from typing import Dict, Iterable, Set, Tuple, Union, cast

import numpy
import onnx
from onnx import numpy_helper

from ..common.debugging import assert_true
from ..onnx.onnx_utils import ONNX_OPS_TO_NUMPY_IMPL, get_attribute
from ..torch.numpy_module import NumpyModule
from .base_quantized_op import ONNX_OPS_TO_QUANTIZED_IMPL, QuantizedOp
from .quantized_array import QuantizedArray
from .quantized_module import QuantizedModule


class PostTrainingAffineQuantization:
    """Post-training Affine Quantization.

    Create the quantized version of the passed numpy module.

    Args:
        n_bits (int, Dict): Number of bits to quantize the model. If an int is passed for n_bits,
            the value will be used for activation, inputs and weights. If a dict is passed, then it
            should contain "inputs", "weights" and "outputs" keys with corresponding number of
            quantization bits for:
            - inputs : any input data to any layers
            - weights: learned parameters or constants in the network
            - outputs: final model output
        numpy_model (NumpyModule): Model in numpy.
        is_signed (bool): Whether the weights of the layers can be signed. Currently, only the
            weights can be signed.

    Returns:
        QuantizedModule: A quantized version of the numpy model.
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
            or (isinstance(n_bits, Dict) and n_bits.keys() == {"inputs", "weights", "outputs"}),
            "Invalid n_bits, either pass an integer or a dictionary containing integer values for"
            " the `inputs`, `weights` and 'outputs' keys",
        )
        self.n_bits = n_bits
        self.quant_params = {}
        self.numpy_model = numpy_model
        self.is_signed = is_signed

    @property
    def n_bits_inputs(self):
        """Get the number of bits to use for the quantization of all layers' input.

        Returns:
            n_bits (int): number of bits for input quantization
        """
        if isinstance(self.n_bits, Dict):
            return self.n_bits["inputs"]
        return self.n_bits

    @property
    def n_bits_outputs(self):
        """Get the number of bits to use for the quantization of the last layer's output.

        Returns:
            n_bits (int): number of bits for output quantization
        """
        if isinstance(self.n_bits, Dict):
            return self.n_bits["outputs"]
        return self.n_bits

    @property
    def n_bits_weights(self):
        """Get the number of bits to use for the quantization of any constant (usually weights).

        Returns:
            n_bits (int): number of bits for constants quantization
        """
        if isinstance(self.n_bits, Dict):
            return self.n_bits["weights"]
        return self.n_bits

    # TODO: https://github.com/zama-ai/concrete-ml-internal/issues/195
    # probably could work by calling layer.quantize_params on each layer just before quantizing them
    # layer to have weight quantization that manages all that by itself.
    def _quantize_params(self):
        """Transform all floating points initializers to integers."""
        graph: onnx.GraphProto = self.numpy_model.get_onnx().graph
        inits = graph.initializer
        self.quant_params.update(
            (
                onnx_init.name,
                QuantizedArray(
                    self.n_bits_weights,
                    numpy_helper.to_array(onnx_init),
                    is_signed=numpy_helper.to_array(onnx_init).min() < 0,
                    is_symmetric=True,
                ),
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

        graph = self.numpy_model.get_onnx().graph

        node_results: Dict[str, Union[numpy.ndarray, QuantizedArray]] = dict(
            {
                graph_input.name: input_value
                for graph_input, input_value in zip(graph.input, input_calibration_data)
            },
            **self.quant_params,
        )

        constants: Set[str] = set(self.quant_params.keys())

        # Retrieve the last node
        last_node = None
        for node in reversed(graph.node):
            if node.op_type in ["MatMul", "Gemm", "Conv", "Exp"]:
                last_node = node
                break

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
            op_type = node.op_type
            attributes = {attribute.name: get_attribute(attribute) for attribute in node.attribute}

            # For now only single output nodes
            assert_true(len(node.output) == 1)

            output_name = node.output[0]
            if op_type == "Constant":

                # FIXME: Do not handle constants with QuantizedArray, use numpy.ndarray
                # let the ops quantize their own inputs
                node_results[output_name] = QuantizedArray(
                    self.n_bits_weights,
                    ONNX_OPS_TO_NUMPY_IMPL["Constant"](**attributes)[0],
                    self.is_signed,
                )
                constants.add(output_name)
                continue

            # All inputs
            curr_inputs = {input_name: node_results[input_name] for input_name in node.input}

            # Constant inputs
            curr_cst_inputs = {
                input_idx: value
                for input_idx, (input_name, value) in enumerate(curr_inputs.items())
                if input_name in constants
            }

            has_variable_inputs = (len(curr_inputs) - len(curr_cst_inputs)) > 0

            # If we depend on a variable input use the quantized version of the operator
            if has_variable_inputs:

                assert_true(
                    op_type in ONNX_OPS_TO_QUANTIZED_IMPL,
                    f"{op_type} can't be found in {ONNX_OPS_TO_QUANTIZED_IMPL}",
                )

                quantized_op_class = ONNX_OPS_TO_QUANTIZED_IMPL[op_type]
                variable_input_names = [
                    input_name for input_name in curr_inputs if input_name not in constants
                ]
                curr_calibration_data = tuple(
                    curr_inputs[input_name] for input_name in variable_input_names
                )

                # For mypy
                assert_true(all(isinstance(val, numpy.ndarray) for val in curr_calibration_data))
                curr_calibration_data = cast(Tuple[numpy.ndarray, ...], curr_calibration_data)

                # The last node's values are quantized using the number of bits given for outputs
                n_bits_op = self.n_bits_outputs if node == last_node else self.n_bits_inputs

                # Find the unique integer producers of the current's op output tensor
                node_integer_inputs = set.union(
                    *[tensor_int_producers.get(input_node, set()) for input_node in node.input]
                )

                quantized_op_instance = quantized_op_class(
                    n_bits_op, node_integer_inputs, curr_cst_inputs, **attributes
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

                output_calibration_data = self._calibrate_layers_activation(
                    variable_input_names, output_name, quantized_op_instance, *curr_calibration_data
                )
                node_results[output_name] = output_calibration_data

            # Otherwise use the original operator to operate on float values to do constant folding
            else:
                # For mypy
                assert_true(
                    all(
                        isinstance(key, int) and isinstance(val, QuantizedArray)
                        for key, val in curr_cst_inputs.items()
                    )
                )
                curr_q_cst_inputs = cast(Dict[int, QuantizedArray], curr_cst_inputs)

                # Get the non quantized values
                real_cst_inputs = (qarray.values for qarray in curr_q_cst_inputs.values())
                node_output = ONNX_OPS_TO_NUMPY_IMPL[op_type](*real_cst_inputs, **attributes)
                assert_true(
                    (num_output := len(node_output)) == 1,
                    f"Currently {self.__class__.__name__} can only manage single output operator, "
                    f"got {num_output} for op {op_type}",
                    RuntimeError,
                )
                q_node_output = QuantizedArray(self.n_bits_weights, node_output[0], self.is_signed)
                node_results[output_name] = q_node_output
                constants.add(output_name)

    def _calibrate_layers_activation(
        self,
        variable_input_names: Iterable[str],
        name: str,
        quantized_op: QuantizedOp,
        *calibration_data: numpy.ndarray,
    ) -> numpy.ndarray:
        """Calibrate the QuantizedOp linked to name with previous calibration data.

        Args:
            variable_input_names (Iterable[str]): an iterable containing the ordered variable input
                names to the layer being calibrated.
            name (str): the name of the output/layer coming from the ONNX model.
            quantized_op (QuantizedOp): the quantized operator for the current layer.
            *calibration_data: numpy.ndarray: the previous layer's calibration data.

        Returns:
            numpy.ndarray: the output of the newly calibrated layer.
        """
        # Calibrate the output of the layer
        quantized_op.calibrate(*calibration_data)

        # Store the learned quantized layer
        self.quant_ops_dict[name] = (tuple(variable_input_names), quantized_op)

        # Create new calibration data (output of the previous layer)
        q_calibration_data = (QuantizedArray(self.n_bits_inputs, data) for data in calibration_data)

        # Dequantize to have the value in clear and ready for next calibration
        return quantized_op(*q_calibration_data).dequant()

    def quantize_module(self, *calibration_data: numpy.ndarray) -> QuantizedModule:
        """Quantize numpy module.

        Following https://arxiv.org/abs/1712.05877 guidelines.

        Args:
            *calibration_data (numpy.ndarray):   Data that will be used to compute the bounds,
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
            (graph_input.name for graph_input in self.numpy_model.get_onnx().graph.input),
            (graph_output.name for graph_output in self.numpy_model.get_onnx().graph.output),
            self.quant_ops_dict,
        )
        q_input = (QuantizedArray(self.n_bits_inputs, val) for val in calibration_data)
        quantized_module.set_inputs_quantization_parameters(*q_input)
        return quantized_module
