"""Post Training Quantization methods."""

from typing import Dict, Iterable, Set, Tuple, Union, cast

import numpy
import onnx
from onnx import numpy_helper

from ..common.debugging import assert_true
from ..onnx.onnx_utils import ONNX_OPS_TO_NUMPY_IMPL, get_attribute
from ..torch.numpy_module import NumpyModule
from .quantized_array import QuantizedArray
from .quantized_module import QuantizedModule
from .quantized_ops import ONNX_OPS_TO_QUANTIZED_IMPL, QuantizedOp


class PostTrainingAffineQuantization:
    """Post-training Affine Quantization.

    Create the quantized version of the passed numpy module.

    Args:
        n_bits (int):                   Number of bits to quantize the model. Currently this
                                        n_bits will be used for all activation/inputs/weights.
        numpy_model (NumpyModule):   Model in numpy.
        is_signed:                      Whether the weights of the layers can be signed.
                                        Currently, only the weights can be signed.

    Returns:
        QuantizedModule: A quantized version of the numpy model.
    """

    quant_ops_dict: Dict[str, Tuple[Tuple[str, ...], QuantizedOp]]
    n_bits: int
    quant_params: Dict[str, QuantizedArray]
    numpy_model: NumpyModule
    is_signed: bool

    def __init__(self, n_bits: int, numpy_model: NumpyModule, is_signed: bool = False):
        self.quant_ops_dict = {}
        self.n_bits = n_bits
        self.quant_params = {}
        self.numpy_model = numpy_model
        self.is_signed = is_signed

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
                QuantizedArray(
                    self.n_bits, numpy_helper.to_array(onnx_init), is_signed=True, is_symmetric=True
                ),
            )
            if numpy_helper.to_array(onnx_init).min() < 0
            else (
                onnx_init.name,
                QuantizedArray(
                    self.n_bits,
                    numpy_helper.to_array(onnx_init),
                    is_signed=False,
                    is_symmetric=True,
                ),
            )
            for onnx_init in inits
        )

    def _quantize_layers(self, *input_calibration_data: numpy.ndarray):
        """Compute all parameters for the static post-training quantization.

        Does a forward pass over a batch of data and compute all
        quantization parameters for activations and layers.

        Args:
            *input_calibration_data (numpy.ndarray): Data that will be used to compute the bounds,
                scales and zero point values for every quantized object.
        """

        graph = self.numpy_model.onnx_model.graph

        node_results: Dict[str, Union[numpy.ndarray, QuantizedArray]] = dict(
            {
                graph_input.name: input_value
                for graph_input, input_value in zip(graph.input, input_calibration_data)
            },
            **self.quant_params,
        )

        constants: Set[str] = set(self.quant_params.keys())

        for node in graph.node:
            op_type = node.op_type
            attributes = {attribute.name: get_attribute(attribute) for attribute in node.attribute}
            # For now only single output nodes
            assert_true(len(node.output) == 1)
            output_name = node.output[0]
            if op_type == "Constant":
                node_results[output_name] = QuantizedArray(
                    self.n_bits, ONNX_OPS_TO_NUMPY_IMPL["Constant"](**attributes)[0], self.is_signed
                )
                constants.add(output_name)
                continue

            curr_inputs = {input_name: node_results[input_name] for input_name in node.input}
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
                quantized_op_instance = quantized_op_class(
                    self.n_bits, curr_cst_inputs, **attributes
                )
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
                q_node_output = QuantizedArray(self.n_bits, node_output[0], self.is_signed)
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
        q_calibration_data = (QuantizedArray(self.n_bits, data) for data in calibration_data)
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
            (graph_input.name for graph_input in self.numpy_model.onnx_model.graph.input),
            (graph_output.name for graph_output in self.numpy_model.onnx_model.graph.output),
            self.quant_ops_dict,
        )
        q_input = (QuantizedArray(self.n_bits, val) for val in calibration_data)
        quantized_module.set_inputs_quantization_parameters(*q_input)
        return quantized_module
