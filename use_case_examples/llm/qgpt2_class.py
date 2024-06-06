from typing import Any, Dict, Optional

import numpy as np
import torch
from concrete.fhe import GraphProcessor
from concrete.fhe.compilation import Circuit, Configuration
from concrete.fhe.tracing import Tracer
from load_huggingface import get_gpt2_model
from preprocessor import InsertRounding
from quant_framework import DualArray, Quantizer
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from concrete import fhe


def quantize_dict_of_tensors(
    dict_of_tensors: Dict[Any, torch.Tensor], n_bits: int
) -> Dict[Any, DualArray]:
    """Convert a dict of torch tensors into a dict of DualArray.

    Args:
        dict_of_tensors (Dict[Any, torch.Tensor]): The dictionary to quantize.
        n_bits (int): The number of bits to consider for quantizing the tensors.

    Returns:
        q_dict (Dict[Any, DualArray]): The quantized dictionary.
    """
    q_dict = {
        key: DualArray(float_array=value.detach().cpu().numpy(), n_bits=n_bits)
        for key, value in dict_of_tensors.items()
    }
    return q_dict


class QuantizedModel:
    """Base class for quantized models.

    A QuantizedModel works along an associated Quantizer. This object is primarily used to store
    all scales and zero points in a dictionary. Each one of these quantization parameters are
    tied to a specific quantized operator thanks to their unique key. In order to compute and store
    them, a first calibration pass is done in float using an inputset. They are then re-used during
    FHE computations to properly quantize and de-quantize the values.
    """

    def __init__(self, n_bits: int):
        """Initialize the model using a quantizer.

        Args:
            n_bits (int): The number of bits to use to initialize the quantizer.
        """
        self.quantizer = Quantizer(n_bits=n_bits)
        self.x_calib: Optional[torch.Tensor] = None

    def finalize(self, x: DualArray):
        """Finalize the output value.

        If the DualArray's integer array is a Tracer, an object used during compilation, return it
        as is. Else, return the DualArray. This is called at the end of the run_numpy method because
        the compiler can only consider Tracer objects or Numpy arrays as input and outputs.

        Args:
            x (DualArray): The value to consider.

        Returns:
            Union[Tracer, DualArray]: The finalized value.

        """
        if isinstance(x.int_array, Tracer):
            return x.int_array
        else:
            return x

    def run_torch(self, inputs: torch.Tensor, fhe: str = "disable", true_float: bool = False):
        """Run the quantized operators, with additional pre and post-processing steps.

        This method is used to take and output torch tensors with floating points.

        Args:
            inputs (torch.Tensor): The input values to consider, in floating points.
            fhe (str): The FHE mode to consider, either "disable", "simulate" or "execute". Default
                to "disable".
            true_float (bool): If the FHE mode is set to "disable", indicate if the operations
                should be in floating points instead of being quantized. Default to False.

        Returns:
            torch.Tensor: The output values, in floating points.
        """

        # Convert the torch tensor to a numpy array
        inputs = inputs.detach().cpu().numpy()

        # Store the inputs as the calibration values. This is done in order to be able to easily
        # compile the model without having to manually extract the model's intermediary hidden
        # states. More importantly, these values are used to convert the quantized inputs from the
        # run_numpy method into their DualArray equivalent, as the compiler only accepts Numpy
        # arrays
        self.x_calib = inputs

        # Quantize the inputs
        q_inputs = self.quantizer.quantize(inputs, key="inputs_quant")

        # If the FHE mode is set to disable, we only need to run the quantized operators in the
        # clear and dequantize
        if fhe == "disable":
            q_y = self.run_numpy(q_inputs)

            if true_float:
                # Directly returning the output DualArray's floating points does not propagate the
                # quantization parameters. Therefore, these values are the result of float-only
                # computations
                y = q_y.float_array

            else:
                # De-quantizing the output DualArray propagates the quantization parameters. These
                # values should represent the expected values from FHE computations as they are the
                # result of quantized-only computations
                y = q_y.dequantize(key="y_dequant").float_array

        # Else, the FHE circuit, built thanks to the compilation step, needs to be called
        else:
            assert (
                self.circuit is not None
            ), "Module is not compiled. Please run `compile` on a representative inputset."

            # Batched operations is not yet handled by Concrete Python and inputs need to be
            # processed one by one
            y_all = []
            for q_x in q_inputs:

                # The circuit is expecting an input with a batch size of 1 in the first axis
                q_x = np.expand_dims(q_x, axis=0)

                if fhe == "simulate":
                    q_y = self.circuit.simulate(q_x)

                elif fhe == "execute":
                    q_y = self.circuit.encrypt_run_decrypt(q_x)

                else:
                    raise ValueError(
                        "Parameter 'fhe' can only be 'disable', 'simulate' or 'execute'"
                    )

                # The quantizer needs to be directly called in order to de-quantize the circuit's
                # output, as they are here stored in a Numpy array instead of a DualArray object
                y_all.append(self.quantizer.dequantize(q_y, key="y_dequant"))

            y = np.concatenate(y_all)

        # Return the values in a torch tensor, in floating points
        return torch.from_numpy(y).type(torch.float32)

    def run_numpy(self, q_inputs: np.ndarray) -> np.ndarray:
        """Run the quantized operators that will eb converted to FHE.

        Args:
            q_inputs (np.ndarray): The quantized inputs.

        Returns:
            np.ndarray: The quantized outputs.
        """
        raise NotImplementedError("This method must be implemented by subclasses.")

    def compile(
        self,
        configuration: Optional[Configuration] = None,
        msbs_round: Optional[int] = None,
        rounding_kwargs: Optional[Dict] = None,
    ) -> Circuit:
        """Compile the model using the stored calibration data.

        For now, the model can only be compiled on a batch made of a single input.

        Args:
            configuration (Optional[Configuration]): The configuration to use during compilation.
                Default to None.
            msbs_round (Optional[int]): msbs to keep after rounding
            rounding_kwargs (Optional[Dict]): optional keyword arguments of `InsertRounding`

        Returns:
            Circuit: The underlying FHE circuit.
        """
        assert self.x_calib is not None, "Module is not calibrated."

        # Quantize the calibration data
        q_inputs = self.quantizer.quantize(self.x_calib, key="inputs_quant")

        # Create the inputset with batches of 1
        inputset = [np.expand_dims(q_x, axis=0) for q_x in q_inputs]

        # Instantiate the compiler
        compiler = fhe.Compiler(self.run_numpy, {"q_inputs": "encrypted"})

        # Handle rounding
        if configuration is None:
            configuration = Configuration()
        if msbs_round is None:
            assert rounding_kwargs is None
        if rounding_kwargs is None:
            rounding_kwargs = {}
        rounding_preprocessor = InsertRounding(msbs_round, **rounding_kwargs)
        assert isinstance(rounding_preprocessor, GraphProcessor)
        configuration.additional_pre_processors.append(rounding_preprocessor)

        # Compile the circuit on the calibration quantized data
        self.circuit = compiler.compile(
            inputset, configuration=configuration, compress_input_ciphertexts=True
        )

        # Print the maximum bit-width reached in the circuit
        print(
            f"Circuit compiled with at most {self.circuit.graph.maximum_integer_bit_width()} bits"
        )

        return self.circuit


class QGPT2(QuantizedModel):
    """Class that implements quantized operators needed in the GPT-2 implementation."""

    def __init__(self, n_bits: int, layer: int, n_bits_weights: Optional[int] = None):
        """Initialize the class using some number of bits used for quantization.

        Args:
            n_bits (int): The number of bits to use to quantize inputs and activations.
            layer (int): The index representing the GPT-2 layer to consider.
            n_bits_weights (Optional[int]): The number of bits to use to quantize the weights. If
                None, n_bits will be used. Default to None.
        """
        super().__init__(n_bits=n_bits)
        self.circuit = None
        self.layer = layer

        # Load the model in order to retrieve GPT-2's weights and hyper-parameters
        self.float_torch_model = get_gpt2_model("gpt2_model")
        self.hyper_params = self.float_torch_model.config.to_dict()
        self.weights = dict(self.float_torch_model.state_dict())

        # Quantize the weights using DualArray instances
        self.q_weights = quantize_dict_of_tensors(
            self.weights, n_bits_weights if n_bits_weights is not None else n_bits
        )

    @property
    def config(self) -> GPT2Config:
        """Get GPT-2's configuration.

        Returns:
            GPT2Config: GPT-2's configuration.
        """
        return self.float_torch_model.config

    def softmax(self, q_x: DualArray):
        """Compute the softmax function, with quantized values.

        Args:
            q_x (DualArray): The quantized values to consider.

        Returns:
            q_x_softmax (DualArray): The quantized outputs.
        """

        # Compute the max value for each sequence
        q_x_max = q_x.max(axis=-1, keepdims=True, key="max")

        # Subtract max for numerical stability
        q_x_minus_max = q_x.sub(q_x_max, key="sub_max", requant=False)

        # Apply the exponential
        x_exp = q_x_minus_max.exp(key="exp")

        # Compute the sum along the sequence axis
        q_x_exp_sum = x_exp.sum("sum", axis=-1, keepdims=True)

        # Compute the inverse of the sum
        x_inverse_exp_sum = q_x_exp_sum.rtruediv(1, key="rtruediv")

        # Compute the final softmax values
        q_x_softmax = x_exp.mul(x_inverse_exp_sum, key="enc_mul")

        return q_x_softmax

    def attention(self, q_q: DualArray, q_k: DualArray, q_v: DualArray):
        """Attention mechanism as defined in transformers, with quantized values.

        Args:
            q_q (DualArray): The quantized query projections to consider.
            q_k (DualArray): The quantized key projections to consider.
            q_v (DualArray): The quantized value projections to consider.
        """

        # Re-quantize for precision stability. Another possibility could be to use the rounding
        # feature instead
        # q_q, q_k, q_v are expected to have shape (n_batch, n_head, n_seq, n_embed // n_head)
        q_q = q_q.requant(key="q_q")
        q_k = q_k.requant(key="q_k")
        q_v = q_v.requant(key="q_v")

        # Compute scores by computing the dot product of queries and keys
        # q_scores is expected to have shape (n_batch, n_head, n_seq, n_seq)
        q_scores = q_q.matmul(q_k.transpose(axes=(0, 1, 3, 2), key="transpose_key"), key="qk^T")

        # Scale by square root of the key's dimension for stability
        dk = q_k.shape[-1]
        scaled_scores = q_scores.truediv(np.sqrt(dk), "truediv")

        # Create a causal mask using an upper triangular matrix made of ones
        seq_length = q_k.shape[2]
        causal_mask = [[1 if j <= i else 0 for j in range(seq_length)] for i in range(seq_length)]

        # Normally, the mask_value is set to -inf. However, this would make the quantization process
        # unreliable. Therefore, we consider the minimum value found in the array
        if not isinstance(scaled_scores.float_array, Tracer):
            self.mask_value = scaled_scores.float_array.min()

        # Apply the causal mask mechanism
        scaled_scores.float_array = np.where(
            causal_mask, scaled_scores.float_array, self.mask_value
        )

        # Apply the softmax to get attention weights and re-quantize for precision stability
        q_attention_weights = self.softmax(scaled_scores)
        q_attention_weights = q_attention_weights.requant(key="q_attention_weights_requant")

        # Compute the output values by projecting the weights on the value matrix
        q_output = q_attention_weights.matmul(q_v, key="matmul_attention_values")

        return q_output
