from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
from concrete.fhe.tracing import Tracer
from utility_functions import enc_split, max_fhe_relu, simple_slice

EPSILON = 2**-11

import numpy as np


def compute_scale_zp_from_float_int(
    float_array: np.ndarray, int_array: np.ndarray, is_symmetric: bool = True
) -> Tuple[float, Union[float, int]]:
    """Compute the scale and zero point based on floats and their associated quantized values.

    Args:
        float_array (np.ndarray): The floating point values.
        int_array (np.ndarray): The quantized values associated with the float values.
        is_symmetric (bool): If the quantization should be symmetric. Default to True.

    Returns:
        (scale, zp) (Tuple[float, Union[float, int]]): The values' scale and zero point.
    """

    # Retrieve the inputs' min and max values
    float_array_min, float_array_max = np.min(float_array), np.max(float_array)
    int_array_min, int_array_max = np.min(int_array), np.max(int_array)

    # If the quantized values' min and max are the same, scale is 1 and zero point is 0
    if int_array_min == int_array_max:
        scale = 1
        zp = 0

    else:
        # With symmetric quantization, the zero point is set to 0
        if is_symmetric:
            scale = (float_array_max - float_array_min) / (int_array_max - int_array_min)
            zp = 0

        else:
            scale = (float_array_max - float_array_min) / (int_array_max - int_array_min)
            zp = (-float_array_max * int_array_min + float_array_min * int_array_max) / (
                float_array_min - float_array_max
            )

    return scale, zp


def compute_scale_zp_from_n_bits(
    float_array: np.ndarray, n_bits: int, is_symmetric: bool = True
) -> Tuple[float, Union[float, int]]:
    """Compute the scale and zero point based on floats the number of bits to use to quantize.

    Args:
        float_array (np.ndarray): The floating point values.
        n_bits (int): The number of bits to use to quantize the floating points.
        is_symmetric (bool): If the quantization should be symmetric. Default to True.

    Returns:
        (scale, zp) (Tuple[float, Union[float, int]]): The values' scale and zero point.
    """

    if not is_symmetric:
        raise NotImplementedError("is_symmetric = False is not yet fully supported.")

    # Retrieve the inputs' min and max values
    min_val = np.min(float_array)
    max_val = np.max(float_array)

    # If the values' min and max are the same, scale is 1 and zero point is 0
    if min_val == max_val:
        scale = 1
        zero_point = 0

    # Else, apply symmetric quantization over n_bits
    else:
        max_abs_val = np.maximum(abs(min_val), abs(max_val))
        scale = max_abs_val / (2 ** (n_bits - 1) - 1)
        zero_point = 0

    return scale, zero_point


class Quantizer:
    """
    Quantizer class that provides methods to handle any quantized operators.
    """

    def __init__(self, n_bits: int = 8):
        """Initialize with the number of bits to use in quantization.

        A Quantizer instance is primarily used to store all scales and zero points in a dictionary.
        Each one of these quantization parameters are tied to a specific quantized operator thanks
        to their unique key. In order to compute and store them, a first calibration pass is done
        in float using an inputset. They are then re-used during FHE computations to properly
        quantize and de-quantize the values.

        Args:
            n_bits (int): The number of bits to use for quantization.
        """
        self.n_bits = n_bits
        self.scale_dict = {}

    def quantize(
        self, float_array: np.ndarray, key: Optional[str] = None, is_symmetric: bool = True
    ) -> np.ndarray:
        """Quantize a floating point array.

        Args:
            float_array (np.ndarray): The floating point values.
            key (Optional[str]): The key representing the float_array's scale and zero_point if
                already known. If None, the floating points are quantized over n_bits. Default to
                None.
            is_symmetric (bool): If the quantization should be symmetric. Default to True.

        Returns:
            np.ndarray: The quantized values.
        """

        # Retrieve or compute the scale and zero point
        scale_zp = (
            self.scale_dict[key]
            if key in self.scale_dict
            else compute_scale_zp_from_n_bits(float_array, self.n_bits, is_symmetric)
        )
        self.scale_dict[key] = scale_zp

        # Quantize the values
        return np.rint((float_array / scale_zp[0]) + scale_zp[1]).astype(np.int64)

    def dequantize(
        self,
        int_array: np.ndarray,
        float_array: Optional[np.ndarray] = None,
        key: Optional[str] = None,
        is_symmetric: bool = True,
    ) -> np.ndarray:
        """De-quantize an integer array.

        Args:
            int_array (np.ndarray): The quantized values.
            float_array (Optional[np.ndarray]): The floating point values associated with the
                quantized values. Default to None.
            key (Optional[str]): The key representing the float_array's scale and zero_point if
                already known. If None, the scale and zero point are computed using the integer and
                    associated float arrays. Default to None.
            is_symmetric (bool): If the quantization should be symmetric. Default to True.

        Returns:
            np.ndarray: The de-quantized values.

        Raises:
            ValueError: If no scale and zero point associated to the input values exist and one of
                the integer or float arrays were not provided
        """
        if key not in self.scale_dict and (float_array is None or int_array is None):
            raise ValueError("'float_array' and 'int_array' must be provided.")

        # If the key does not exist yet, compute the scale and zero point using the int and float
        # arrays
        elif key not in self.scale_dict:
            self.scale_dict[key] = compute_scale_zp_from_float_int(
                float_array=float_array, int_array=int_array, is_symmetric=is_symmetric
            )

        # Dequantize the values
        return (int_array - self.scale_dict[key][1]) * self.scale_dict[key][0]


class DualArray:
    """
    A dual representation array, propagating both the floating points and their quantized versions.
    """

    def __init__(
        self,
        float_array: Optional[np.ndarray] = None,
        int_array: Optional[np.ndarray] = None,
        quantizer: Optional[Quantizer] = None,
        n_bits: Optional[int] = None,
    ):
        """Initialize with a floating point array, integer array and a quantizer.

        Args:
            float_array (Optional[np.ndarray]): Some floating point values. Default to None.
            int_array (Optional[np.ndarray]): Some quantized values. Default to None.
            quantizer (Optional[Quantizer]): A quantizer. Default to None.
            n_bits (Optional[int]): The number of bits to use for quantization if quantizer is None.
                Default to None.
        """
        self.float_array = float_array
        self.int_array = int_array
        self.quantizer = quantizer if quantizer is not None else Quantizer(n_bits=n_bits)

    @property
    def shape(self) -> Optional[Union[int, Tuple[int]]]:
        """Return the shape of the DualArray.

        Returns:
            Optional[Union[int, Tuple[int]]]: The DualArray's shape
        """
        if self.float_array is not None:
            return self.float_array.shape

        elif self.int_array is not None:
            return self.int_array.shape

        else:
            return None

    def _ensure_quantized(self, key: str, is_symmetric: bool = True) -> np.ndarray:
        """Helper method to ensure the integer representation is available."""
        if self.int_array is None:
            return self.quantizer.quantize(self.float_array, key=key, is_symmetric=is_symmetric)
        else:
            return self.int_array

    def _ensure_dequantized(self, key: str, is_symmetric: bool = True) -> np.ndarray:
        """Helper method to ensure the integer representation is available."""
        if self.int_array is not None:
            return self.quantizer.dequantize(
                self.int_array, self.float_array, key=key, is_symmetric=is_symmetric
            )
        else:
            return self.float_array

    def dequantize(self, key: str) -> DualArray:
        """Open the integer array to floating point using de-quantization."""
        if self.int_array is not None:
            float_array = self.quantizer.dequantize(self.int_array, self.float_array, key=key)
            return DualArray(float_array=float_array, int_array=None, quantizer=self.quantizer)
        else:
            return self

    def quantize(self, key: str) -> DualArray:
        """Close the floating point array to integer using quantization."""
        if self.float_array is not None:
            int_array = self.quantizer.quantize(self.float_array, key=key)
            return DualArray(
                float_array=self.float_array, int_array=int_array, quantizer=self.quantizer
            )
        else:
            return self

    def requant(self, key: str) -> DualArray:
        """Re-quantize the integer values over n_bits."""
        float_array = self.quantizer.dequantize(
            self.int_array, self.float_array, key=f"dequant_{key}"
        )
        int_array = self.quantizer.quantize(float_array, key=f"quant_{key}")
        return DualArray(
            float_array=self.float_array, int_array=int_array, quantizer=self.quantizer
        )

    def exp(self, key: str) -> DualArray:
        """Compute the exponential."""
        float_array = self._ensure_dequantized(key=key)
        return DualArray(
            float_array=np.exp(float_array),
            int_array=None,
            quantizer=self.quantizer,
        )

    def sum(self, key: str, axis: Optional[int] = None, keepdims: bool = False) -> DualArray:
        """Compute the sum along the specified axis."""
        int_array = self._ensure_quantized(key=key)
        float_array = (
            np.sum(self.float_array, axis=axis, keepdims=keepdims)
            if self.float_array is not None and not isinstance(self.float_array, Tracer)
            else None
        )
        int_array = np.sum(int_array, axis=axis, keepdims=keepdims)
        return DualArray(float_array=float_array, int_array=int_array, quantizer=self.quantizer)

    def mul(self, other: DualArray, key: str) -> DualArray:
        """Compute the multiplication."""
        self_int_array = self._ensure_quantized(key=f"{key}_self")
        other_int_array = other._ensure_quantized(key=f"{key}_other")
        float_array = (
            self.float_array * other.float_array
            if self.float_array is not None and not isinstance(self.float_array, Tracer)
            else None
        )
        int_array = self_int_array * other_int_array
        return DualArray(float_array=float_array, int_array=int_array, quantizer=self.quantizer)

    def matmul(self, other: DualArray, key: str) -> DualArray:
        """Compute the matrix multiplication."""
        self_int_array = self._ensure_quantized(key=f"{key}_self")
        other_int_array = other._ensure_quantized(key=f"{key}_other")
        float_array = (
            self.float_array @ other.float_array
            if self.float_array is not None and not isinstance(self.float_array, Tracer)
            else None
        )
        return DualArray(
            float_array=float_array,
            int_array=self_int_array @ other_int_array,
            quantizer=self.quantizer,
        )

    def truediv(self, denominator: Union[int, float], key: str) -> DualArray:
        """Compute the true division."""
        float_array = self._ensure_dequantized(key=key)
        return DualArray(
            float_array=float_array / denominator, int_array=None, quantizer=self.quantizer
        )

    def rtruediv(self, numerator: Union[int, float], key: str) -> DualArray:
        """Compute the reverse true division."""
        float_array = self._ensure_dequantized(key=key)
        return DualArray(
            float_array=numerator / float_array, int_array=None, quantizer=self.quantizer
        )

    def transpose(self, axes: Union[Tuple[int], List[int]], key: str) -> DualArray:
        """Transpose the arrays using the given axes."""
        int_array = self._ensure_quantized(key=key)
        float_array = (
            np.transpose(self.float_array, axes=axes)
            if self.float_array is not None and not isinstance(self.float_array, Tracer)
            else None
        )
        int_array = np.transpose(int_array, axes=axes)
        return DualArray(float_array=float_array, int_array=int_array, quantizer=self.quantizer)

    def max(self, key, axis: Optional[int] = None, keepdims: bool = None) -> DualArray:
        """Compute the max."""
        int_array = self._ensure_quantized(key=key)
        float_array = (
            np.max(self.float_array, axis=axis, keepdims=keepdims)
            if self.float_array is not None and not isinstance(self.float_array, Tracer)
            else None
        )
        int_array = max_fhe_relu(int_array, axis=axis, keepdims=keepdims)
        return DualArray(float_array=float_array, int_array=int_array, quantizer=self.quantizer)

    def sqrt(self, key: str) -> DualArray:
        """Compute the square root"""
        float_array = self._ensure_dequantized(key=key)
        return DualArray(
            float_array=np.sqrt(float_array),
            int_array=None,
            quantizer=self.quantizer,
        )

    def _sub_add(self, other: DualArray, factor: int, key: str, requant: bool) -> DualArray:
        """Compute the addition or the subtraction, with a possible re-quantization step."""
        if requant:
            # We de-quantize both arrays if they aren't already
            self_float_array = self._ensure_dequantized(key=f"{key}_sub_add_self")
            other_float_array = other._ensure_dequantized(key=f"{key}_sub_add_other")

            if (
                not isinstance(self.int_array, Tracer)
                and not isinstance(self.float_array, Tracer)
                and not f"{key}_sub_add_self" in self.quantizer.scale_dict
            ):
                # Combine both float array for quantization
                self_orig_shape = self_float_array.shape
                other_orig_shape = other_float_array.shape
                combined_array = np.concatenate(
                    [self_float_array.ravel(), other_float_array.ravel()]
                )

                # Requantize both array together
                combined_int_array = self.quantizer.quantize(
                    combined_array, key=f"{key}_sub_add_requant"
                )

                # Split array back to their object
                self_int_array, other_int_array = np.split(
                    combined_int_array, [np.prod(self_orig_shape)]
                )

                # Reshape the quant arrays back to their original shapes
                self_int_array = self_int_array.reshape(self_orig_shape)
                other_int_array = other_int_array.reshape(other_orig_shape)
            else:
                self_int_array = self.quantizer.quantize(
                    self_float_array, key=f"{key}_sub_add_requant"
                )
                other_int_array = self.quantizer.quantize(
                    other_float_array, key=f"{key}_sub_add_requant"
                )
        else:
            self_int_array = self._ensure_quantized(key=f"{key}_quant_self")
            other_int_array = other._ensure_quantized(key=f"{key}_quant_other")

        self_float_array = (
            self.float_array + (factor * other.float_array)
            if (
                not isinstance(self.float_array, Tracer)
                and self.float_array is not None
                and other.float_array is not None
            )
            else None
        )
        return DualArray(
            float_array=self_float_array,
            int_array=self_int_array + (factor * other_int_array),
            quantizer=self.quantizer,
        )

    def add(self, other: DualArray, key: str, requant: bool = True) -> DualArray:
        """Compute the addition."""
        return self._sub_add(other=other, factor=1, key=key, requant=requant)

    def sub(self, other: DualArray, key: str, requant: bool = True) -> DualArray:
        """Compute the subtraction."""
        return self._sub_add(other=other, factor=-1, key=key, requant=requant)

    def linear(self, weight: DualArray, bias: DualArray, key: str) -> DualArray:
        """Compute a linear operation with some weight and bias values."""
        assert bias is not None, "None bias is not supported in the linear op, use matmul instead."

        x_matmul = self.matmul(weight, key=f"linear_matmul_{key}")

        x_linear = x_matmul.add(bias, key=f"linear_add_{key}")

        return x_linear

    # Concrete-Python does not support numpy.array_split and numpy.take so we need to build a custom
    # split method instead
    # FIXME: https://github.com/zama-ai/concrete-internal/issues/329
    def enc_split(self, n: int, axis: int, key: str) -> Tuple[DualArray]:
        """Split the arrays in n parts along a given axis."""
        self_int_array = self._ensure_quantized(key=f"{key}_self")

        splitted_float_array = enc_split(self.float_array, n=n, axis=axis)
        splitted_int_array = enc_split(self_int_array, n=n, axis=axis)

        return tuple(
            DualArray(
                float_array=i_float_array,
                int_array=i_int_array,
                quantizer=self.quantizer,
            )
            for i_float_array, i_int_array in zip(splitted_float_array, splitted_int_array)
        )

    def reshape(self, newshape: Union[int, Tuple[int]], key: str) -> DualArray:
        """Reshape the arrays into the given shape."""
        self_int_array = self._ensure_quantized(key=f"{key}_self")

        reshaped_float_array = (
            self.float_array.reshape(newshape)
            if self.float_array is not None and not isinstance(self.float_array, Tracer)
            else None
        )
        reshaped_int_array = self_int_array.reshape(newshape)

        return DualArray(
            float_array=reshaped_float_array,
            int_array=reshaped_int_array,
            quantizer=self.quantizer,
        )

    def expand_dims(self, key: str, axis: int = 0) -> DualArray:
        """Add a dimension in the arrays along the given axis."""
        self_int_array = self._ensure_quantized(key=f"{key}_self")

        return DualArray(
            float_array=np.expand_dims(self.float_array, axis=axis),
            int_array=np.expand_dims(self_int_array, axis=axis),
            quantizer=self.quantizer,
        )

    def slice_array(self, indices: List[List[int]], key: str, axis: int = 0) -> DualArray:
        """Slice the arrays using the given indices along the given axis."""
        self_int_array = self._ensure_quantized(key=f"{key}_self")

        indices = np.array(indices).flatten()

        return DualArray(
            float_array=simple_slice(self.float_array, indices=indices, axis=axis),
            int_array=simple_slice(self_int_array, indices=indices, axis=axis),
            quantizer=self.quantizer,
        )
