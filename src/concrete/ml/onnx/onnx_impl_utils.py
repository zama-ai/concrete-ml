"""Utility functions for onnx operator implementations."""

from typing import Tuple, Union

import numpy
from concrete.fhe import conv as cnp_conv
from concrete.fhe import ones as cnp_ones

from ..common.debugging import assert_true


def numpy_onnx_pad(
    x: numpy.ndarray,
    pads: Tuple[int, ...],
    pad_value: Union[float, int, numpy.ndarray] = 0,
    int_only: bool = False,
) -> numpy.ndarray:
    """Pad a tensor according to ONNX spec, using an optional custom pad value.

    Args:
        x (numpy.ndarray): input tensor to pad
        pads (List[int]): padding values according to ONNX spec
        pad_value (Optional[Union[float, int]]): value used to fill in padding, default 0
        int_only (bool): set to True to generate integer only code with Concrete

    Returns:
        res(numpy.ndarray): the input tensor with padding applied
    """

    x_pad = x
    if numpy.any(numpy.asarray(pads) > 0):
        # Weight shape is O x I x dim1 x dim2 x .. x dimN
        # of which dim1 x dim2 x .. x dimN are the dimensions of the kernel (self.kernel_shape)
        # I is the number of input feature maps, O is the number of output feature maps

        # Need to get the number of kernel dimensions
        ndim = x.ndim - 2

        # Compute padded shape, keep batch size and channels number
        # Add the pads to the other dimensions.
        # Pads are in the form of
        # dim1_start, dim2_start, ..., dimN_start, dim1_end, dim2_end, ... dimN_end
        padded_shape = [x.shape[0], x.shape[1]]
        padded_shape += [x.shape[i + 2] + pads[i] + pads[ndim + i] for i in range(ndim)]

        # Initialize a padded version of the input, setting
        # the values on the edges to the input zero_point, which corresponds
        # to the real-axis 0
        if int_only:
            # Work in integer Concrete mode
            x_pad = cnp_ones(tuple(padded_shape)) * numpy.int64(pad_value)
        else:
            # Floating point mode
            x_pad = numpy.ones(padded_shape, dtype=numpy.float32) * pad_value

        # Create the indices for slice assignment, copy all on batch size and channels dimension
        indices = [slice(None), slice(None)] + [
            slice(pads[i], x_pad.shape[i + 2] - pads[ndim + i]) for i in range(ndim)
        ]

        x_pad[tuple(indices)] = x

    return x_pad


def compute_conv_output_dims(
    input_shape: Tuple[int, ...],
    kernel_shape: Tuple[int, ...],
    pads: Tuple[int, ...],
    strides: Tuple[int, ...],
    ceil_mode: int,
) -> Tuple[int, ...]:
    """Compute the output shape of a pool or conv operation.

    See https://pytorch.org/docs/stable/generated/torch.nn.AvgPool2d.html for details
    on the computation of the output shape.

    Args:
        input_shape (Tuple[int, ...]): shape of the input to be padded as N x C x H x W
        kernel_shape (Tuple[int, ...]): shape of the conv or pool kernel, as Kh x Kw (or n-d)
        pads (Tuple[int, ...]): padding values following ONNX spec:
            dim1_start, dim2_start, .. dimN_start, dim1_end, dim2_end, ... dimN_end
            where in the 2-d case dim1 is H, dim2 is W
        strides (Tuple[int, ...]): strides for each dimension
        ceil_mode (int): set to 1 to use the `ceil` function to compute the output shape, as
            described in the PyTorch doc

    Returns:
        res (Tuple[int, ...]): shape of the output of a conv or pool operator with given parameters
    """

    assert_true(ceil_mode in {0, 1})

    height_out = (input_shape[2] + pads[0] + pads[2] - kernel_shape[0]) / strides[0] + 1
    width_out = (input_shape[3] + pads[1] + pads[3] - kernel_shape[1]) / strides[1] + 1

    if ceil_mode == 0:
        height_out = numpy.floor(height_out)
        width_out = numpy.floor(width_out)
    else:
        height_out = numpy.ceil(height_out)
        width_out = numpy.ceil(width_out)

    height_out = int(height_out)
    width_out = int(width_out)

    return (input_shape[0], input_shape[1], height_out, width_out)


def compute_onnx_pool_padding(
    input_shape: Tuple[int, ...],
    kernel_shape: Tuple[int, ...],
    pads: Tuple[int, ...],
    strides: Tuple[int, ...],
    ceil_mode: int,
) -> Tuple[int, ...]:
    """Compute any additional padding needed to compute pooling layers.

    The ONNX standard uses ceil_mode=1 to match TensorFlow style pooling output computation.
    In this setting, the kernel can be placed at a valid position even though it contains values
    outside of the input shape including padding. The ceil_mode parameter controls whether
    this mode is enabled. If the mode is not enabled, the output shape follows PyTorch rules.

    Args:
        input_shape (Tuple[int, ...]): shape of the input to be padded as N x C x H x W
        kernel_shape (Tuple[int, ...]): shape of the conv or pool kernel, as Kh x Kw (or n-d)
        pads (Tuple[int, ...]): padding values following ONNX spec:
            dim1_start, dim2_start, .. dimN_start, dim1_end, dim2_end, ... dimN_end
            where in the 2-d case dim1 is H, dim2 is W
        strides (Tuple[int, ...]): strides for each dimension
        ceil_mode (int): set to 1 to use the `ceil` function to compute the output shape, as
            described in the PyTorch doc

    Returns:
        res (Tuple[int, ...]): shape of the output of a conv or pool operator with given parameters
    """

    pads2 = list(pads)
    if ceil_mode == 1:
        # We will pad the input with additional rows to respect TensorFlow style
        # padding (ceil_mode == 1)

        # Compute the dimensions for floor/ceil output computation modes
        dims_floor = compute_conv_output_dims(input_shape, kernel_shape, pads, strides, 0)
        dims_ceil = compute_conv_output_dims(input_shape, kernel_shape, pads, strides, 1)

        # Compute the amount of additional padding necessary
        # The additional padding should be done down on Y and right on X

        pads2[2] += dims_ceil[2] - dims_floor[2]  # pad_y_end += diff_y
        pads2[3] += dims_ceil[3] - dims_floor[3]  # pad_x_end += diff_x

    return tuple(pads2)


def onnx_avgpool_compute_norm_const(
    input_shape: Tuple[int, ...],
    kernel_shape: Tuple[int, ...],
    pads: Tuple[int, ...],
    strides: Tuple[int, ...],
    ceil_mode: int,
) -> Union[numpy.ndarray, float]:
    """Compute the average pooling normalization constant.

    This constant can be a tensor of the same shape as the input or a scalar.

    Args:
        input_shape (Tuple[int, ...]): shape of the input to be padded as N x C x H x W
        kernel_shape (Tuple[int, ...]): shape of the conv or pool kernel, as Kh x Kw (or n-d)
        pads (Tuple[int, ...]): padding values following ONNX spec:
            dim1_start, dim2_start, .. dimN_start, dim1_end, dim2_end, ... dimN_end
            where in the 2-d case dim1 is H, dim2 is W
        strides (Tuple[int, ...]): strides for each dimension
        ceil_mode (int): set to 1 to use the `ceil` function to compute the output shape, as
            described in the PyTorch doc

    Returns:
        res (float): tensor or scalar, corresponding to normalization factors to apply for the
            average pool computation for each valid kernel position
    """
    # Handle the TensorFlow pooling mode
    if ceil_mode == 1:
        n_in_channels = input_shape[1]
        kernel = numpy.ones(
            (n_in_channels, 1, kernel_shape[0], kernel_shape[1]),
            dtype=numpy.int64,
        )

        # TensorFlow (and ONNX pool with ceil_mode==1) allow the kernel of the pooling op
        # to be placed in positions that include out-of-bounds indices.
        # For example an input of size 2 containing values V,
        # with padding P of 1 to the left and right:
        #      P V V P
        # The pooling of size 2 can be applied at positions: 0,1,2,3:
        #    (P+V)/2   (V+V)/2  (V+P)/2   P
        # Even though at position 3 it is out of bounds.

        # When the kernel is applied with out of bounds indices, these are ignored
        # and the averaging is done counting only the valid values (P or V) in its support

        # We thus need to find the number of valid indices for each kernel position

        # Compute the padded input tensor in Floor mode (PyTorch)
        pool_pads_floor = compute_onnx_pool_padding(input_shape, kernel_shape, pads, strides, 0)

        # Compute it again in TensorFlow mode
        pool_pads_ceil = compute_onnx_pool_padding(input_shape, kernel_shape, pads, strides, 1)

        # Create a tensor of ones for PyTorch mode and one of zeros for TF mode
        padded_flr = numpy_onnx_pad(numpy.ones(input_shape, dtype=numpy.int64), pool_pads_floor, 1)
        padded_ceil = numpy_onnx_pad(numpy.zeros(input_shape, dtype=numpy.int64), pool_pads_ceil, 0)

        # Initialize a final tensor that has 1s in valid indices and 0s in invalid ones
        padded_ceil[:, :, 0 : padded_flr.shape[2], 0 : padded_flr.shape[3]] = 1

        # Compute the sum of valid indices in each kernel position
        norm_const = cnp_conv(
            padded_ceil, kernel, None, [0, 0, 0, 0], strides, None, None, n_in_channels
        )
    else:
        # For the PyTorch mode, only positions with all valid indices are used so
        # the averaging is done over the number of cells in the kernel
        norm_const = numpy.prod(kernel_shape)

    return norm_const
