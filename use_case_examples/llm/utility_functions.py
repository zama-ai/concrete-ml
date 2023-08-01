from collections import OrderedDict

import numpy as np
import torch

from concrete import fhe


def relu(x):
    """Define the ReLU function."""
    return np.maximum(0, x)


def max_fhe_relu(q_x, axis=-1, keepdims=True):
    """Find the maximum value in FHE along a specified axis."""
    with fhe.tag("Max"):
        # Normalize axis to handle negative values
        axis = axis if axis >= 0 else q_x.ndim + axis

        # Initialize result as the first slice along the specified axis
        slicer = [slice(None)] * q_x.ndim
        slicer[axis] = 0
        result = q_x[tuple(slicer)]

        # Iterate over the specified axis
        for i in range(1, q_x.shape[axis]):
            slicer[axis] = i
            next_element = q_x[tuple(slicer)]
            result = result + relu(next_element - result)

        # Keep the same dimensions as the input if keepdims is True
        if keepdims:
            shape = list(result.shape)
            shape.insert(axis, 1)
            result = result.reshape(tuple(shape))

    return result


def simple_slice(array, indices, axis):
    # this does the same as np.take() except only supports simple slicing, not
    # advanced indexing, and thus is much faster
    sl = [slice(None)] * array.ndim
    sl[axis] = indices
    return array[tuple(sl)]


def enc_split(array, n, axis):
    n_total = array.shape[axis]

    assert (
        n_total % n == 0
    ), f"array of shape {array.shape} cannot be split into {n} sub-arrays along axis {axis}"

    section = n_total // n

    split_arrays = ()
    for i in range(n):
        split_array = simple_slice(
            array=array, indices=slice(i * section, (i + 1) * section), axis=axis
        )
        split_arrays += (split_array,)

    return split_arrays


def slice_tensor(tensor, dim=0, indices=None):
    if tensor is None or indices is None:
        return tensor

    if isinstance(indices, int):
        sliced_tensor = tensor.select(dim, indices)
    else:
        sliced_tensor = tensor.index_select(dim, torch.tensor(indices).flatten())

    return sliced_tensor


def slice_ordered_dict(odict, dim=0, indices=None):
    return OrderedDict((k, slice_tensor(v, dim=dim, indices=indices)) for k, v in odict.items())
