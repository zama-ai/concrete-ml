"""Test utils functions."""

import numpy
import pandas
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from concrete.ml.common.debugging.custom_assert import assert_true
from concrete.ml.common.utils import compute_bits_precision
from concrete.ml.pytest.torch_models import QuantCustomModel
from concrete.ml.pytest.utils import data_calibration_processing


def get_data(load_data):
    """Get classification data-set."""

    x, y = load_data(
        QuantCustomModel(n_bits=5, input_shape=6, hidden_shape=100, output_shape=3),
        **{
            "n_samples": 1000,
            "n_classes": 3,
            "n_features": 6,
            "n_informative": 4,
            "n_redundant": 2,
        }
    )

    return x, y


@pytest.mark.parametrize(
    "x, expected_n_bits", [(range(2**8), 8), (range(2**8 + 1), 9), (range(8), 3)]
)
def test_compute_bits_precision(x, expected_n_bits):
    """Test the function that computes the number of bits necessary to represent an array."""
    assert_true(compute_bits_precision(numpy.array(x)) == expected_n_bits)


@pytest.mark.parametrize("input_type", ["dataloader", "pandas", "list", "numpy", "torch"])
def test_data_processing_valid_input(input_type, load_data):
    """Check if the _update_attr method raises an exception when an undefined attribute is given."""

    x, y = get_data(load_data)

    if input_type.lower() == "dataloader":
        # Turn into `torch.utils.data.dataloader.DataLoader`
        x, y = torch.tensor(x), torch.tensor(y)
        x = DataLoader(TensorDataset(x, y), batch_size=32)
        y = None
    if input_type.lower() == "pandas":
        # Turn into Pandas
        x = pandas.DataFrame(x)
        y = pandas.Series(y) if y.ndim == 1 else pandas.DataFrame(y)
    elif input_type.lower() == "torch":
        # Turn into Torch
        x = torch.tensor(x)
        y = torch.tensor(y)
    elif input_type.lower() == "list":
        # Turn into List
        y = y.tolist()

    x_calib, y = data_calibration_processing(data=x, targets=y, n_sample=1)

    assert_true(isinstance(x_calib, (numpy.ndarray)))
    assert_true(isinstance(y, (numpy.ndarray)))


@pytest.mark.parametrize("input_type", ["tuple"])
def test_data_processing_invalid_input(input_type, load_data):
    """Check if the _update_attr method raises an exception when an undefined attribute is given."""

    x, y = get_data(load_data)

    if input_type.lower() == "tuple":
        x = tuple(x)
        y = tuple(y)

        with pytest.raises(TypeError, match="Only numpy arrays, torch tensors and .*"):
            _ = data_calibration_processing(data=x, targets=y, n_sample=1)
