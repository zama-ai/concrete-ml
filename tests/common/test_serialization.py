"""Test serialization.

Here we test the custom dump(s)/load(s) methods we have in Concrete ML.
In all tests we create an object, dump it to and load it again.
We then check some properties on the loaded object.
"""
import io
import json
import random
from typing import Tuple

import numpy
import onnx
import pytest
import torch
from numpy.random import RandomState

from concrete.ml.common.serialization.dumpers import dumps, dumps_random_state
from concrete.ml.common.serialization.encoder import CustomEncoder, dumps_onnx
from concrete.ml.common.serialization.loaders import loads, loads_onnx, loads_random_state
from concrete.ml.pytest.torch_models import SimpleNet
from concrete.ml.quantization.quantizers import (
    MinMaxQuantizationStats,
    QuantizationOptions,
    QuantizedArray,
    UniformQuantizationParameters,
    UniformQuantizer,
)


def is_random_state_state_equal(state_1: Tuple, state_2: Tuple) -> bool:
    """Check if some tuples are equal element-wise.

    If some elements are numpy.array we check that all values are equal.
    This is used to compare numpy.RandomState states.

    Args:
        state_1 (Tuple): a tuple.
        state_2 (Tuple): another tuple.

    Returns:
        bool: True if all values are equal, false otherwise.
    """
    for elt_1, elt_2 in zip(state_1, state_2):
        if isinstance(elt_1, numpy.ndarray):
            if (elt_1 != elt_2).any():
                return False
        elif elt_1 != elt_2:
            return False
    return True


def test_serialize_random_state():
    """Test serialization of random_state."""
    seed = random.randint(a=10, b=1000)
    random_state = RandomState(seed)
    serialized = dumps_random_state(random_state)
    loaded_random_state = loads_random_state(serialized_random_state=serialized)
    assert isinstance(loaded_random_state, RandomState)
    state_1, state_2 = loaded_random_state.get_state(), random_state.get_state()
    assert isinstance(state_1, tuple) and isinstance(state_2, tuple)
    assert is_random_state_state_equal(state_1, state_2)

    for random_state in [None, 42]:
        serialized = dumps_random_state(random_state)
        loaded_random_state = loads_random_state(serialized_random_state=serialized)
        assert loaded_random_state == random_state

    random_state = RandomState(seed)
    serialized = json.dumps(random_state, cls=CustomEncoder)
    loaded_random_state = loads_random_state(serialized_random_state=serialized)
    assert is_random_state_state_equal(loaded_random_state.get_state(), random_state.get_state())


def test_serialize_numpy_integer():
    """Test serialization of numpy.integer."""
    value = numpy.array([10])
    assert isinstance(value[0], numpy.integer)
    serialized = json.dumps(value[0], cls=CustomEncoder)
    loaded = json.loads(serialized)
    # The type did change though
    assert value[0] == loaded

    serialized = json.dumps(value, cls=CustomEncoder)
    loaded = numpy.array(json.loads(serialized))
    # The type did change though
    assert (value == loaded).all()


def test_serialize_numpy_float():
    """Test serialization of numpy.float."""
    value = numpy.array([10.2], dtype=numpy.float64)
    assert isinstance(value[0], numpy.floating)
    serialized = json.dumps(value[0], cls=CustomEncoder)
    loaded = json.loads(serialized)
    # The type did change though
    assert value[0] == loaded

    serialized = json.dumps(value, cls=CustomEncoder)
    loaded = numpy.array(json.loads(serialized))
    # The type did change though
    assert (value == loaded).all()


def test_serialize_numpy_array():
    """Test serialization of numpy.ndarray."""
    value = numpy.random.random((10, 10, 10, 3))
    assert isinstance(value, numpy.ndarray)
    serialized = json.dumps(value, cls=CustomEncoder)
    loaded = json.loads(serialized)
    assert (value == numpy.array(loaded)).all()


def test_serialize_type():
    """Test serialization of types."""
    value = int
    assert isinstance(value, type)
    serialized = json.dumps(value, cls=CustomEncoder)
    loaded = json.loads(serialized)
    assert value.__name__ == loaded


def test_serialize_onnx():
    """Test serialization of onnx."""
    inputs = torch.zeros(
        10,
    )[None, ...]
    model = SimpleNet()
    model(inputs)
    io_stream = io.BytesIO(initial_bytes=b"")
    torch.onnx.export(
        model=model,
        args=inputs,
        f=io_stream,
    )
    value = onnx.load_model_from_string(io_stream.getvalue())
    assert isinstance(value, onnx.ModelProto)

    serialized = json.dumps(value, cls=CustomEncoder)
    loaded = loads_onnx(json.loads(serialized))
    assert loaded == value

    loaded = loads_onnx(dumps_onnx(value))
    assert loaded == value


def test_crash():
    """Test that trying to load a non-Concrete ML object crashes."""
    serialized = dumps({42: 24})

    with pytest.raises(ValueError, match="The content provided is not a Concrete ML dumped model."):
        loads(serialized)


# pylint: disable-next=too-many-statements
def test_quantizers():
    """Test serialization of quantizers objects."""
    # Some array to quantize
    arr = numpy.linspace((1, 2), (10, 20), 10, dtype=numpy.float64)

    # A buffer to test serialization to file.
    with io.StringIO() as buffer:
        # Quantization Options
        quantization_options = QuantizationOptions(
            n_bits=20,
            is_signed=True,
            is_symmetric=False,
            is_qat=True,
        )
        buffer.seek(0, 0)
        quantization_options.dump(buffer)
        buffer.seek(0, 0)
        loaded = QuantizationOptions.load(buffer)
        buffer.truncate(0)
        dumpsed = quantization_options.dumps()
        assert loaded.dumps() == dumpsed
        assert QuantizationOptions.loads(dumpsed).dumps() == dumpsed

        # MinMaxQuantizationStats
        min_max_quantization_stats = MinMaxQuantizationStats()
        min_max_quantization_stats.compute_quantization_stats(arr)
        buffer.seek(0, 0)
        min_max_quantization_stats.dump(buffer)
        buffer.seek(0, 0)
        loaded = MinMaxQuantizationStats.load(buffer)
        buffer.truncate(0)
        dumpsed = min_max_quantization_stats.dumps()
        assert loaded.dumps() == dumpsed
        assert MinMaxQuantizationStats.loads(dumpsed).dumps() == dumpsed

        # UniformQuantizationParameters
        uniform_quantization_parameters = UniformQuantizationParameters()
        buffer.seek(0, 0)
        uniform_quantization_parameters.dump(buffer)
        buffer.seek(0, 0)
        loaded = UniformQuantizationParameters.load(buffer)
        buffer.truncate(0)
        dumpsed = uniform_quantization_parameters.dumps()
        assert loaded.dumps() == dumpsed
        assert UniformQuantizationParameters.loads(dumpsed).dumps() == dumpsed

        # UniformQuantizationParameters with compute_quantization_parameters
        uniform_quantization_parameters = UniformQuantizationParameters()
        uniform_quantization_parameters.compute_quantization_parameters(
            stats=min_max_quantization_stats, options=quantization_options
        )
        buffer.seek(0, 0)
        uniform_quantization_parameters.dump(buffer)
        buffer.seek(0, 0)
        loaded = UniformQuantizationParameters.load(buffer)
        dumpsed = uniform_quantization_parameters.dumps()
        assert loaded.dumps() == dumpsed
        assert UniformQuantizationParameters.loads(dumpsed).dumps() == dumpsed

        # UniformQuantizer
        uniform_quantizer = UniformQuantizer(
            options=quantization_options,
            stats=min_max_quantization_stats,
            params=uniform_quantization_parameters,
        )
        buffer.seek(0, 0)
        uniform_quantizer.dump(buffer)
        buffer.seek(0, 0)
        loaded = UniformQuantizer.load(buffer)
        buffer.truncate(0)
        dumpsed = uniform_quantizer.dumps()
        assert loaded.dumps() == dumpsed
        assert UniformQuantizer.loads(dumpsed).dumps() == dumpsed

        # QuantizedArray
        quantized_array = QuantizedArray(
            n_bits=20,
            values=arr,
            value_is_float=True,
        )
        buffer.seek(0, 0)
        quantized_array.dump(buffer)
        buffer.seek(0, 0)
        loaded = QuantizedArray.load(buffer)
        buffer.truncate(0)
        assert (loaded.values == quantized_array.values).all()
        dumpsed = quantized_array.dumps()
        assert loaded.dumps() == dumpsed
        assert QuantizedArray.loads(dumpsed).dumps() == dumpsed

        # QuantizedArray with more kwargs
        quantized_array = QuantizedArray(
            n_bits=20,
            values=arr,
            value_is_float=True,
            options=quantization_options,
            stats=min_max_quantization_stats,
            params=uniform_quantization_parameters,
        )
        buffer.seek(0, 0)
        quantized_array.dump(buffer)
        buffer.seek(0, 0)
        loaded = QuantizedArray.load(buffer)
        buffer.truncate(0)
        assert (loaded.values == quantized_array.values).all()
        dumpsed = quantized_array.dumps()
        assert loaded.dumps() == dumpsed
        assert QuantizedArray.loads(dumpsed).dumps() == dumpsed
