"""Placeholder test"""

import random

import pytest

from concrete import ml


@pytest.mark.repeat(3)
def test_placeholder():
    """Placeholder test"""

    ml.placeholder()
    print("random", random.randint(0, 1000))
