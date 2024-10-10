"""Tests of custom testing infrastructure"""

import pytest
import numpy as np
from stcal.testing_helpers import MemoryThreshold, MemoryThresholdExceeded


def test_memory_threshold():
    with MemoryThreshold(1000):
        buff = np.empty(200, dtype=np.uint8)


def test_memory_threshold_raise():
    with pytest.raises(MemoryThresholdExceeded):
        with MemoryThreshold(1000):
            buff = np.empty(2000, dtype=np.uint8)