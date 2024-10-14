"""Tests of custom testing infrastructure"""

import pytest
import numpy as np
from stcal.testing_helpers import MemoryThreshold, MemoryThresholdExceeded


def test_memory_threshold():
    with MemoryThreshold("1 KB"):
        buff = np.empty(100, dtype=np.uint8)


def test_memory_threshold_exceeded():
    with pytest.raises(MemoryThresholdExceeded):
        with MemoryThreshold("500. B"):
            buff = np.empty(1000, dtype=np.uint8)