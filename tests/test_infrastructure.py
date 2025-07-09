"""Tests of custom testing infrastructure"""

import numpy as np
import pytest

from stcal.testing_helpers import MemoryThreshold, MemoryThresholdExceededError


def test_memory_threshold():
    with MemoryThreshold("10 KB"):
        buff = np.ones(1000, dtype=np.uint8)


def test_memory_threshold_exceeded():
    with pytest.raises(MemoryThresholdExceededError):
        with MemoryThreshold("500. B"):
            buff = np.ones(10000, dtype=np.uint8)
