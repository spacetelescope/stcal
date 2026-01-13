"""Tests of MemoryThreshold"""

import numpy as np
import pytest

from stcal._testing.memory_threshold import MemoryThreshold, MemoryThresholdExceeded


def test_memory_threshold():
    with MemoryThreshold("10 KB"):
        np.ones(1000, dtype=np.uint8)


def test_memory_threshold_exceeded():
    with pytest.raises(MemoryThresholdExceeded):
        with MemoryThreshold("500. B"):
            np.ones(10000, dtype=np.uint8)
