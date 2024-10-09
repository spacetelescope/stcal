import pytest
from tests.helpers import MemoryThreshold


@pytest.fixture
def memory_threshold(expected_usage):
    with MemoryThreshold(expected_usage) as tracker:
        yield tracker