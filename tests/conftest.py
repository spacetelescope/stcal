import pytest
from tests.helpers import MemoryThreshold


@pytest.fixture
def memory_threshold(expected_usage):
    """Fixture to check peak memory usage against an expected threshold."""
    with MemoryThreshold(expected_usage) as tracker:
        yield tracker