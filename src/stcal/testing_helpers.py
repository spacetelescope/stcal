import tracemalloc

MEMORY_UNIT_CONVERSION = {"B": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3, "TB": 1024 ** 4}

class MemoryThresholdExceeded(Exception):
    pass


class MemoryThreshold:
    """
    Context manager to check peak memory usage against an expected threshold.

    example usage:
    with MemoryThreshold(expected_usage):
        # code that should not exceed expected

    If the code in the with statement uses more than the expected_usage
    memory a ``MemoryThresholdExceeded`` exception
    will be raised.
    
    Note that this class does not prevent allocations beyond the threshold
    and only checks the actual peak allocations to the threshold at the
    end of the with statement.
    """

    def __init__(self, expected_usage):
        """
        Parameters
        ----------
        expected_usage : str
            Expected peak memory usage expressed as a whitespace-separated string
            with a number and a memory unit (e.g. "100 KB").
            Supported units are "B", "KB", "MB", "GB", "TB".
        """
        expected, self.units = expected_usage.upper().split()
        self.expected_usage_bytes = float(expected) * MEMORY_UNIT_CONVERSION[self.units]

    def __enter__(self):
        tracemalloc.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if peak > self.expected_usage_bytes:
            scaling = MEMORY_UNIT_CONVERSION[self.units]
            msg = ("Peak memory usage exceeded expected usage: "
                  f"{peak / scaling:.2f} {self.units} > "
                  f"{self.expected_usage_bytes / scaling:.2f} {self.units} ")
            raise MemoryThresholdExceeded(msg)
