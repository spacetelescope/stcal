import tracemalloc

MEMORY_UNIT_CONVERSION = {"B": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3}

class MemoryThresholdExceeded(Exception):
    pass


class MemoryThreshold:
    """
    Context manager to check peak memory usage against an expected threshold.

    example usage:
    with MemoryThreshold(expected_usage):
        # code that should not exceed expected

    If the code in the with statement uses more than the expected_usage
    memory (in bytes) a ``MemoryThresholdExceeded`` exception
    will be raised.
    
    Note that this class does not prevent allocations beyond the threshold
    and only checks the actual peak allocations to the threshold at the
    end of the with statement.
    """

    def __init__(self, expected_usage, log_units="KB"):
        """
        Parameters
        ----------
        expected_usage : int
            Expected peak memory usage in bytes

        log_units : str, optional
            Units in which to display memory usage for error message. 
            Supported are "B", "KB", "MB", "GB". Default is "KB".
        """
        self.expected_usage = expected_usage
        self.log_units = log_units

    def __enter__(self):
        tracemalloc.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if peak > self.expected_usage:
            scaling = MEMORY_UNIT_CONVERSION[self.log_units]
            msg = ("Peak memory usage exceeded expected usage: "
                  f"{peak / scaling:.2f} {self.log_units} > "
                  f"{self.expected_usage / scaling:.2f} {self.log_units} ")
            raise MemoryThresholdExceeded(msg)
