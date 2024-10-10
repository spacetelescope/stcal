import tracemalloc

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

    def __init__(self, expected_usage):
        """
        Parameters
        ----------
        expected_usage : int
            Expected peak memory usage in bytes
        """
        self.expected_usage = expected_usage

    def __enter__(self):
        tracemalloc.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        if peak > self.expected_usage:
            msg = ("Peak memory usage exceeded expected usage: "
                  f"{peak / 1024:.2f} KB > {self.expected_usage / 1024:.2f} KB")
            raise MemoryThresholdExceeded(msg)
