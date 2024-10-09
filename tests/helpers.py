import tracemalloc

class MemoryThreshold:
    def __init__(self, expected_usage):
        self.expected_usage = expected_usage

    def __enter__(self):
        tracemalloc.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        assert peak <= self.expected_usage, (
            "Peak memory usage exceeded expected usage: "
            f"{peak / 1024:.2f} KB > {self.expected_usage / 1024:.2f} KB"
        )