__all__ = ["compute_num_cores"]


def compute_num_cores(max_cores, nchunks, max_available):
    """
    Compute the number of chunks to be created for multiprocessing.

    Parameters
    ----------
    max_cores : str
        Number of cores to use for multiprocessing. If set to 'none' (the default),
        then no multiprocessing will be done. The other allowable values are 'quarter',
        'half', and 'all' and string integers. This is the fraction of cores
        to use for multi-proc.
    nchunks : int
        The total number of chunks that will be processed. If more cores are requested
        than chunks, then the number of chunks will be used as the output
        to make sure that each process has some data.
    max_available: int
        This is the total number of cores available. The total number of cores
        includes the SMT cores (Hyper Threading for Intel).

    Returns
    -------
    number_slices : int
        The number of slices for multiprocessing.
    """
    number_slices = 1
    if max_cores.isnumeric():
        number_slices = int(max_cores)
    elif max_cores.lower() == "none" or max_cores.lower() == "one":
        number_slices = 1
    elif max_cores == "quarter":
        number_slices = max_available // 4 or 1
    elif max_cores == "half":
        number_slices = max_available // 2 or 1
    elif max_cores == "all":
        number_slices = max_available
    # Make sure we don't have more slices than chunks or available cores.
    return min([nchunks, number_slices, max_available])
