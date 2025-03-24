#! /usr/bin/env python
#
# utils.py: utility functions
import logging
import warnings

import numpy as np


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# Replace zero or negative variances with this:
LARGE_VARIANCE = 1.0e8
LARGE_VARIANCE_THRESHOLD = 0.01 * LARGE_VARIANCE


def compute_num_slices(max_cores, nrows, max_available):
    """
    Computes the number of slices to be created for multiprocessing.

    Parameters
    ----------
    max_cores : str
        Number of cores to use for multiprocessing. If set to 'none' (the default),
        then no multiprocessing will be done. The other allowable values are 'quarter',
        'half', and 'all' and string integers. This is the fraction of cores
        to use for multi-proc.
    nrows : int
        The number of rows that will be used across all process. This is the
        maximum number of slices to make sure that each process has some data.
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
    # Make sure we don't have more slices than rows or available cores.
    return min([nrows, number_slices, max_available])
