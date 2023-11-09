# cython: language_level=3str

import numpy as np
cimport numpy as cnp
from cython cimport boundscheck, cdivision, wraparound

from stcal.ramp_fitting.ols_cas22._read_pattern cimport ReadPattern

cnp.import_array()

cdef class ReadPattern:
    """
    Class to contain the read pattern derived metadata
        This exists only to allow us to output multiple memory views at the same time
        from the same cython function. This is needed because neither structs nor unions
        can contain memory views.

        In the case of this code memory views are the fastest "safe" array data structure.
        This class will immediately be unpacked into raw memory views, so that we avoid
        any further overhead of swithcing between python and cython.

    Attributes:
    ----------
    t_bar : np.ndarray[float_t, ndim=1]
        The mean time of each resultant
    tau : np.ndarray[float_t, ndim=1]
        The variance in time of each resultant
    n_reads : np.ndarray[cnp.int32_t, ndim=1]
        The number of reads in each resultant
    """

    def _to_dict(ReadPattern self):
        """
        This is a private method to convert the ReadPattern object to a dictionary,
            so that attributes can be directly accessed in python. Note that this
            is needed because class attributes cannot be accessed on cython classes
            directly in python. Instead they need to be accessed or set using a
            python compatible method. This method is a pure puthon method bound
            to to the cython class and should not be used by any cython code, and
            only exists for testing purposes.
        """
        return dict(t_bar=np.array(self.t_bar, dtype=np.float32),
                    tau=np.array(self.tau, dtype=np.float32),
                    n_reads=np.array(self.n_reads, dtype=np.int32))


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef ReadPattern from_read_pattern(list[list[int]] read_pattern, float read_time, int n_resultants):
    """
    Derive the input data from the the read pattern
        This is faster than using __init__ or __cinit__ to construct the object with
        these calls.

    Parameters
    ----------
    read pattern: list[list[int]]
        read pattern for the image
    read_time : float
        Time to perform a readout.
    n_resultants : int
        Number of resultants in the image

    Returns
    -------
    ReadPattern
        Contains:
        - t_bar
        - tau
        - n_reads
    """

    cdef ReadPattern data = ReadPattern()
    data.t_bar = np.empty(n_resultants, dtype=np.float32)
    data.tau = np.empty(n_resultants, dtype=np.float32)
    data.n_reads = np.empty(n_resultants, dtype=np.int32)

    cdef int index, n_reads
    cdef list[int] resultant
    for index, resultant in enumerate(read_pattern):
            n_reads = len(resultant)

            data.n_reads[index] = n_reads
            data.t_bar[index] = read_time * np.mean(resultant)
            data.tau[index] = np.sum((2 * (n_reads - np.arange(n_reads)) - 1) * resultant) * read_time / n_reads**2

    return data