import numpy as np
cimport numpy as cnp
from cython cimport boundscheck, wraparound

from stcal.ramp_fitting.ols_cas22._read_pattern cimport ReadPattern

cnp.import_array()

cdef class ReadPattern:
    """
    Class to contain the read pattern derived metadata

    Attributes:
    ----------
    n_resultants : int
        The number of resultants in the read pattern
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
cpdef ReadPattern from_read_pattern(list[list[int]] read_pattern, float read_time, int n_resultants):
    """
    Derive the input data from the the read pattern

        read pattern is a list of resultant lists, where each resultant list is
        a list of the reads in that resultant.

    Parameters
    ----------
    read pattern: list[list[int]]
        read pattern for the image
    read_time : float
        Time to perform a readout.

    Returns
    -------
    ReadPattern
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