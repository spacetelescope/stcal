import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log10
from libcpp.vector cimport vector
cimport cython

from stcal.ramp_fitting.ols_cas22._core cimport Ramp


cdef float threshold(float intercept, float constant, float slope):
    return intercept - constant * log10(slope)


cdef class Jump(Ramp):
    """
    Class to contain the data for a single ramp fit with jump detection
    """

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline float correction_factor(Jump self, int i, int j):
        """Compute the correction factor

        Parameters
        ----------
        i : int
            The index of the first read in the segment
        j : int
            The index of the last read in the segment
        """
        cdef float denom = self.t_bar[-1] - self.t_bar[0]

        if j - i == 1:
            return (1 - (self.t_bar[i + 1] - self.t_bar[i]) / denom) ** 2
        else:
            return (1 - 0.75 * (self.t_bar[i + 2] - self.t_bar[i]) / denom) ** 2

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline float delta_var(Jump self, int i, int j, float slope):

        return (
                (
                    self.read_noise * (1 / self.n_reads[i] + 1 / self.n_reads[j]) +
                    slope * (self.tau[i] + self.tau[j] - np.min(self.t_bar[i], self.t_bar[j])) *
                    self.correction_factor(i, j)
                ) / ((self.t_bar[j] - self.t_bar[i]) ** 2)
            )

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline float stat(Jump self, int i, int j, float slope):
        cdef float delta = ((self.resultants[j] - self.resultants[i]) / (self.t_bar[j] - self.t_bar[i])) - slope

        return delta / sqrt(self.delta_var(i, j, slope))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline float statistic(Jump self, float slope):
        cdef int n_stats = len(self.n_reads), i

        cdef vector[float] stats = vector[float](n_stats)
        cdef float stat_1, stat_2

        for i in range(n_stats):
            stat_1 = self.stat(i, i + 1, slope)
            stat_2 = self.stat(i, i + 2, slope)

            stats[i] = max(stat_1, stat_2)

        return max(stats)
