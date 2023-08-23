import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log10
from libcpp.vector cimport vector
cimport cython

from stcal.ramp_fitting.ols_cas22._core cimport Ramp


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
    cdef inline vector[float] statistic(Jump self, float slope):
        cdef int n_stats = len(self.n_reads), i

        cdef vector[float] stats = vector[float](n_stats)
        cdef float stat_1, stat_2

        for i in range(n_stats):
            stat_1 = self.stat(i, i + 1, slope)
            stat_2 = self.stat(i, i + 2, slope)

            stats[i] = max(stat_1, stat_2)

        return stats

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline (float, float, float, vector[float]) jump(Jump self):
        cdef float slope, read_var, poisson_var 
        slope, read_var, poisson_var = self.fit()

        cdef vector[float] stats = self.statistic(slope)

        return slope, read_var, poisson_var, stats

    # @cython.boundscheck(False)
    # @cython.wraparound(False)
    # @cython.cdivision(True)
    # cdef inline (Jump, Jump) split(Jump self, int split):
    #     cdef Jump jump_1 = make_jump(
    #             self.resultants, self.start, self.start + split, self.read_noise,
    #             self.t_bar, self.tau, self.n_reads)

    #     cdef Jump jump_2 = make_jump(
    #             self.resultants, self.start + split + 2, self.end, self.read_noise,
    #             self.t_bar, self.tau, self.n_reads)

    #     return jump_1, jump_2




cdef float threshold(float intercept, float constant, float slope):
    return intercept - constant * log10(slope)


cdef inline Jump make_jump(
        float [:] resultants, int start, int end, float read_noise,
        vector[float] t_bar, vector[float] tau, vector[int] n_reads):

    """
    Fast constructor for the Jump C class.

    This is signifantly faster than using the `__init__` or `__cinit__`
        this is because this does not have to pass through the Python as part
        of the construction.

    Parameters
    ----------
    resultants : float [:]
        array of resultants for single pixel
            - memoryview of a numpy array to avoid passing through Python
    start : int
        starting point of portion to fit within this pixel
    end : int
        ending point of portion to fit within this pixel
    read_noise : float
        read noise for this pixel
    t_bar : vector[float]
        mean times of resultants
    tau : vector[float]
        variance weighted mean times of resultants
    n_reads : vector[int]
        number of reads contributing to reach resultant

    Return
    ------
    jump : Jump
        Jump C-class object
    """

    cdef Jump jump = Jump()

    jump.start = start
    jump.end = end

    jump.resultants = resultants
    jump.t_bar = t_bar
    jump.tau = tau

    jump.read_noise = read_noise

    jump.n_reads = n_reads

    return jump


cdef (vector[float], vector[float], vector[float]) fit(
        float [:] resultants, int start, int end, float read_noise,
        vector[float] t_bar, vector[float] tau, vector[int] n_reads,
        float intercept, float constant):

    cdef vector[float] slopes
    cdef vector[float] read_vars
    cdef vector[float] poisson_vars

    cdef Jump jump, jump_1, jump_2
    cdef int split
    cdef vector[float] stats
    cdef float slope, read_var, poisson_var

    cdef list[Jump] jumps = [make_jump(resultants, start, end, read_noise, t_bar, tau, n_reads)]
    while jumps:
        jump = jumps.pop()
        slope, read_var, poisson_var, stats = jump.jump()

        if max(stats) > threshold(intercept, constant, slope):
            split = np.argmax(stats)

            jump_1 = make_jump(
                    jump.resultants, jump.start, jump.start + split, jump.read_noise,
                    jump.t_bar, jump.tau, jump.n_reads)

            jump_2 = make_jump(
                    jump.resultants, jump.start + split + 2, jump.end, jump.read_noise,
                    jump.t_bar, jump.tau, jump.n_reads)

            jumps.append(jump_1)
            jumps.append(jump_2)

        else:
            slopes.push_back(slope)
            read_vars.push_back(read_var)
            poisson_vars.push_back(poisson_var)


        return stats, read_vars, poisson_vars
