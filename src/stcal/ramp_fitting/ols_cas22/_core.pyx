from libc.math cimport sqrt, fabs
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np
cimport cython

from stcal.ramp_fitting.ols_cas22._core cimport Ramp

# Casertano+2022, Table 2
cdef float[2][6] PTABLE = [
    [-np.inf, 5, 10, 20, 50, 100],
    [0,     0.4,  1,  3,  6,  10]]
cdef int PTABLE_LENGTH = 6

cdef inline float get_weight_power(float s):
    cdef int i
    for i in range(PTABLE_LENGTH):
        if s < PTABLE[0][i]:
            return PTABLE[1][i - 1]
    return PTABLE[1][i]

cdef class Ramp:
    """
    Class to contain the data for a single pixel ramp to be fit

    This has to be a class rather than a struct in order to contain memory views

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
    """

    cdef inline (float, float, float) fit(Ramp self):
        """Fit a portion of single ramp using the Casertano+22 algorithm.

        Returns
        -------
        slope : float
            fit slope
        slope_read_var : float
            read noise induced variance in slope
        slope_poisson_var : float
            coefficient of Poisson-noise induced variance in slope
            multiply by true flux to get actual Poisson variance.
        """
        cdef int n_resultants = self.end - self.start + 1

        # Special case where there is no or one resultant, there is no fit.
        if n_resultants <= 1:
            return 0, 0, 0

        # Else, do the fitting.
        cdef int i = 0, j = 0
        cdef vector[float] weights = vector[float](n_resultants)
        cdef vector[float] coeffs = vector[float](n_resultants)
        cdef float slope = 0, slope_read_var = 0, slope_poisson_var = 0
        cdef float t_bar_mid = (self.t_bar[self.start] + self.t_bar[self.end]) / 2

        # Casertano+2022 Eq. 44
        # Note we've departed from Casertano+22 slightly;
        # there s is just resultants[end].  But that doesn't seem good if, e.g.,
        # a CR in the first resultant has boosted the whole ramp high but there
        # is no actual signal.
        cdef float s = max(self.resultants[self.end] - self.resultants[self.start], 0)
        s = s / sqrt(self.read_noise**2 + s)
        cdef float power = get_weight_power(s)

        # It's easy to use up a lot of dynamic range on something like
        # (tbar - tbarmid) ** 10.  Rescale these.
        cdef float t_scale = (self.t_bar[self.end] - self.t_bar[self.start]) / 2
        t_scale = 1 if t_scale == 0 else t_scale

        cdef float f0 = 0, f1 = 0, f2 = 0

        # Issue when tbar[] == tbarmid causes exception otherwise
        with cython.cpow(True):
            for i in range(n_resultants):
                # Casertano+22, Eq. 45
                weights[i] = ((((1 + power) * self.n_reads[self.start + i]) /
                              (1 + power * self.n_reads[self.start + i])) *
                              fabs((self.t_bar[self.start + i] - t_bar_mid) /
                              t_scale) ** power)

                # Casertano+22 Eq. 35
                f0 += weights[i]
                f1 += weights[i] * self.t_bar[self.start + i]
                f2 += weights[i] * self.t_bar[self.start + i]**2

        # Casertano+22 Eq. 36
        cdef float det = f2 * f0 - f1 ** 2
        if det == 0:
            return (0.0, 0.0, 0.0)

        for i in range(n_resultants):
            # Casertano+22 Eq. 37
            coeffs[i] = (f0 * self.t_bar[self.start + i] - f1) * weights[i] / det

        for i in range(n_resultants):
            # Casertano+22 Eq. 38
            slope += coeffs[i] * self.resultants[self.start + i]

            # Casertano+22 Eq. 39
            slope_read_var += (coeffs[i] ** 2 * self.read_noise ** 2 /
                               self.n_reads[self.start + i])

            # Casertano+22 Eq 40
            slope_poisson_var += coeffs[i] ** 2 * self.tau[self.start + i]
            for j in range(i + 1, n_resultants):
                slope_poisson_var += (2 * coeffs[i] * coeffs[j] *
                                      self.t_bar[self.start + i])

        return (slope, slope_read_var, slope_poisson_var)


cdef inline Ramp make_ramp(
        float [:] resultants, int start, int end, float read_noise,
        vector[float] t_bar, vector[float] tau, vector[int] n_reads):
    """
    Fast constructor for the Ramp C class.

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
    ramp : Ramp
        Ramp C-class object
    """

    cdef Ramp ramp = Ramp()

    ramp.start = start
    ramp.end = end

    ramp.resultants = resultants
    ramp.t_bar = t_bar
    ramp.tau = tau

    ramp.read_noise = read_noise

    ramp.n_reads = n_reads

    return ramp
