
from libc.math cimport sqrt, fabs
import numpy as np
cimport numpy as np

cimport cython

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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline (float, float, float) fit_one_ramp(
        float [:] resultants, int start, int end, float read_noise,
        float [:] t_bar, float [:] tau, int [:] n_reads):
    """Fit a portion of single ramp using the Casertano+22 algorithm.

    Parameters
    ----------
    resultants : float [:]
        array of resultants for single pixel
    start : int
        starting point of portion to fit within this pixel
    end : int
        ending point of portion to fit within this pixel
    read_noise : float
        read noise for this pixel
    t_bar : float [:]
        mean times of resultants
    tau : float [:]
        variance weighted mean times of resultants
    n_reads : int [:]
        number of reads contributing to reach resultant

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
    cdef int n_resultants = end - start + 1

    # Special case where there is no or one resultant, there is no fit.
    if n_resultants <= 1:
        return 0, 0, 0

    # Else, do the fitting.
    cdef int i = 0, j = 0
    cdef float weights[2048]
    cdef float coeffs[2048]
    cdef float slope = 0, slope_read_var = 0, slope_poisson_var = 0
    cdef float t_bar_mid = (t_bar[start] + t_bar[end]) / 2

    # Casertano+2022 Eq. 44
    # Note we've departed from Casertano+22 slightly;
    # there s is just resultants[end].  But that doesn't seem good if, e.g.,
    # a CR in the first resultant has boosted the whole ramp high but there
    # is no actual signal.
    cdef float s = max(resultants[end] - resultants[start], 0)
    s = s / sqrt(read_noise**2 + s)
    cdef float power = get_weight_power(s)

    # It's easy to use up a lot of dynamic range on something like
    # (tbar - tbarmid) ** 10.  Rescale these.
    cdef float t_scale = (t_bar[end] - t_bar[start]) / 2
    t_scale = 1 if t_scale == 0 else t_scale

    cdef float f0 = 0, f1 = 0, f2 = 0

    with cython.cpow(True):  # Issue when tbar[] == tbarmid causes exception otherwise
        for i in range(n_resultants):
            # Casertano+22, Eq. 45
            weights[i] = ((((1 + power) * n_reads[start + i]) /
                (1 + power * n_reads[start + i])) *
                fabs((t_bar[start + i] - t_bar_mid) / t_scale) ** power)

            # Casertano+22 Eq. 35
            f0 += weights[i]
            f1 += weights[i] * t_bar[start + i]
            f2 += weights[i] * t_bar[start + i]**2

    # Casertano+22 Eq. 36
    cdef float det = f2 * f0 - f1 ** 2
    if det == 0:
        return (0.0, 0.0, 0.0)

    for i in range(n_resultants):
        # Casertano+22 Eq. 37
        coeffs[i] = (f0 * t_bar[start + i] - f1) * weights[i] / det

    for i in range(n_resultants):
        # Casertano+22 Eq. 38
        slope += coeffs[i] * resultants[start + i]

        # Casertano+22 Eq. 39
        slope_read_var += coeffs[i] ** 2 * read_noise ** 2 / n_reads[start + i]

        # Casertano+22 Eq 40
        slope_poisson_var += coeffs[i] ** 2 * tau[start + i]
        for j in range(i + 1, n_resultants):
            slope_poisson_var += 2 * coeffs[i] * coeffs[j] * t_bar[start + i]

    return (slope, slope_read_var, slope_poisson_var)
