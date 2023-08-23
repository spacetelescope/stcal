from libc.math cimport sqrt, fabs
from libcpp.vector cimport vector
from libcpp cimport bool
import numpy as np
cimport numpy as np
cimport cython

from stcal.ramp_fitting.ols_cas22._core cimport Ramp


cdef class Fixed:
    """
    Class to contain the data fixed for all pixels and commonly referenced
    universal values for jump detection

    Parameters
    ----------
    read_noise : float
        read noise for this pixel
    t_bar : vector[float]
        mean times of resultants
    tau : vector[float]
        variance weighted mean times of resultants
    n_reads : vector[int]
        number of reads contributing to reach resultant

    t_bar_1 : vector[float]
        single differences of t_bar (t_bar[i+1] - t_bar[i])
    t_bar_1_sq : vector[float]
        squared single differences of t_bar (t_bar[i+1] - t_bar[i])**2
    t_bar_2 : vector[float]
        double differences of t_bar (t_bar[i+2] - t_bar[i])
    t_bar_2_sq: vector[float]
        squared double differences of t_bar (t_bar[i+2] - t_bar[i])**2
    sigma_1 : vector[float]
        single of sigma term read_noise * ((1/n_reads[i+1]) + (1/n_reads[i]))
    sigma_2 : vector[float]
        double of sigma term read_noise * ((1/n_reads[i+1]) + (1/n_reads[i]))
    slope_var_1 : vector[float]
        single of slope variance term
        ([tau[i] + tau[i+1] - min(t_bar[i], t_bar[i+1])) * correction(i, i+1)
    slope_var_2 : vector[float]
        double of slope variance term
        ([tau[i] + tau[i+2] - min(t_bar[i], t_bar[i+2])) * correction(i, i+2)
    """

    cdef inline float[:] t_bar_diff(Fixed self, int offset):
        """
        Compute the difference offset of t_bar

        Parameters
        ----------
        offset : int
            index offset to compute difference
        """
        cdef int n_diff = len(self.t_bar) - offset
        cdef float[:] diff = (np.roll(self.t_bar, -offset) - self.t_bar)[:n_diff]

        return diff

    cdef inline float[:] t_bar_diff_sq(Fixed self, int offset):
        """
        Compute the square difference offset of t_bar

        Parameters
        ----------
        offset : int
            index offset to compute difference
        """
        cdef int n_diff = len(self.t_bar) - offset
        cdef float[:] diff = (np.roll(self.t_bar, -offset) - self.t_bar)[:n_diff] ** 2

        return diff

    cdef inline float[:] sigma_val(Fixed self, int offset):
        """
        Compute the sigma values
            read_noise * (1/n_reads[i+offset] + 1/n_reads[i])

        Parameters
        ----------
        offset : int
            index offset to compute difference
        """
        cdef int n_diff = len(self.t_bar) - offset

        # cdef float[:] sig = self.read_noise * (
        #     (1 / np.roll(self.n_reads, -offset) + 1 / np.array(self.n_reads))[:n_diff]).astype(float)
        
        cdef float[:] sig = (1 / np.roll(self.n_reads, -offset)).astype(np.float32)

        return sig

    cdef inline float correction(Fixed self, int i, int j):
        """Compute the correction factor

        Parameters
        ----------
        i : int
            The index of the first read in the segment
        j : int
            The index of the last read in the segment
        """
        cdef float denom = self.t_bar[self.n_reads[i] - 1] - self.t_bar[0]

        if j - i == 1:
            return (1 - (self.t_bar[i + 1] - self.t_bar[i]) / denom) ** 2
        else:
            return (1 - 0.75 * (self.t_bar[i + 2] - self.t_bar[i]) / denom) ** 2

    cdef inline float[:] slope_var_val(Fixed self, int offset):
        """
        Compute the sigma values
            (tau[i] + tau[i+offset] - min(t_bar[i], t_bar[i+offset])) *
                correction(i, i+offset)

        Parameters
        ----------
        offset : int
            index offset to compute difference
        """
        cdef int n_diff = len(self.t_bar) - offset

        cdef float[:] slope_var_val = (
            (self.tau + np.roll(self.tau, -offset) -
             np.minimum(self.t_bar, np.roll(self.t_bar, -offset))) *
            self.correction(0, offset))[:n_diff]

        return slope_var_val


cdef inline Fixed make_fixed(
        float read_noise, float[:] t_bar, float[:] tau, int[:] n_reads, bool use_jump):

    cdef Fixed fixed = Fixed()

    fixed.use_jump = use_jump
    fixed.read_noise = read_noise
    fixed.t_bar = t_bar
    fixed.tau = tau
    fixed.n_reads = n_reads

    if use_jump:
        fixed.t_bar_1 = fixed.t_bar_diff(1)
        fixed.t_bar_2 = fixed.t_bar_diff(2)

        fixed.t_bar_1_sq = fixed.t_bar_diff_sq(1)
        fixed.t_bar_2_sq = fixed.t_bar_diff_sq(2)

        fixed.sigma_1 = fixed.sigma_val(1)
        fixed.sigma_2 = fixed.sigma_val(2)

        fixed.slope_var_1 = fixed.slope_var_val(1)
        fixed.slope_var_2 = fixed.slope_var_val(2)

    return fixed


    # cdef inline vector[float] t_bar_diff_sq(Fixed self, int offset):
    #     """
    #     Compute the square difference offset of t_bar

    #     Parameters
    #     ----------
    #     offset : int
    #         index offset to compute difference
    #     """
    #     cdef int n_diff = len(self.t_bar) - offset
    #     cdef vector[float] diff = vector[float](n_diff)

    #     for i in range(n_diff):
    #         diff[i] = (self.t_bar[i + offset] - self.t_bar[i])**2

    #     return diff

    # cdef inline vector[float] sigma(Fixed, self, int offset):
    #     """
    #     Compute
    #         read_noise * (1/n_reads[i+offset] + 1/n_reads[i])

    #     Parameters
    #     ----------
    #     offset : int
    #         index offset to compute difference
    #     """
    #     cdef int n_diff = len(self.t_bar) - offset
    #     cdef vector[float] sig = vector[float](n_diff)

    #     for i in range(n_diff):
    #         sig[i] = read_noise * (1 / self.n_reads[i + offset] + 1 / self.n_reads[i])

    #     return sig

    # cdef inline vector[float] slope_var(Fixed, self, int offset):
    #     """
    #     Compute
    #         read_noise * (1/n_reads[i+offset] + 1/n_reads[i])

    #     Parameters
    #     ----------
    #     offset : int
    #         index offset to compute difference
    #     """
    #     cdef int n_diff = len(self.t_bar) - offset
    #     cdef vector[float] sig = vector[float](n_diff)

    #     for i in range(n_diff):
    #         sig[i] = read_noise * (1 / self.n_reads[i + offset] + 1 / self.n_reads[i])

    #     return sig

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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
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
