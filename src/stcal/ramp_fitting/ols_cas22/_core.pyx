from libc.math cimport sqrt, fabs, log10
from libcpp.vector cimport vector
from libcpp.stack cimport stack
from libcpp cimport bool
import numpy as np
cimport numpy as np
cimport cython

from stcal.ramp_fitting.ols_cas22._core cimport RampIndex, Thresh, Fit, Fixed, Ramp


cdef class Fixed:
    """
    Class to contain the data fixed for all pixels and commonly referenced
    universal values for jump detection

    Parameters
    ----------
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

    cdef inline float[:] recip_val(Fixed self, int offset):
        """
        Compute the recip values
            (1/n_reads[i+offset] + 1/n_reads[i])

        Parameters
        ----------
        offset : int
            index offset to compute difference
        """
        cdef int n_diff = len(self.t_bar) - offset
        
        cdef float[:] recip = ((1 / np.roll(self.n_reads, -offset)).astype(np.float32) +
                               (1 / np.array(self.n_reads)).astype(np.float32))[:n_diff]

        return recip

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


cdef inline Fixed make_fixed(float[:] t_bar, float[:] tau, int[:] n_reads, bool use_jump):

    cdef Fixed fixed = Fixed()

    fixed.use_jump = use_jump
    fixed.t_bar = t_bar
    fixed.tau = tau
    fixed.n_reads = n_reads

    if use_jump:
        fixed.t_bar_1 = fixed.t_bar_diff(1)
        fixed.t_bar_2 = fixed.t_bar_diff(2)

        fixed.t_bar_1_sq = fixed.t_bar_diff_sq(1)
        fixed.t_bar_2_sq = fixed.t_bar_diff_sq(2)

        fixed.recip_1 = fixed.recip_val(1)
        fixed.recip_2 = fixed.recip_val(2)

        fixed.slope_var_1 = fixed.slope_var_val(1)
        fixed.slope_var_2 = fixed.slope_var_val(2)

    return fixed


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
    """

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline float[:] resultants_diff(Ramp self, int offset):
        """
        Compute the difference offset of resultants

        Parameters
        ----------
        offset : int
            index offset to compute difference
        """
        cdef int n_diff = len(self.resultants) - offset
        cdef float[:] diff = (np.roll(self.resultants, -offset) - self.t_bar)[:n_diff]

        return diff

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline (float, float, float) fit(Ramp self, int start, int end):
        """Fit a portion of single ramp using the Casertano+22 algorithm.
        Parameters
        ----------
        start : int
            Start of range to fit ramp
        end : int
            End of range to fit ramp
        fixed : Fixed
            Fixed values for all pixels

        Returns
        -------
        slope : float
            fit slope
        read_var : float
            read noise induced variance in slope
        poisson_var : float
            coefficient of Poisson-noise induced variance in slope
            multiply by true flux to get actual Poisson variance.
        """
        cdef int n_resultants = end - start + 1

        # Special case where there is no or one resultant, there is no fit.
        if n_resultants <= 1:
            return 0, 0, 0

        # Setup data for fitting (work over subset of data)
        cdef float[:] resultants = self.fixed.resultants[start:end + 1]
        cdef float[:] t_bar = self.fixed.t_bar[start:end + 1]
        cdef float[:] tau = self.fixed.tau[start:end + 1]
        cdef int[:] n_reads = self.fixed.n_reads[start:end + 1]

        # Else, do the fitting.
        cdef int i = 0, j = 0
        cdef vector[float] weights = vector[float](n_resultants)
        cdef vector[float] coeffs = vector[float](n_resultants)
        cdef float slope = 0, read_var = 0, poisson_var = 0
        cdef float t_bar_mid = (t_bar[0] + t_bar[- 1]) / 2

        # Casertano+2022 Eq. 44
        # Note we've departed from Casertano+22 slightly;
        # there s is just resultants[end].  But that doesn't seem good if, e.g.,
        # a CR in the first resultant has boosted the whole ramp high but there
        # is no actual signal.
        cdef float s = max(resultants[-1] - resultants[0], 0)
        s = s / sqrt(self.fixed.read_noise**2 + s)
        cdef float power = get_weight_power(s)

        # It's easy to use up a lot of dynamic range on something like
        # (tbar - tbarmid) ** 10.  Rescale these.
        cdef float t_scale = (t_bar[-1] - t_bar[0]) / 2
        t_scale = 1 if t_scale == 0 else t_scale

        cdef float f0 = 0, f1 = 0, f2 = 0

        # Issue when tbar[] == tbarmid causes exception otherwise
        with cython.cpow(True):
            for i in range(n_resultants):
                # Casertano+22, Eq. 45
                weights[i] = ((((1 + power) * n_reads[i]) / (1 + power * n_reads[i])) *
                              fabs((t_bar[i] - t_bar_mid) / t_scale) ** power)

                # Casertano+22 Eq. 35
                f0 += weights[i]
                f1 += weights[i] * t_bar[i]
                f2 += weights[i] * t_bar[i]**2

        # Casertano+22 Eq. 36
        cdef float det = f2 * f0 - f1 ** 2
        if det == 0:
            return (0.0, 0.0, 0.0)

        for i in range(n_resultants):
            # Casertano+22 Eq. 37
            coeffs[i] = (f0 * t_bar[i] - f1) * weights[i] / det

        for i in range(n_resultants):
            # Casertano+22 Eq. 38
            slope += coeffs[i] * resultants[i]

            # Casertano+22 Eq. 39
            read_var += (coeffs[i] ** 2 * self.fixed.read_noise ** 2 / n_reads[i])

            # Casertano+22 Eq 40
            poisson_var += coeffs[i] ** 2 * tau[i]
            for j in range(i + 1, n_resultants):
                poisson_var += (2 * coeffs[i] * coeffs[j] * t_bar[i])

        return (slope, read_var, poisson_var)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline float[:] stats(Ramp self, float slope, int start, int end):
        cdef np.ndarray[float] delta_1 = np.array(self.delta_1[start:end-1]) - slope
        cdef np.ndarray[float] delta_2 = np.array(self.delta_2[start:end-1]) - slope

        cdef np.ndarray[float] var_1 = ((np.array(self.sigma_1[start:end-1]) +
                                         slope * np.array(self.slope_var_1[start:end-1])) /
                                        self.fixed.t_bar_1_sq[start:end-1]).astype(np.float32)
        cdef np.ndarray[float] var_2 = ((np.array(self.sigma_2[start:end-1]) +
                                         slope * np.array(self.slope_var_2[start:end-1])) /
                                        self.fixed.t_bar_2_sq[start:end-1]).astype(np.float32)

        cdef np.ndarray[float] stats_1 = delta_1 / sqrt(var_1)
        cdef np.ndarray[float] stats_2 = delta_2 / sqrt(var_2)

        return np.maximum(stats_1, stats_2)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline (stack[float], stack[float], stack[float]) fits(Ramp self, stack[RampIndex] ramps, Thresh thresh):
        cdef stack[float] slopes, read_vars, poisson_vars
        cdef RampIndex ramp
        cdef float slope = 0, read_var = 0, poisson_var = 0
        cdef float [:] stats
        cdef int split

        while not ramps.empty():
            ramp = ramps.top()
            ramps.pop()
            slope, read_var, poisson_var = self.fit(ramp.start, ramp.end)
            stats = self.stats(slope, ramp.start, ramp.end)
            
            if max(stats) > threshold(thresh, slope):
                split = np.argmax(stats)

                ramps.push(RampIndex(ramp.start, ramp.start + split))
                ramps.push(RampIndex(ramp.start + split + 2, ramp.end))
            else:
                slopes.push(slope)
                read_vars.push(read_var)
                poisson_vars.push(poisson_var)

        return slopes, read_vars, poisson_vars
    

cdef float threshold(Thresh thresh, float slope):
    return thresh.intercept - thresh.constant * log10(slope)


cdef inline Ramp make_ramp(Fixed fixed, float read_noise, float [:] resultants):
    """
    Fast constructor for the Ramp C class.

    This is signifantly faster than using the `__init__` or `__cinit__`
        this is because this does not have to pass through the Python as part
        of the construction.

    Parameters
    ----------
    fixed : Fixed
        Fixed values for all pixels
    resultants : float [:]
        array of resultants for single pixel
            - memoryview of a numpy array to avoid passing through Python

    Return
    ------
    ramp : Ramp
        Ramp C-class object
    """

    cdef Ramp ramp = Ramp()

    ramp.fixed = fixed
    ramp.read_noise = read_noise
    ramp.resultants = resultants

    if fixed.use_jump:
        ramp.delta_1 = (np.array(ramp.resultants_diff(1)) / np.array(fixed.t_bar_1)).astype(np.float32)
        ramp.delta_2 = (np.array(ramp.resultants_diff(2)) / np.array(fixed.t_bar_2)).astype(np.float32)

        ramp.sigma_1 = read_noise * np.array(fixed.recip_1)
        ramp.sigma_2 = read_noise * np.array(fixed.recip_2)

    return ramp
