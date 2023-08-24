from libc.math cimport sqrt, fabs, log10
from libcpp.vector cimport vector
from libcpp.stack cimport stack

import numpy as np
cimport numpy as np
cimport cython


from stcal.ramp_fitting.ols_cas22._core cimport get_power, reverse_fits, threshold, Fit, Fits, RampIndex, Thresh
from stcal.ramp_fitting.ols_cas22._ramp cimport make_ramp, Ramp


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
    cdef inline Fit fit(Ramp self, RampIndex ramp):
        """Fit a portion of single ramp using the Casertano+22 algorithm.
        Parameters
        ----------

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
        cdef int n_resultants = ramp.end - ramp.start + 1

        # Special case where there is no or one resultant, there is no fit.
        if n_resultants <= 1:
            return 0, 0, 0

        # Setup data for fitting (work over subset of data)
        cdef float[:] resultants = self.fixed.resultants[ramp.start:ramp.end + 1]
        cdef float[:] t_bar = self.fixed.t_bar[ramp.start:ramp.end + 1]
        cdef float[:] tau = self.fixed.tau[ramp.start:ramp.end + 1]
        cdef int[:] n_reads = self.fixed.n_reads[ramp.start:ramp.end + 1]

        # Else, do the fitting.
        cdef int i = 0, j = 0
        cdef vector[float] weights = vector[float](n_resultants)
        cdef vector[float] coeffs = vector[float](n_resultants)
        cdef Fit fit = Fit(0, 0, 0)

        cdef float t_bar_mid = (t_bar[0] + t_bar[- 1]) / 2

        # Casertano+2022 Eq. 44
        # Note we've departed from Casertano+22 slightly;
        # there s is just resultants[ramp.end].  But that doesn't seem good if, e.g.,
        # a CR in the first resultant has boosted the whole ramp high but there
        # is no actual signal.
        cdef float s = max(resultants[-1] - resultants[0], 0)
        s = s / sqrt(self.fixed.read_noise**2 + s)
        cdef float power = get_power(s)

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
            fit.slope += coeffs[i] * resultants[i]

            # Casertano+22 Eq. 39
            fit.read_var += (coeffs[i] ** 2 * self.fixed.read_noise ** 2 / n_reads[i])

            # Casertano+22 Eq 40
            fit.poisson_var += coeffs[i] ** 2 * tau[i]
            for j in range(i + 1, n_resultants):
                fit.poisson_var += (2 * coeffs[i] * coeffs[j] * t_bar[i])

        return fit

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline float[:] stats(Ramp self, float slope, RampIndex ramp):
        cdef np.ndarray[float] delta_1 = np.array(self.delta_1[ramp.start:ramp.end-1]) - slope
        cdef np.ndarray[float] delta_2 = np.array(self.delta_2[ramp.start:ramp.end-1]) - slope

        cdef np.ndarray[float] var_1 = ((np.array(self.sigma_1[ramp.start:ramp.end-1]) +
                                         slope * np.array(self.slope_var_1[ramp.start:ramp.end-1])) /
                                        self.fixed.t_bar_1_sq[ramp.start:ramp.end-1]).astype(np.float32)
        cdef np.ndarray[float] var_2 = ((np.array(self.sigma_2[ramp.start:ramp.end-1]) +
                                         slope * np.array(self.slope_var_2[ramp.start:ramp.end-1])) /
                                        self.fixed.t_bar_2_sq[ramp.start:ramp.end-1]).astype(np.float32)

        cdef np.ndarray[float] stats_1 = delta_1 / sqrt(var_1)
        cdef np.ndarray[float] stats_2 = delta_2 / sqrt(var_2)

        return np.maximum(stats_1, stats_2)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline Fits fits(Ramp self, stack[RampIndex] ramps, Thresh thresh):
        cdef Fits fits

        cdef RampIndex ramp
        cdef Fit fit
        cdef float [:] stats
        cdef int split

        while not ramps.empty():
            ramp = ramps.top()
            ramps.pop()
            fit = self.fit(ramp)
            stats = self.stats(fit.slope, ramp)
            
            if max(stats) > threshold(thresh, fit.slope) and self.fixed.use_jump:
                split = np.argmax(stats)

                ramps.push(RampIndex(ramp.start, ramp.start + split))
                ramps.push(RampIndex(ramp.start + split + 2, ramp.end))
            else:
                fits.slope.push_back(fit.slope)
                fits.read_var.push_back(fit.read_var)
                fits.poisson_var.push_back(fit.poisson_var)

        return reverse_fits(fits)


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