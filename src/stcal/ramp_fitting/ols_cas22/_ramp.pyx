"""
Define the C class for the CAS22 algorithm for fitting ramps with jump detection

Objects
-------
Ramp : class
    Class to handle ramp fit with jump detection for a single pixel
    Provides fits method which fits all the ramps for a single pixel

Functions
---------
make_ramp : function
    Fast constructor for the Ramp class
"""
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
    Class to contain the data to fit ramps for a single pixel.
        This data is drawn from for all ramps for a single pixel.
        This class pre-computes jump detection values shared by all ramps
        for a given pixel.
    
    Parameters
    ----------
    fixed : Fixed
        Fixed values for all pixels (pre-computed data)
    read_noise : float
        The read noise for the given pixel (data input)
    resultants : float [:]
        array of resultants for single pixel (data input)

    delta_1 : float [:]
        single difference delta+slope:
            (resultants[i+1] - resultants[i]) / (t_bar[i+1] - t_bar[i])
    delta_2 : float [:]
        double difference delta+slope:
            (resultants[i+2] - resultants[i]) / (t_bar[i+2] - t_bar[i])
    sigma_1 : float [:]
        single difference "sigma":
            read_noise * ((1/n_reads[i+1]) + (1/n_reads[i]))
    sigma_2 : float [:]
        double difference "sigma":
            read_noise * ((1/n_reads[i+2]) + (1/n_reads[i]))

    Notes
    -----
    - delta_*, sigma_* are only computed if use_jump is True.  These values
      represent reused computations for jump detection which are used by every
      ramp for the given pixel for jump detection. They are computed once and
      stored for reuse by all ramp computations for the pixel.
    - The computations are done using vectorized operations for some performance
      increases. However, this is marginal compaired with the performance increase
      from pre-computing the values and reusing them.

    Methods
    -------
    fits (ramp_stack) : method
        Compute all the ramps for a single pixel using the Casertano+22 algorithm
            with jump detection.
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
        Returns
        -------
        (resultants[i+offset] - resultants[i])
        """
        cdef int n_diff = len(self.resultants) - offset
        cdef float[:] diff = (np.roll(self.resultants, -offset) - self.t_bar)[:n_diff]

        return diff

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline Fit fit(Ramp self, RampIndex ramp):
        """
        Fit a single ramp using Casertano+22 algorithm.

        Parameters
        ----------
        ramp : RampIndex
            Struct for start and end of ramp to fit

        Returns
        -------
        Fit struct of slope, read_var, poisson_var
        """
        cdef int n_resultants = ramp.end - ramp.start + 1
        cdef Fit fit = Fit(0, 0, 0)

        # Special case where there is no or one resultant, there is no fit.
        if n_resultants <= 1:
            return fit
        # Else, do the fitting.

        # Setup data for fitting (work over subset of data)
        cdef float[:] resultants = self.fixed.resultants[ramp.start:ramp.end + 1]
        cdef float[:] t_bar = self.fixed.t_bar[ramp.start:ramp.end + 1]
        cdef float[:] tau = self.fixed.tau[ramp.start:ramp.end + 1]
        cdef int[:] n_reads = self.fixed.n_reads[ramp.start:ramp.end + 1]

        # initalize fit
        cdef int i = 0, j = 0
        cdef vector[float] weights = vector[float](n_resultants)
        cdef vector[float] coeffs = vector[float](n_resultants)
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
            return fit

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
        """
        Compute fit statistics for jump detection on a single ramp
        Computed using:

            var_1[i] = ((sigma_1[i] + slope * slope_var_1[i]) / t_bar_1_sq[i])
            var_2[i] = ((sigma_2[i] + slope * slope_var_2[i]) / t_bar_2_sq[i])

            s_1[i] = (delta_1[i] - slope) / sqrt(var_1[i])
            s_2[i] = (delta_2[i] - slope) / sqrt(var_2[i])

            stats[i] = max(s_1[i], s_2[i])
        Parameters
        ----------
        ramp : RampIndex
            Struct for start and end of ramp to fit

        Returns
        -------
        list of statistics for each resultant 
            except for the last 2 due to single/double difference due to indexing
        """
        cdef np.ndarray[float] delta_1 = np.array(self.delta_1[ramp.start:ramp.end-1]) - slope
        cdef np.ndarray[float] delta_2 = np.array(self.delta_2[ramp.start:ramp.end-1]) - slope

        cdef np.ndarray[float] var_1 = ((np.array(self.sigma_1[ramp.start:ramp.end-1]) +
                                         slope * np.array(self.slope_var_1[ramp.start:ramp.end-1])) /
                                        self.fixed.t_bar_1_sq[ramp.start:ramp.end-1]).astype(np.float32)
        cdef np.ndarray[float] var_2 = ((np.array(self.sigma_2[ramp.start:ramp.end-1]) +
                                         slope * np.array(self.slope_var_2[ramp.start:ramp.end-1])) /
                                        self.fixed.t_bar_2_sq[ramp.start:ramp.end-1]).astype(np.float32)

        cdef np.ndarray[float] stats_1 = (delta_1 / np.sqrt(var_1, dtype=np.float32)).astype(np.float32)
        cdef np.ndarray[float] stats_2 = (delta_2 / np.sqrt(var_2, dtype=np.float32)).astype(np.float32)

        return np.maximum(stats_1, stats_2)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline Fits fits(Ramp self, stack[RampIndex] ramps, Thresh thresh):
        """
        Compute all the ramps for a single pixel using the Casertano+22 algorithm
            with jump detection.

        Note: This algorithm computes the ramps for the pixel in reverse time order
            so that the last uncomputed ramp in time is always on top of the stack.
            This means we compute the slopes in reverse time order, so we have to
            reverse the order of the output data to be consistent with user
            expectations.

        Parameters
        ----------
        ramps : stack[RampIndex]
            Stack of initial ramps to fit for a single pixel
            multiple ramps are possible due to dq flags
        thresh : Thresh
            Thresholds struct passed in for jump detection
        
        Returns
        -------
        Fits struct of all the fits for a single pixel
        """
        # Setup algorithm
        cdef Fits fits
        cdef RampIndex ramp
        cdef Fit fit
        cdef float [:] stats
        cdef int split

        # Run while the stack is non-empty
        while not ramps.empty():
            # Remove top ramp of the stack to use
            ramp = ramps.top()
            ramps.pop()

            # Compute fit
            fit = self.fit(ramp)

            if self.fixed.use_jump:
                stats = self.stats(fit.slope, ramp)
            
                if max(stats) > threshold(thresh, fit.slope):
                    # Compute split point to create two new ramps
                    split = np.argmax(stats)

                    # add ramps so last ramp in time is on top of stack
                    ramps.push(RampIndex(ramp.start, ramp.start + split))
                    ramps.push(RampIndex(ramp.start + split + 2, ramp.end))

                    # Return to top of loop to fit new ramps (without adding to fits)
                    continue

            # Add fit to fits if no jump detection or stats are less than threshold
            fits.slope.push_back(fit.slope)
            fits.read_var.push_back(fit.read_var)
            fits.poisson_var.push_back(fit.poisson_var)

        # Reverse the slope data
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
    Ramp C-class object (with pre-computed values if use_jump is True)
    """
    cdef Ramp ramp = Ramp()

    # Fill in input information for pixel
    ramp.fixed = fixed
    ramp.read_noise = read_noise
    ramp.resultants = resultants

    # Pre-compute values for jump detection shared by all ramps for this pixel
    if fixed.use_jump:
        ramp.delta_1 = (np.array(ramp.resultants_diff(1)) / np.array(fixed.t_bar_1)).astype(np.float32)
        ramp.delta_2 = (np.array(ramp.resultants_diff(2)) / np.array(fixed.t_bar_2)).astype(np.float32)

        ramp.sigma_1 = read_noise * np.array(fixed.recip_1)
        ramp.sigma_2 = read_noise * np.array(fixed.recip_2)

    return ramp