"""
Define the C class for the CAS22 algorithm for fitting ramps with jump detection

Objects
-------
Pixel : class
    Class to handle ramp fit with jump detection for a single pixel
    Provides fits method which fits all the ramps for a single pixel

Functions
---------
make_ramp : function
    Fast constructor for the Pixel class
"""
from libc.math cimport sqrt, fabs
from libcpp.vector cimport vector
from libcpp.stack cimport stack

import numpy as np
cimport numpy as np
cimport cython


from stcal.ramp_fitting.ols_cas22._core cimport get_power, threshold, RampFit, RampFits, RampIndex
from stcal.ramp_fitting.ols_cas22._pixel cimport Pixel


cdef class Pixel:
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
    cdef inline float[:] resultants_diff(Pixel self, int offset):
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
        cdef float[:] resultants = self.resultants
        cdef int end = len(resultants)

        return np.subtract(resultants[offset:], resultants[:end - offset])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline RampFit fit_ramp(Pixel self, RampIndex ramp):
        """
        Fit a single ramp using Casertano+22 algorithm.

        Parameters
        ----------
        ramp : RampIndex
            Struct for start and end of ramp to fit

        Returns
        -------
        RampFit struct of slope, read_var, poisson_var
        """
        cdef int n_resultants = ramp.end - ramp.start + 1
        cdef RampFit ramp_fit = RampFit(0, 0, 0)

        # Special case where there is no or one resultant, there is no fit.
        if n_resultants <= 1:
            return ramp_fit
        # Else, do the fitting.

        # Cast vectors to memory views for faster access
        #    This way of doing it is potentially memory unsafe because the memory
        #    can outlive the vector. However, this is much faster (no copies) and
        #    much simpler than creating an intermediate wrapper which can pretend
        #    to be a memory view. In this case, I make sure that the memory view
        #    stays local to the function t_bar, tau, n_reads are used only for
        #    computations whose results are stored in new objects, so they are local
        cdef float[:] t_bar_ = <float [:self.fixed.data.t_bar.size()]> self.fixed.data.t_bar.data()
        cdef float[:] tau_ = <float [:self.fixed.data.tau.size()]> self.fixed.data.tau.data()
        cdef int[:] n_reads_ = <int [:self.fixed.data.n_reads.size()]> self.fixed.data.n_reads.data()

        # Setup data for fitting (work over subset of data)
        cdef float[:] resultants = self.resultants[ramp.start:ramp.end + 1]
        cdef float[:] t_bar = t_bar_[ramp.start:ramp.end + 1]
        cdef float[:] tau = tau_[ramp.start:ramp.end + 1]
        cdef int[:] n_reads = n_reads_[ramp.start:ramp.end + 1]
        cdef float read_noise = self.read_noise

        cdef int end = len(resultants) - 1

        # initalize fit
        cdef int i = 0, j = 0
        cdef vector[float] weights = vector[float](n_resultants)
        cdef vector[float] coeffs = vector[float](n_resultants)
        cdef float t_bar_mid = (t_bar[0] + t_bar[end]) / 2

        # Casertano+2022 Eq. 44
        # Note we've departed from Casertano+22 slightly;
        # there s is just resultants[ramp.end].  But that doesn't seem good if, e.g.,
        # a CR in the first resultant has boosted the whole ramp high but there
        # is no actual signal.
        cdef float s = max(resultants[end] - resultants[0], 0)
        s = s / sqrt(read_noise**2 + s)
        cdef float power = get_power(s)

        # It's easy to use up a lot of dynamic range on something like
        # (tbar - tbarmid) ** 10.  Rescale these.
        cdef float t_scale = (t_bar[end] - t_bar[0]) / 2
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
            return ramp_fit

        for i in range(n_resultants):
            # Casertano+22 Eq. 37
            coeffs[i] = (f0 * t_bar[i] - f1) * weights[i] / det

        for i in range(n_resultants):
            # Casertano+22 Eq. 38
            ramp_fit.slope += coeffs[i] * resultants[i]

            # Casertano+22 Eq. 39
            ramp_fit.read_var += (coeffs[i] ** 2 * read_noise ** 2 / n_reads[i])

            # Casertano+22 Eq 40
            ramp_fit.poisson_var += coeffs[i] ** 2 * tau[i]
            for j in range(i + 1, n_resultants):
                ramp_fit.poisson_var += (2 * coeffs[i] * coeffs[j] * t_bar[i])

        return ramp_fit

    cdef inline float correction(Pixel self, int i, int offset, RampIndex ramp):
        cdef float comp = ((self.fixed.data.t_bar[i + offset] - self.fixed.data.t_bar[i]) /
                           (self.fixed.data.t_bar[ramp.end] - self.fixed.data.t_bar[ramp.start]))

        if offset == 1:
            return (1 - comp)**2
        elif offset == 2:
            return (1 - 0.75 * comp)**2
        else:
            raise ValueError("offset must be 1 or 2")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline float[:] stats(Pixel self, float slope, RampIndex ramp):
        """
        Compute fit statistics for jump detection on a single ramp
        Computed using:
            corr_1[i] = correction(i, 1, ramp)
            corr_2[i] = correction(i, 2, ramp)

            var_1[i] = ((sigma_1[i] + slope * slope_var_1[i] * corr_1[i]) / t_bar_1_sq[i])
            var_2[i] = ((sigma_2[i] + slope * slope_var_2[i] * corr_2[i]) / t_bar_2_sq[i])

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
        cdef int start = ramp.start
        cdef int end = ramp.end - 1

        cdef float[:] slope_var_1 = np.zeros(end - start + 1, dtype=np.float32)
        cdef float[:] slope_var_2 = np.zeros(end - start + 1, dtype=np.float32)
        cdef int i
        for i in range(end - start + 1):
            slope_var_1[i] = slope * self.fixed.slope_var_1[start + i] * self.correction(start + i, 1, ramp)
            slope_var_2[i] = slope * self.fixed.slope_var_2[start + i] * self.correction(start + i, 2, ramp)

        cdef float[:] delta_1 = np.subtract(self.delta_1[start:end], slope)
        cdef float[:] delta_2 = np.subtract(self.delta_2[start:end], slope)

        cdef float[:] var_1 = np.divide(np.add(self.sigma_1[start:end], slope_var_1),
                                        self.fixed.t_bar_1_sq[start:end])
        cdef float[:] var_2 = np.divide(np.add(self.sigma_2[start:end], slope_var_2),
                                        self.fixed.t_bar_2_sq[start:end])

        cdef float[:] stats_1 = np.divide(delta_1, np.sqrt(var_1))
        cdef float[:] stats_2 = np.divide(delta_2, np.sqrt(var_2))

        return np.maximum(stats_1, stats_2, dtype=np.float32)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline RampFits fit_ramps(Pixel self, stack[RampIndex] ramps):
        """
        Compute all the ramps for a single pixel using the Casertano+22 algorithm
            with jump detection.

        Parameters
        ----------
        ramps : stack[RampIndex]
            Stack of initial ramps to fit for a single pixel
            multiple ramps are possible due to dq flags

        Returns
        -------
        RampFits struct of all the fits for a single pixel
        """
        # Setup algorithm
        cdef RampFits ramp_fits
        cdef RampIndex ramp
        cdef RampFit ramp_fit
        cdef float [:] stats
        cdef int split
        cdef float weight, total_weight = 0

        # Run while the stack is non-empty
        while not ramps.empty():
            # Remove top ramp of the stack to use
            ramp = ramps.top()
            ramps.pop()

            # Compute fit
            ramp_fit = self.fit_ramp(ramp)

            if self.fixed.use_jump:
                stats = self.stats(ramp_fit.slope, ramp)

                if max(stats) > threshold(self.threshold, ramp_fit.slope):
                    # Compute split point to create two new ramps
                    split = np.argmax(stats)

                    # add ramps so last ramp in time is on top of stack
                    ramps.push(RampIndex(ramp.start, ramp.start + split))
                    ramps.push(RampIndex(ramp.start + split + 2, ramp.end))

                    # Return to top of loop to fit new ramps (without adding to fits)
                    continue

            # Add ramp_fit to ramp_fits if no jump detection or stats are less
            #    than threshold
            # Note that ramps are computed backward in time meaning we need to
            #  reverse the order of the fits at the end
            ramp_fits.fits.push_back(ramp_fit)
            ramp_fits.index.push_back(ramp)

            # Start computing the averages
            weight = 0 if ramp_fit.read_var == 0 else 1 / ramp_fit.read_var
            total_weight += weight

            ramp_fits.average.slope += weight * ramp_fit.slope
            ramp_fits.average.read_var += weight**2 * ramp_fit.read_var
            ramp_fits.average.poisson_var += weight**2 * ramp_fit.poisson_var

        # Reverse to order in time
        ramp_fits.fits = ramp_fits.fits[::-1]
        ramp_fits.index = ramp_fits.index[::-1]

        # Finish computing averages
        ramp_fits.average.slope /= total_weight if total_weight != 0 else 1
        ramp_fits.average.read_var /= total_weight**2 if total_weight != 0 else 1
        ramp_fits.average.poisson_var /= total_weight**2 if total_weight != 0 else 1

        # Multiply poisson term by flux, (no negative fluxes)
        ramp_fits.average.poisson_var *= max(ramp_fits.average.slope, 0)

        return ramp_fits


cdef inline Pixel make_pixel(Fixed fixed, float read_noise, float [:] resultants):
    """
    Fast constructor for the Pixel C class.

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
    Pixel C-class object (with pre-computed values if use_jump is True)
    """
    cdef Pixel pixel = Pixel()

    # Fill in input information for pixel
    pixel.fixed = fixed
    pixel.read_noise = read_noise
    pixel.resultants = resultants

    # Pre-compute values for jump detection shared by all pixels for this pixel
    if fixed.use_jump:
        pixel.delta_1 = (np.array(pixel.resultants_diff(1)) /
                         np.array(fixed.t_bar_1)).astype(np.float32)
        pixel.delta_2 = (np.array(pixel.resultants_diff(2)) /
                         np.array(fixed.t_bar_2)).astype(np.float32)

        pixel.sigma_1 = read_noise * np.array(fixed.recip_1)
        pixel.sigma_2 = read_noise * np.array(fixed.recip_2)

    return pixel
