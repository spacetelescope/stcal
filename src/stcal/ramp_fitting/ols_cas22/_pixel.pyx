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


from stcal.ramp_fitting.ols_cas22._core cimport get_power, threshold, RampFit, RampFits, RampIndex, Diff
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

    delta : float [:, :]
        single difference delta+slope:
            delta[0, :] = (resultants[i+1] - resultants[i]) / (t_bar[i+1] - t_bar[i])
        double difference delta+slope:
            delta[1, :] = (resultants[i+2] - resultants[i]) / (t_bar[i+2] - t_bar[i])
    sigma : float [:, :]
        single difference "sigma":
            sigma[0, :] = read_noise * ((1/n_reads[i+1]) + (1/n_reads[i]))
        double difference "sigma":
            sigma[1, :] = read_noise * ((1/n_reads[i+2]) + (1/n_reads[i]))

    Notes
    -----
    - delta, sigma are only computed if use_jump is True.  These values represent
      reused computations for jump detection which are used by every ramp for
      the given pixel for jump detection. They are computed once and stored for
      reuse by all ramp computations for the pixel.
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
    cdef inline float[:, :] delta_val(Pixel self):
        """
        Compute the difference offset of resultants

        Returns
        -------
        [
            <(resultants[i+1] - resultants[i])>,
            <(resultants[i+2] - resultants[i])>,
        ]
        """
        cdef float[:] resultants = self.resultants
        cdef int end = len(resultants)

        cdef np.ndarray[float, ndim=2] t_bar_diff = np.array(self.fixed.t_bar_diff, dtype=np.float32)
        cdef np.ndarray[float, ndim=2] delta = np.zeros((2, end - 1), dtype=np.float32)

        delta[Diff.single, :] = (np.subtract(resultants[1:], resultants[:end - 1]) / t_bar_diff[0, :]).astype(np.float32)
        delta[Diff.double, :end-2] = (np.subtract(resultants[2:], resultants[:end - 2]) / t_bar_diff[1, :end-2]).astype(np.float32)
        delta[Diff.double, end-2] = np.nan  # last double difference is undefined

        return delta

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

        # Special case where there is no or one resultant, there is no fit and
        # we bail out before any computations.
        #    Note that in this case, we cannot compute the slope or the variances
        #    because these computations require at least two resultants. Therefore,
        #    this case is degernate and we return NaNs for the values.
        if n_resultants <= 1:
            return RampFit(np.nan, np.nan, np.nan)

        # Start computing the fit

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
        #    Recall that the RampIndex contains the index of the first and last
        #    index of the ramp. Therefore, the Python slice needed to get all the
        #    data within the ramp is:
        #         ramp.start:ramp.end + 1
        cdef float[:] resultants = self.resultants[ramp.start:ramp.end + 1]
        cdef float[:] t_bar = t_bar_[ramp.start:ramp.end + 1]
        cdef float[:] tau = tau_[ramp.start:ramp.end + 1]
        cdef int[:] n_reads = n_reads_[ramp.start:ramp.end + 1]

        # Reference read_noise as a local variable to avoid calling through Python
        # every time it is accessed.
        cdef float read_noise = self.read_noise

        # Compute mid point time
        cdef int end = len(resultants) - 1
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

        # Initalize the fit loop
        cdef int i = 0, j = 0
        cdef vector[float] weights = vector[float](n_resultants)
        cdef vector[float] coeffs = vector[float](n_resultants)
        cdef RampFit ramp_fit = RampFit(0, 0, 0)
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

    cdef inline float correction(Pixel self, RampIndex ramp, int index, int diff):
        cdef float comp = (self.fixed.t_bar_diff[diff, index] /
                           (self.fixed.data.t_bar[ramp.end] - self.fixed.data.t_bar[ramp.start]))

        if diff == 0:
            return (1 - comp)**2
        elif diff == 1:
            return (1 - 0.75 * comp)**2
        else:
            raise ValueError("offset must be 1 or 2")

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline float stat(Pixel self, float slope, RampIndex ramp, int index, int diff):
        """
        Compute a single set of fit statistics
            delta / sqrt(var)
        where
            delta = ((R[j] - R[i]) / (t_bar[j] - t_bar[i]) - slope)
                    * (t_bar[j] - t_bar[i])
            var   = sigma * (1/N[j] + 1/N[i]) 
                    + slope * (tau[j] + tau[i] - min(t_bar[j], t_bar[i]))
                    * correction(offset)
        
        Parameters
        ----------
        slope : float
            The computed slope for the ramp
        ramp : RampIndex
            Struct for start and end indices resultants for the ramp
        index : int
            The main index for the resultant to compute the statistic for
        diff : int
            The offset to use for the delta and sigma values
                0 : single difference
                1 : double difference

        Returns
        -------
            Create a single instance of the stastic for the given parameters
        """
        cdef float delta = ((self.delta[diff, index] - slope) *
                            fabs(self.fixed.t_bar_diff[diff, index]))
        cdef float var = (self.sigma[diff, index] +
                          slope * self.fixed.slope_var[diff, index] *
                                  self.correction(ramp, index, diff))

        return delta / sqrt(var)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline float[:] stats(Pixel self, float slope, RampIndex ramp):
        """
        Compute fit statistics for jump detection on a single ramp
            stats[i] = max(stat(i, 0), stat(i, 1))
        Note for i == end - 1, no stat(i, 1) exists, so its just stat(i, 0)

        Parameters
        ----------
        slope : float
            The computed slope for the ramp
        ramp : RampIndex
            Struct for start and end of ramp to fit

        Returns
        -------
        list of statistics for each resultant
        """
        cdef int start = ramp.start  # index of first resultant for ramp
        cdef int end = ramp.end      # index of last resultant for ramp

        # Observe that the length of the ramp's sub array of the resultant would
        # be `end - start + 1`. However, we are computing single and double
        # "differences" which means we need to reference at least two points in
        # this subarray at a time. For the single case, the maximum index allowed
        # would be `end - 1`. Observe that `range(start, end)` will iterate over
        #    `start, start+1, start+1, ..., end-2, end-1`
        # as the second argument to the `range` is the first index outside of the
        # range

        cdef np.ndarray[float, ndim=1] stats = np.zeros(end - start, dtype=np.float32)

        cdef int index, stat
        for stat, index in enumerate(range(start, end)):
            if index == end - 1:
                # It is not possible to compute double differences for the second
                # to last resultant in the ramp. Therefore, we just compute the
                # single difference for this resultant.
                stats[stat] = self.stat(slope, ramp, index, Diff.single)
            else:
                stats[stat] = max(self.stat(slope, ramp, index, Diff.double),
                                  self.stat(slope, ramp, index, Diff.double))

        return stats
        

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

        ramp_fits.average.slope = 0
        ramp_fits.average.read_var = 0
        ramp_fits.average.poisson_var = 0

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

            # Run jump detection if enabled
            if self.fixed.use_jump:
                stats = self.stats(ramp_fit.slope, ramp)

                # We have to protect against the case where the passed "ramp" is only
                # a single point. In that case, stats will be empty. This will create
                # an error in the max() call. 
                if len(stats) > 0 and max(stats) > threshold(self.fixed.threshold, ramp_fit.slope):
                    # Compute split point to create two new ramps
                    #   The split will map to the index of the resultant with the detected jump
                    #       resultant_jump_index = ramp.start + split
                    #   This resultant index needs to be removed, therefore the two possible new
                    #   ramps are:
                    #       RampIndex(ramp.start, ramp.start + split - 1)
                    #       RampIndex(ramp.start + split + 1, ramp.end)
                    #   This is because the RampIndex contains the index of the first and last
                    #   resulants in the sub-ramp it describes.
                    split = np.argmax(stats)

                    # The algorithm works via working over the sub-ramps backward
                    #    in time. Therefore, since we are using a stack, we need to
                    #    add the ramps in the time order they were observed in. This
                    #    results in the last observation ramp being the top of the
                    #    stack; meaning that, it will be the next ramp handeled.

                    if split > 0:
                        # When split == 0, the jump has been detected in the resultant
                        # corresponding to the first resultant in the ramp, i.e
                        #    ramp.start
                        # So the "split" is just excluding the first resultant in the
                        # ramp currently being considered. Therefore, there is no need
                        # to handle a ramp in this case.
                        ramps.push(RampIndex(ramp.start, ramp.start + split - 1))

                    # Note that because the stats can only be calculated for ramp
                    # length - 1 # positions due to the need to compute at least
                    # single differences.  # Therefore the maximum value for
                    # argmax(stats) is ramp length - 2, as the index of the last
                    # element of stats is length of stats - 1. Thus
                    #     max(argmax(stats)) = len(stats) - 1
                    #                        = len(ramp) - 2
                    #                        = ramp.end - ramp.start - 1
                    # So we have that the maximium value for the lower index of
                    # this sub-ramp is
                    #     ramp.start + split + 1 = ramp.start + ramp.end 
                    #                                         - ramp.start - 1 + 1
                    #                            = ramp.end
                    # This is always a valid ramp.
                    ramps.push(RampIndex(ramp.start + split + 1, ramp.end))

                    # Return to top of loop to fit new ramps (without adding to fits)
                    continue

            # Add ramp_fit to ramp_fits if no jump detection or stats are less
            #    than threshold
            # Note that ramps are computed backward in time meaning we need to
            #  reverse the order of the fits at the end
            ramp_fits.fits.push_back(ramp_fit)
            ramp_fits.index.push_back(ramp)

            # Start computing the averages
            #    Note we do not do anything in the NaN case for degenerate ramps
            if not np.isnan(ramp_fit.slope):
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
        pixel.delta = pixel.delta_val()
        pixel.sigma = read_noise * np.array(fixed.recip)

    return pixel
