"""
Define the C class for the Cassertano22 algorithm for fitting ramps with jump detection

Objects
-------
Pixel : class
    Class to handle ramp fit with jump detection for a single pixel
    Provides fits method which fits all the ramps for a single pixel

Functions
---------
    make_pixel : function
        Fast constructor for a Pixel class from input data.
            - cpdef gives a python wrapper, but the python version of this method
              is considered private, only to be used for testing
"""
from libcpp cimport bool
from libc.math cimport sqrt, fabs
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np
cimport cython


from stcal.ramp_fitting.ols_cas22._core cimport get_power, threshold, RampFit, RampFits, RampIndex, Diff
from stcal.ramp_fitting.ols_cas22._fixed cimport FixedValues
from stcal.ramp_fitting.ols_cas22._pixel cimport Pixel


cdef class Pixel:
    """
    Class to contain the data to fit ramps for a single pixel.
        This data is drawn from for all ramps for a single pixel.
        This class pre-computes jump detection values shared by all ramps
        for a given pixel.

    Parameters
    ----------
    fixed : FixedValues
        The object containing all the values and metadata which is fixed for a
        given read pattern>
    read_noise : float
        The read noise for the given pixel
    resultants : float [:]
        Resultants input for the given pixel

    local_slopes : float [:, :]
        These are the local slopes between the resultants for the pixel.
            single difference local slope:
                local_slopes[Diff.single, :] = (resultants[i+1] - resultants[i])
                                                / (t_bar[i+1] - t_bar[i])
            double difference local slope:
                local_slopes[Diff.double, :] = (resultants[i+2] - resultants[i])
                                                / (t_bar[i+2] - t_bar[i])
    var_read_noise : float [:, :]
        The read noise variance term of the jump statistics
            single difference read noise variance:
                var_read_noise[Diff.single, :] = read_noise * ((1/n_reads[i+1]) + (1/n_reads[i]))
            double difference read_noise variance:
                var_read_noise[Diff.doule, :] = read_noise * ((1/n_reads[i+2]) + (1/n_reads[i]))

    Notes
    -----
    - local_slopes and var_read_noise are only computed if use_jump is True. 
      These values represent reused computations for jump detection which are
      used by every ramp for the given pixel for jump detection. They are
      computed once and stored for reuse by all ramp computations for the pixel.
    - The computations are done using vectorized operations for some performance
      increases. However, this is marginal compaired with the performance increase
      from pre-computing the values and reusing them.

    Methods
    -------
    fit_ramp (ramp_index) : method
        Compute the ramp fit for a single ramp defined by an inputed RampIndex
    fit_ramps (ramp_stack) : method
        Compute all the ramps for a single pixel using the Casertano+22 algorithm
        with jump detection.
    """

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline float[:, :] local_slope_vals(Pixel self):
        """
        Compute the local slopes between resultants for the pixel

        Returns
        -------
        [
            <(resultants[i+1] - resultants[i])> / <(t_bar[i+1] - t_bar[i])>,
            <(resultants[i+2] - resultants[i])> / <(t_bar[i+2] - t_bar[i])>,
        ]
        """
        cdef float[:] resultants = self.resultants
        cdef int end = len(resultants)

        # Read the t_bar_diffs into a local variable to avoid calling through Python
        #    multiple times
        cdef np.ndarray[float, ndim=2] t_bar_diffs = np.array(self.fixed.t_bar_diffs, dtype=np.float32)

        cdef np.ndarray[float, ndim=2] local_slope_vals = np.zeros((2, end - 1), dtype=np.float32)

        local_slope_vals[Diff.single, :] = (np.subtract(resultants[1:], resultants[:end - 1])
                                            / t_bar_diffs[Diff.single, :]).astype(np.float32)
        local_slope_vals[Diff.double, :end - 2] = (np.subtract(resultants[2:], resultants[:end - 2])
                                                   / t_bar_diffs[Diff.double, :end-2]).astype(np.float32)
        local_slope_vals[Diff.double, end - 2] = np.nan  # last double difference is undefined

        return local_slope_vals

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

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef inline float correction(Pixel self, RampIndex ramp, float slope):
        """
        Compute the correction factor for the variance used by a statistic

            - slope / (t_bar[end] - t_bar[start])
        
        Parameters
        ----------
        ramp : RampIndex
            Struct for start and end indices resultants for the ramp
        slope : float
            The computed slope for the ramp
        """

        cdef float diff = (self.fixed.data.t_bar[ramp.end] - self.fixed.data.t_bar[ramp.start])

        return - slope / diff

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline float stat(Pixel self, float slope, RampIndex ramp, int index, int diff):
        """
        Compute a single set of fit statistics
            (delta / sqrt(var)) + correction
        where
            delta = ((R[j] - R[i]) / (t_bar[j] - t_bar[i]) - slope)
                    * (t_bar[j] - t_bar[i])
            var   = sigma * (1/N[j] + 1/N[i]) 
                    + slope * (tau[j] + tau[i] - min(t_bar[j], t_bar[i]))
        
        Parameters
        ----------
        slope : float
            The computed slope for the ramp
        ramp : RampIndex
            Struct for start and end indices resultants for the ramp
        index : int
            The main index for the resultant to compute the statistic for
        diff : int
            The offset to use for the delta and sigma values, this should be
            a value from the Diff enum.

        Returns
        -------
            Create a single instance of the stastic for the given parameters
        """
        cdef float delta = (self.local_slopes[diff, index] - slope)
        cdef float var = ((self.var_read_noise[diff, index] +
                           slope * self.fixed.var_slope_coeffs[diff, index])
                          / self.fixed.t_bar_diff_sqrs[diff, index]) 
        cdef float correct = self.correction(ramp, slope)

        return (delta / sqrt(var)) + correct


    @cython.boundscheck(False)
    @cython.wraparound(False)
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
                stats[stat] = max(self.stat(slope, ramp, index, Diff.single),
                                  self.stat(slope, ramp, index, Diff.double))

        return stats
        

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef inline RampFits fit_ramps(Pixel self, vector[RampIndex] ramps, bool include_diagnostic):
        """
        Compute all the ramps for a single pixel using the Casertano+22 algorithm
            with jump detection.

        Parameters
        ----------
        ramps : vector[RampIndex]
            Vector of initial ramps to fit for a single pixel
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
        cdef int jump0, jump1
        cdef float weight, total_weight = 0

        # Run while the stack is non-empty
        while not ramps.empty():
            # Remove top ramp of the stack to use
            ramp = ramps.back()
            ramps.pop_back()

            # Compute fit
            ramp_fit = self.fit_ramp(ramp)

            # Run jump detection if enabled
            if self.fixed.use_jump:
                stats = self.stats(ramp_fit.slope, ramp)

                # We have to protect against the case where the passed "ramp" is
                # only a single point. In that case, stats will be empty. This
                # will create an error in the max() call. 
                if len(stats) > 0 and max(stats) > threshold(self.fixed.threshold, ramp_fit.slope):
                    # Compute jump point to create two new ramps
                    #    This jump point corresponds to the index of the largest
                    #    statistic:
                    #        argmax(stats)
                    #    These statistics are indexed relative to the
                    #    ramp's range. Therefore, we need to add the start index
                    #    of the ramp to the result.
                    #
                    # Note that because the resultants are averages of reads, but
                    # jumps occur in individual reads, it is possible that the
                    # jump is averaged down by the resultant with the actual jump
                    # causing the computed jump to be off by one index.
                    #     In the idealized case this is when the jump occurs near
                    #     the start of the resultant with the jump. In this case,
                    #     the statistic for the resultant will be maximized at
                    #     index - 1 rather than index. This means that we have to
                    #     remove argmax(stats) + 1 as it is also a possible jump.
                    #     This case is difficult to distinguish from the case where
                    #     argmax(stats) does correspond to the jump resultant.
                    #     Therefore, we just remove both possible resultants from
                    #     consideration.
                    jump0 = np.argmax(stats) + ramp.start
                    jump1 = jump0 + 1
                    if include_diagnostic:
                        ramp_fits.jumps.push_back(jump0)
                        ramp_fits.jumps.push_back(jump1)

                    # The two resultant indicies need to be skipped, therefore
                    # the two
                    # possible new ramps are:
                    #     RampIndex(ramp.start, jump0 - 1)
                    #     RampIndex(jump1 + 1, ramp.end)
                    # This is because the RampIndex contains the index of the
                    # first and last resulants in the sub-ramp it describes.
                    #    Note: The algorithm works via working over the sub-ramps
                    #    backward in time. Therefore, since we are using a stack,
                    #    we need to add the ramps in the time order they were
                    #    observed in. This results in the last observation ramp
                    #    being the top of the stack; meaning that,
                    #    it will be the next ramp handeled.

                    if jump0 > ramp.start:
                        # Note that when jump0 == ramp.start, we have detected a
                        # jump in the first resultant of the ramp. This means
                        # there is no sub-ramp before jump0.
                        #    Also, note that this will produce bad results as
                        #    the ramp indexing will go out of bounds. So it is
                        #    important that we exclude it.
                        # Note that jump0 < ramp.start is not possible because
                        # the argmax is always >= 0
                        ramps.push_back(RampIndex(ramp.start, jump0 - 1))

                    if jump1 < ramp.end:
                        # Note that if jump1 == ramp.end, we have detected a
                        # jump in the last resultant of the ramp. This means
                        # there is no sub-ramp after jump1.
                        #    Also, note that this will produce bad results as
                        #    the ramp indexing will go out of bounds. So it is
                        #    important that we exclude it.
                        # Note that jump1 > ramp.end is technically possible
                        # however in those potential cases it will draw on
                        # resultants which are not considered part of the ramp
                        # under consideration. Therefore, we have to exlude all
                        # of those values.
                        ramps.push_back(RampIndex(jump1 + 1, ramp.end))

                    continue

            # Add ramp_fit to ramp_fits if no jump detection or stats are less
            #    than threshold
            # Note that ramps are computed backward in time meaning we need to
            #  reverse the order of the fits at the end
            if include_diagnostic:
                ramp_fits.fits.push_back(ramp_fit)
                ramp_fits.index.push_back(ramp)

            # Start computing the averages
            #    Note we do not do anything in the NaN case for degenerate ramps
            if not np.isnan(ramp_fit.slope):
                # protect weight against the extremely unlikely case of a zero
                # variance
                weight = 0 if ramp_fit.read_var == 0 else 1 / ramp_fit.read_var
                total_weight += weight

                ramp_fits.average.slope += weight * ramp_fit.slope
                ramp_fits.average.read_var += weight**2 * ramp_fit.read_var
                ramp_fits.average.poisson_var += weight**2 * ramp_fit.poisson_var

        # Reverse to order in time
        if include_diagnostic:
            ramp_fits.fits = ramp_fits.fits[::-1]
            ramp_fits.index = ramp_fits.index[::-1]

        # Finish computing averages
        ramp_fits.average.slope /= total_weight if total_weight != 0 else 1
        ramp_fits.average.read_var /= total_weight**2 if total_weight != 0 else 1
        ramp_fits.average.poisson_var /= total_weight**2 if total_weight != 0 else 1

        # Multiply poisson term by flux, (no negative fluxes)
        ramp_fits.average.poisson_var *= max(ramp_fits.average.slope, 0)

        return ramp_fits

    def _to_dict(Pixel self):
        """
        This is a private method to convert the Pixel object to a dictionary, so
            that attributes can be directly accessed in python. Note that this is
            needed because class attributes cannot be accessed on cython classes
            directly in python. Instead they need to be accessed or set using a
            python compatible method. This method is a pure puthon method bound
            to to the cython class and should not be used by any cython code, and
            only exists for testing purposes.
        """

        cdef np.ndarray[float, ndim=1] resultants_ = np.array(self.resultants, dtype=np.float32)

        cdef np.ndarray[float, ndim=2] local_slopes
        cdef np.ndarray[float, ndim=2] var_read_noise

        if self.fixed.use_jump:
            local_slopes = np.array(self.local_slopes, dtype=np.float32)
            var_read_noise = np.array(self.var_read_noise, dtype=np.float32)
        else:
            try:
                self.local_slopes
            except AttributeError:
                local_slopes = np.array([[np.nan],[np.nan]], dtype=np.float32)
            else:
                raise AttributeError("local_slopes should not exist")

            try:
                self.var_read_noise
            except AttributeError:
                var_read_noise = np.array([[np.nan],[np.nan]], dtype=np.float32)
            else:
                raise AttributeError("var_read_noise should not exist")

        return dict(fixed=self.fixed._to_dict(),
                    resultants=resultants_,
                    read_noise=self.read_noise,
                    local_slopes=local_slopes,
                    var_read_noise=var_read_noise)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline Pixel make_pixel(FixedValues fixed, float read_noise, float [:] resultants):
    """
    Fast constructor for the Pixel C class.
        This creates a Pixel object for a single pixel from the input data.

    This is signifantly faster than using the `__init__` or `__cinit__`
        this is because this does not have to pass through the Python as part
        of the construction.

    Parameters
    ----------
    fixed : FixedValues
        Fixed values for all pixels
    read_noise : float
        read noise for the single pixel
    resultants : float [:]
        array of resultants for the single pixel
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
        pixel.local_slopes = pixel.local_slope_vals()
        pixel.var_read_noise = read_noise * np.array(fixed.read_recip_coeffs)

    return pixel
