import numpy as np
cimport numpy as cnp
cimport cython

from libcpp cimport bool
from libc.math cimport sqrt, log10


from stcal.ramp_fitting.ols_cas22._core cimport Diff
from stcal.ramp_fitting.ols_cas22._fixed cimport FixedValues
from stcal.ramp_fitting.ols_cas22._jump cimport Thresh, RampFits
from stcal.ramp_fitting.ols_cas22._pixel cimport Pixel
from stcal.ramp_fitting.ols_cas22._ramp cimport RampIndex, RampQueue, fit_ramp
from stcal.ramp_fitting.ols_cas22._read_pattern cimport ReadPattern

cpdef inline float threshold(Thresh thresh, float slope):
    """
    Compute jump threshold

    Parameters
    ----------
    thresh : Thresh
        threshold parameters struct
    slope : float
        slope of the ramp in question

    Returns
    -------
        intercept - constant * log10(slope)
    """
    slope = slope if slope > 1 else 1
    slope = slope if slope < 1e4 else 1e4

    return thresh.intercept - thresh.constant * log10(slope)




@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float correction(ReadPattern data, RampIndex ramp, float slope):
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

    cdef float diff = (data.t_bar[ramp.end] - data.t_bar[ramp.start])

    return - slope / diff

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline float statstic(Pixel pixel, float slope, RampIndex ramp, int index, int diff):
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
    cdef FixedValues fixed = pixel.fixed

    cdef float delta = (pixel.local_slopes[diff, index] - slope)
    cdef float var = ((pixel.var_read_noise[diff, index] +
                        slope * fixed.var_slope_coeffs[diff, index])
                        / fixed.t_bar_diff_sqrs[diff, index]) 
    cdef float correct = correction(fixed.data, ramp, slope)

    return (delta / sqrt(var + correct))


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float[:] statistics(Pixel pixel, float slope, RampIndex ramp):
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

    cdef cnp.ndarray[float, ndim=1] stats = np.zeros(end - start, dtype=np.float32)

    cdef int index, stat
    for stat, index in enumerate(range(start, end)):
        if index == end - 1:
            # It is not possible to compute double differences for the second
            # to last resultant in the ramp. Therefore, we just compute the
            # single difference for this resultant.
            stats[stat] = statstic(pixel, slope, ramp, index, Diff.single)
        else:
            stats[stat] = max(statstic(pixel, slope, ramp, index, Diff.single),
                              statstic(pixel, slope, ramp, index, Diff.double))

    return stats
    

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline RampFits fit_jumps(Pixel pixel, RampQueue ramps, Thresh thresh, bool include_diagnostic):
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
        ramp_fit = fit_ramp(pixel, ramp)

        # Run jump detection if enabled
        if pixel.fixed.use_jump:
            stats = statistics(pixel, ramp_fit.slope, ramp)

            # We have to protect against the case where the passed "ramp" is
            # only a single point. In that case, stats will be empty. This
            # will create an error in the max() call. 
            if len(stats) > 0 and max(stats) > threshold(thresh, ramp_fit.slope):
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