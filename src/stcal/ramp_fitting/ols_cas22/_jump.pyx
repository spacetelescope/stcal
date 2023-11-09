from cython cimport boundscheck, wraparound, cdivision

from libcpp cimport bool
from libc.math cimport sqrt, log10, fmaxf, NAN, isnan


from stcal.ramp_fitting.ols_cas22._core cimport Diff
from stcal.ramp_fitting.ols_cas22._jump cimport Thresh, RampFits
from stcal.ramp_fitting.ols_cas22._fixed cimport FixedValues
from stcal.ramp_fitting.ols_cas22._pixel cimport PixelOffsets, fill_pixel_values
from stcal.ramp_fitting.ols_cas22._ramp cimport RampIndex, RampQueue, RampFit, fit_ramp

cdef inline float threshold(Thresh thresh, float slope):
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




@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef inline float correction(float[:] t_bar, RampIndex ramp, float slope):
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

    cdef float diff = t_bar[ramp.end] - t_bar[ramp.start]

    return - slope / diff

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef inline float statstic(float local_slope,
                           float var_read_noise,
                           float var_slope_coeff,
                           float t_bar_diff_sqr,
                           float slope,
                           float correct):
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

    cdef float delta = (local_slope - slope)
    cdef float var = ((var_read_noise +
                        slope * var_slope_coeff)
                        / t_bar_diff_sqr) 

    return (delta / sqrt(var + correct))


@boundscheck(False)
@wraparound(False)
cdef inline (int, float) statistics(float[:, :] pixel,
                                    float[:, :] var_slope_coeffs,
                                    float[:, :] t_bar_diff_sqrs,
                                    float[:] t_bar,
                                    float slope, RampIndex ramp):
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
    # Observe that the length of the ramp's sub array of the resultant would
    # be `end - start + 1`. However, we are computing single and double
    # "differences" which means we need to reference at least two points in
    # this subarray at a time. For the single case, the maximum index allowed
    # would be `end - 1`. Observe that `range(start, end)` will iterate over
    #    `start, start+1, start+1, ..., end-2, end-1`
    # as the second argument to the `range` is the first index outside of the
    # range
    cdef int start = ramp.start  # index of first resultant for ramp
    cdef int end = ramp.end      # index of last resultant for ramp

    # Case the enum values into integers for indexing
    cdef int single = Diff.single
    cdef int double = Diff.double

    cdef int single_local_slope = PixelOffsets.single_local_slope
    cdef int double_local_slope = PixelOffsets.double_local_slope
    cdef int single_var_read_noise = PixelOffsets.single_var_read_noise
    cdef int double_var_read_noise = PixelOffsets.double_var_read_noise

    cdef float correct = correction(t_bar, ramp, slope)

    cdef float stat, double_stat

    cdef int argmax = 0
    cdef float max_stat = NAN

    cdef int index, stat_index
    for stat_index, index in enumerate(range(start, end)):
        stat = statstic(pixel[single_local_slope, index],
                        pixel[single_var_read_noise, index],
                        var_slope_coeffs[single, index],
                        t_bar_diff_sqrs[single, index],
                        slope,
                        correct)

        # It is not possible to compute double differences for the second
        # to last resultant in the ramp. Therefore, we include the double
        # differences for every stat except the last one.
        if index != end - 1:
            double_stat = statstic(pixel[double_local_slope, index],
                                   pixel[double_var_read_noise, index],
                                   var_slope_coeffs[double, index],
                                   t_bar_diff_sqrs[double, index],
                                   slope,
                                   correct)
            stat = fmaxf(stat, double_stat)

        if isnan(max_stat) or stat > max_stat:
            max_stat = stat
            argmax = stat_index

    return argmax, max_stat
    

@boundscheck(False)
@wraparound(False)
@cdivision(True)
cdef inline RampFits fit_jumps(float[:] resultants,
                               float read_noise,
                               RampQueue ramps,
                               FixedValues fixed,
                               float[:, :] pixel,
                               Thresh thresh,
                               bool include_diagnostic):
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

    cdef int argmax, jump0, jump1
    cdef float max_stat
    cdef float weight, total_weight = 0

    cdef float[:] t_bar = fixed.data.t_bar
    cdef float[:] tau = fixed.data.tau
    cdef int[:] n_reads = fixed.data.n_reads

    cdef float[:, :] var_slope_coeffs
    cdef float[:, :] t_bar_diff_sqrs

    if fixed.use_jump:
        pixel = fill_pixel_values(pixel, resultants, fixed.t_bar_diffs, fixed.read_recip_coeffs, read_noise, fixed.data.n_resultants)
        var_slope_coeffs = fixed.var_slope_coeffs
        t_bar_diff_sqrs = fixed.t_bar_diff_sqrs
        t_bar = fixed.data.t_bar

    # Run while the stack is non-empty
    while not ramps.empty():
        # Remove top ramp of the stack to use
        ramp = ramps.back()
        ramps.pop_back()

        # Compute fit
        ramp_fit = fit_ramp(resultants,
                            t_bar,
                            tau,
                            n_reads,
                            read_noise,
                            ramp)

        # Run jump detection if enabled
        if fixed.use_jump:
            argmax, max_stat = statistics(pixel,
                                          var_slope_coeffs,
                                          t_bar_diff_sqrs,
                                          t_bar,
                                          ramp_fit.slope,
                                          ramp)

            # We have to protect against the case where the passed "ramp" is
            # only a single point. In that case, stats will be empty. This
            # will create an error in the max() call. 
            if not isnan(max_stat) and max_stat > threshold(thresh, ramp_fit.slope):
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
                jump0 = argmax + ramp.start
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
        if not isnan(ramp_fit.slope):
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