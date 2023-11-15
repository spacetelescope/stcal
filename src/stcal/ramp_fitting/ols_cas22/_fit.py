# cython: language_level=3str

"""
External interface module for the Casertano+22 ramp fitting algorithm with jump detection.
    This module is intended to contain everything needed by external code.

Enums
-----
Parameter :
    Enumerate the index for the output parameters array.

Variance :
    Enumerate the index for the output variances array.

Classes
-------
RampFitOutputs : NamedTuple
    Simple tuple wrapper for outputs from the ramp fitting algorithm
        This clarifies the meaning of the outputs via naming them something
        descriptive.

(Public) Functions
------------------
fit_ramps : function
    Fit ramps using the Castenario+22 algorithm to a set of pixels accounting
    for jumps (if use_jump is True) and bad pixels (via the dq array). This
    is the primary externally callable function.
"""
import cython
from cython.cimports.libc.math import INFINITY, NAN, fabs, fmaxf, isnan, log10, sqrt
from cython.cimports.libcpp import bool as cpp_bool
from cython.cimports.libcpp.list import list as cpp_list
from cython.cimports.libcpp.vector import vector
from cython.cimports.stcal.ramp_fitting.ols_cas22._fit import (
    JUMP_DET,
    FixedOffsets,
    Parameter,
    PixelOffsets,
    Variance,
)

RampIndex = cython.struct(
    start=cython.int,
    end=cython.int,
)
RampQueue = cython.typedef(vector[RampIndex])
RampFit = cython.struct(
    slope=cython.float,
    read_var=cython.float,
    poisson_var=cython.float,
)
JumpFits = cython.struct(
    jumps=vector[cython.int],
    fits=vector[RampFit],
    index=RampQueue,
)
Thresh = cython.struct(
    intercept=cython.float,
    constant=cython.float,
)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.ccall
def fit_ramps(
    resultants: cython.float[:, :],
    dq: cython.int[:, :],
    read_noise: cython.float[:],
    read_time: cython.float,
    read_pattern: vector[vector[cython.int]],
    parameters: cython.float[:, :],
    variances: cython.float[:, :],
    t_bar: cython.float[:],
    tau: cython.float[:],
    n_reads: cython.int[:],
    single_pixel: cython.float[:, :],
    double_pixel: cython.float[:, :],
    single_fixed: cython.float[:, :],
    double_fixed: cython.float[:, :],
    use_jump: cpp_bool,
    intercept: cython.float,
    constant: cython.float,
    include_diagnostic: cpp_bool,
) -> cpp_list[JumpFits]:
    """Fit ramps using the Casertano+22 algorithm.
        This implementation uses the Cas22 algorithm to fit ramps, where
        ramps are fit between bad resultants marked by dq flags for each pixel
        which are not equal to zero. If use_jump is True, it additionally uses
        jump detection to mark additional resultants for each pixel as bad if
        a jump is suspected in them.

    Parameters
    ----------
    resultants : float[n_resultants, n_pixel]
        the resultants in electrons (Note that this can be based as any sort of
        array, such as a numpy array. The memory view is just for efficiency in
        cython)
    dq : np.ndarry[n_resultants, n_pixel]
        the dq array.  dq != 0 implies bad pixel / CR. (Kept as a numpy array
        so that it can be passed out without copying into new numpy array, will
        be working on memory views of this array)
    read_noise : float[n_pixel]
        the read noise in electrons for each pixel (same note as the resultants)
    read_time : float
        Time to perform a readout. For Roman data, this is FRAME_TIME.
    read_pattern : list[list[int]]
        the read pattern for the image
    use_jump : bool
        If True, use the jump detection algorithm to identify CRs.
        If False, use the DQ array to identify CRs.
    intercept : float
        The intercept value for the threshold function. Default=5.5
    constant : float
        The constant value for the threshold function. Default=1/3.0
    include_diagnostic : bool
        If True, include the raw ramp fits in the output. Default=False

    Returns
    -------
    A RampFitOutputs tuple
    """
    n_resultants: cython.int = resultants.shape[0]
    n_pixels: cython.int = resultants.shape[1]

    # Compute the main metadata from the read pattern and cast it to memory views
    _fill_metadata(t_bar, tau, n_reads, read_pattern, read_time, n_resultants)

    if use_jump:
        # Pre-compute the values from the read pattern
        _fill_fixed_values(single_fixed, double_fixed, t_bar, tau, n_reads, n_resultants)

    # Create a threshold struct
    thresh: Thresh = Thresh(intercept, constant)

    # Create variable to old the diagnostic data
    # Use list because this might grow very large which would require constant
    #    reallocation. We don't need random access, and this gets cast to a python
    #    list in the end.
    ramp_fits: cpp_list[JumpFits] = cpp_list[JumpFits]()

    # Run the jump fitting algorithm for each pixel
    fit: JumpFits
    index: cython.int
    for index in range(n_pixels):
        # Fit all the ramps for the given pixel
        fit = _fit_pixel(
            parameters[index, :],
            variances[index, :],
            resultants[:, index],
            dq[:, index],
            read_noise[index],
            t_bar,
            tau,
            n_reads,
            n_resultants,
            single_pixel,
            double_pixel,
            single_fixed,
            double_fixed,
            thresh,
            use_jump,
            include_diagnostic,
        )

        # Store diagnostic data if requested
        if include_diagnostic:
            ramp_fits.push_back(fit)

    # Cast memory views into numpy arrays for ease of use in python.
    return ramp_fits


_slope = cython.declare(cython.int, Parameter.slope)
_read_var = cython.declare(cython.int, Variance.read_var)
_poisson_var = cython.declare(cython.int, Variance.poisson_var)
_total_var = cython.declare(cython.int, Variance.total_var)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.inline
@cython.cfunc
def _fit_pixel(
    parameters: cython.float[:],
    variances: cython.float[:],
    resultants: cython.float[:],
    dq: cython.int[:],
    read_noise: cython.float,
    t_bar: cython.float[:],
    tau: cython.float[:],
    n_reads: cython.int[:],
    n_resultants: cython.int,
    single_pixel: cython.float[:, :],
    double_pixel: cython.float[:, :],
    single_fixed: cython.float[:, :],
    double_fixed: cython.float[:, :],
    thresh: Thresh,
    use_jump: cpp_bool,
    include_diagnostic: cpp_bool,
) -> JumpFits:
    """
    Compute all the ramps for a single pixel using the Casertano+22 algorithm
        with jump detection.

    Parameters
    ----------
    resultants : float[:]
        The resultants for the pixel
    dq : int[:]
        The dq flags for the pixel. This is modified in place, so the external
        dq flag array will be modified as a side-effect.
    read_noise : float
        The read noise for the pixel.
    ramps : RampQueue
        RampQueue for initial ramps to fit for the pixel
        multiple ramps are possible due to dq flags
    t_bar : float[:]
        The average time for each resultant
    tau : float[:]
        The time variance for each resultant
    n_reads : int[:]
        The number of reads for each resultant
    n_resultants : int
        The number of resultants for the pixel
    fixed : float[:, :]
        The jump detection pre-computed values for a given read_pattern
    pixel : float[:, :]
        A pre-allocated array for the jump detection fixed values for the
        given pixel. This will be modified in place, it is passed in to avoid
        re-allocating it for each pixel.
    thresh : Thresh
        The threshold parameter struct for jump detection
    use_jump : bool
        Turn on or off jump detection.
    include_diagnostic : bool
        Turn on or off recording all the diaganostic information on the fit

    Returns
    -------
    RampFits struct of all the fits for a single pixel
    """
    # Find initial set of ramps
    ramps: RampQueue = _init_ramps(dq, n_resultants)

    # Initialize algorithm
    parameters[:] = 0
    variances[:] = 0

    jumps: vector[cython.int] = vector[cython.int]()
    fits: vector[RampFit] = vector[RampFit]()
    index: RampQueue = RampQueue()

    # Declare variables for the loop
    ramp: RampIndex
    ramp_fit: RampFit
    stat: Stat
    jump0: cython.int
    jump1: cython.int
    weight: cython.float
    total_weight: cython.float = 0

    # Fill in the jump detection pre-compute values for a single pixel
    if use_jump:
        _fill_pixel_values(
            single_pixel, double_pixel, single_fixed, double_fixed, resultants, read_noise, n_resultants
        )

    # Run while the Queue is non-empty
    while not ramps.empty():
        # Remove top ramp of the stack to use
        ramp = ramps.back()
        ramps.pop_back()

        # Compute fit using the Casertano+22 algorithm
        ramp_fit = _fit_ramp(resultants, t_bar, tau, n_reads, read_noise, ramp)

        # Run jump detection if enabled
        if use_jump:
            stat = _fit_statistic(
                single_pixel, double_pixel, single_fixed, double_fixed, t_bar, ramp_fit.slope, ramp
            )

            # Note that when a "ramp" is a single point, _fit_statistic returns
            # a NaN for max_stat. Note that NaN > anything is always false so the
            # result drops through as desired.
            if stat.max_stat > _threshold(thresh, ramp_fit.slope):
                # Compute jump point to create two new ramps
                #    This jump point corresponds to the index of the largest
                #    statistic:
                #        argmax = argmax(stats)
                #    These statistics are indexed relative to the
                #    ramp's range. Therefore, we need to add the start index
                #    of the ramp to the result.
                #
                jump0 = stat.arg_max + ramp.start

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
                jump1 = jump0 + 1

                # Update the dq flags
                dq[jump0] = JUMP_DET
                dq[jump1] = JUMP_DET

                # Record jump diagnostics
                if include_diagnostic:
                    jumps.push_back(jump0)
                    jumps.push_back(jump1)

                # The two resultant indices need to be skipped, therefore
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
                #    it will be the next ramp handled.

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
                    # under consideration. Therefore, we have to exclude all
                    # of those values.
                    ramps.push_back(RampIndex(jump1 + 1, ramp.end))

                # Skip recording the ramp as it has a detected jump
                continue

        # Start recording the the fit (no jump detected)

        # Record the diagnositcs
        if include_diagnostic:
            fits.push_back(ramp_fit)
            index.push_back(ramp)

        # Start computing the averages using a lazy process
        #    Note we do not do anything in the NaN case for degenerate ramps
        if not isnan(ramp_fit.slope):
            # protect weight against the extremely unlikely case of a zero
            # variance
            weight = 0 if ramp_fit.read_var == 0 else 1 / ramp_fit.read_var
            total_weight += weight

            parameters[_slope] += weight * ramp_fit.slope
            variances[_read_var] += weight**2 * ramp_fit.read_var
            variances[_poisson_var] += weight**2 * ramp_fit.poisson_var

    # Finish computing averages using the lazy process
    parameters[_slope] /= total_weight if total_weight != 0 else 1
    variances[_read_var] /= total_weight**2 if total_weight != 0 else 1
    variances[_poisson_var] /= total_weight**2 if total_weight != 0 else 1

    # Multiply poisson term by flux, (no negative fluxes)
    variances[_poisson_var] *= max(parameters[_slope], 0)
    variances[_total_var] = variances[_read_var] + variances[_poisson_var]

    return JumpFits(jumps, fits, index)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.inline
@cython.cfunc
def _fit_ramp(
    resultants_: cython.float[:],
    t_bar_: cython.float[:],
    tau_: cython.float[:],
    n_reads_: cython.int[:],
    read_noise: cython.float,
    ramp: RampIndex,
) -> RampFit:
    """
    Fit a single ramp using Casertano+22 algorithm.

    Parameters
    ----------
    resultants_ : float[:]
        All of the resultants for the pixel
    t_bar_ : float[:]
        All the t_bar values
    tau_ : float[:]
        All the tau values
    n_reads_ : int[:]
        All the n_reads values
    read_noise : float
        The read noise for the pixel
    ramp : RampIndex
        Struct for start and end of ramp to fit

    Returns
    -------
    RampFit
        struct containing
        - slope
        - read_var
        - poisson_var
    """
    n_resultants: cython.int = ramp.end - ramp.start + 1

    # Special case where there is no or one resultant, there is no fit and
    # we bail out before any computations.
    #    Note that in this case, we cannot compute the slope or the variances
    #    because these computations require at least two resultants. Therefore,
    #    this case is degernate and we return NaNs for the values.
    if n_resultants <= 1:
        return RampFit(NAN, NAN, NAN)

    # Compute the fit
    i: cython.int = 0
    j: cython.int = 0

    # Setup data for fitting (work over subset of data) to make things cleaner
    #    Recall that the RampIndex contains the index of the first and last
    #    index of the ramp. Therefore, the Python slice needed to get all the
    #    data within the ramp is:
    #         ramp.start:ramp.end + 1
    resultants: cython.float[:] = resultants_[ramp.start : ramp.end + 1]
    t_bar: cython.float[:] = t_bar_[ramp.start : ramp.end + 1]
    tau: cython.float[:] = tau_[ramp.start : ramp.end + 1]
    n_reads: cython.int[:] = n_reads_[ramp.start : ramp.end + 1]

    # Compute mid point time
    end: cython.int = n_resultants - 1
    t_bar_mid: cython.float = (t_bar[0] + t_bar[end]) / 2

    # Casertano+2022 Eq. 44
    #    Note we've departed from Casertano+22 slightly;
    #    there s is just resultants[ramp.end].  But that doesn't seem good if, e.g.,
    #    a CR in the first resultant has boosted the whole ramp high but there
    #    is no actual signal.
    power: cython.float = fmaxf(resultants[end] - resultants[0], 0)
    power = power / sqrt(read_noise**2 + power)
    power = _get_power(power)

    # It's easy to use up a lot of dynamic range on something like
    # (tbar - tbarmid) ** 10.  Rescale these.
    t_scale: cython.float = (t_bar[end] - t_bar[0]) / 2
    t_scale = 1 if t_scale == 0 else t_scale

    # Initialize the fit loop
    #   it is faster to generate a c++ vector than a numpy array
    weights: vector[cython.float] = vector[float](n_resultants)
    coeffs: vector[cython.float] = vector[float](n_resultants)
    ramp_fit: RampFit = RampFit(0, 0, 0)
    f0: cython.float = 0
    f1: cython.float = 0
    f2: cython.float = 0
    coeff: cython.float

    # Issue when tbar[] == tbarmid causes exception otherwise
    with cython.cpow(True):
        for i in range(n_resultants):
            # Casertano+22, Eq. 45
            weights[i] = (((1 + power) * n_reads[i]) / (1 + power * n_reads[i])) * fabs(
                (t_bar[i] - t_bar_mid) / t_scale
            ) ** power

            # Casertano+22 Eq. 35
            f0 += weights[i]
            f1 += weights[i] * t_bar[i]
            f2 += weights[i] * t_bar[i] ** 2

    # Casertano+22 Eq. 36
    det: cython.float = f2 * f0 - f1**2
    if det == 0:
        return ramp_fit

    for i in range(n_resultants):
        # Casertano+22 Eq. 37
        coeff = (f0 * t_bar[i] - f1) * weights[i] / det
        coeffs[i] = coeff

        # Casertano+22 Eq. 38
        ramp_fit.slope += coeff * resultants[i]

        # Casertano+22 Eq. 39
        ramp_fit.read_var += coeff**2 * read_noise**2 / n_reads[i]

        # Casertano+22 Eq 40
        #    Note that this is an inversion of the indexing from the equation;
        #    however, commutivity of addition results in the same answer. This
        #    makes it so that we don't have to loop over all the resultants twice.
        ramp_fit.poisson_var += coeff**2 * tau[i]
        for j in range(i):
            ramp_fit.poisson_var += 2 * coeff * coeffs[j] * t_bar[j]

    return ramp_fit


# Casertano+2022, Table 2
_P_TABLE = cython.declare(
    cython.float[6][2],
    [
        [-INFINITY, 5, 10, 20, 50, 100],
        [0, 0.4, 1, 3, 6, 10],
    ],
)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.inline
@cython.cfunc
@cython.exceptval(check=False)
def _get_power(signal: cython.float) -> cython.float:
    """
    Return the power from Casertano+22, Table 2.

    Parameters
    ----------
    signal: float
        signal from the resultants

    Returns
    -------
    signal power from Table 2
    """
    i: cython.int
    for i in range(6):
        if signal < _P_TABLE[0][i]:
            return _P_TABLE[1][i - 1]

    return _P_TABLE[1][i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.inline
@cython.ccall
def _init_ramps(dq: cython.int[:], n_resultants: cython.int) -> RampQueue:
    """
    Create the initial ramp "queue" for each pixel
        if dq[index_resultant, index_pixel] == 0, then the resultant is in a ramp
        otherwise, the resultant is not in a ramp.

    Parameters
    ----------
    dq : int[n_resultants]
        DQ array
    n_resultants : int
        Number of resultants

    Returns
    -------
    RampQueue
        vector of RampIndex objects
            - vector with entry for each ramp found (last entry is last ramp found)
            - RampIndex with start and end indices of the ramp in the resultants
    """
    ramps: RampQueue = RampQueue()

    # Note: if start/end are -1, then no value has been assigned
    # ramp.start == -1 means we have not started a ramp
    # dq[index_resultant, index_pixel] == 0 means resultant is in ramp
    ramp: RampIndex = RampIndex(-1, -1)
    index_resultant: cython.int
    for index_resultant in range(n_resultants):
        # Checking for start of ramp
        if ramp.start == -1:
            if dq[index_resultant] == 0:
                # This resultant is in the ramp
                # => We have found the start of a ramp!
                ramp.start = index_resultant

        # This resultant cannot be the start of a ramp
        # => Checking for end of ramp
        elif dq[index_resultant] != 0:
            # This pixel is not in the ramp
            # => index_resultant - 1 is the end of the ramp
            ramp.end = index_resultant - 1

            # Add completed ramp to the queue and reset ramp
            ramps.push_back(ramp)
            ramp = RampIndex(-1, -1)

    # Handle case where last resultant is in ramp (so no end has been set)
    if ramp.start != -1 and ramp.end == -1:
        # Last resultant is end of the ramp => set then add to stack
        ramp.end = n_resultants - 1
        ramps.push_back(ramp)

    return ramps


@cython.inline
@cython.cfunc
@cython.exceptval(check=False)
def _threshold(thresh: Thresh, slope: cython.float) -> cython.float:
    """
    Compute jump threshold.

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


Stat = cython.struct(arg_max=cython.int, max_stat=cython.float)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.inline
@cython.cfunc
def _fit_statistic(
    single_pixel: cython.float[:, :],
    double_pixel: cython.float[:, :],
    single_fixed: cython.float[:, :],
    double_fixed: cython.float[:, :],
    t_bar: cython.float[:],
    slope: cython.float,
    ramp: RampIndex,
) -> Stat:
    """
    Compute the maximum index and its value over all fit statistics for a given
        ramp. Each index's stat is the max of the single and double difference
        statistics:
            all_stats = <max(single_stats), max(double_stats)>.

    Parameters
    ----------
    pixel : float[:, :]
        The pre-computed fixed values for a given pixel
    fixed : float[:, :]
        The pre-computed fixed values for a given read_pattern
    t_bar : float[:, :]
        The average time for each resultant
    slope : float
        The computed slope for the ramp
    ramp : RampIndex
        Struct for start and end of ramp to fit

    Returns
    -------
        argmax(all_stats), max(all_stats)
    """
    # Note that a ramp consisting of a single point is degenerate and has no
    #   fit statistic so we bail out here
    if ramp.start == ramp.end:
        return Stat(0, NAN)

    # Start computing fit statistics
    correct: cython.float = _correction(t_bar, ramp, slope)

    # We are computing single and double differences of using the ramp's resultants.
    #    Each of these computations requires two points meaning that there are
    #    start - end - 1 possible differences. However, we cannot compute a double
    #    difference for the last point as there is no point after it. Therefore,
    #    We use this point's single difference as our initial guess for the fit
    #    statistic. Note that the fit statistic can technically be negative so
    #    this makes it much easier to compute a "lazy" max.
    index: cython.int = ramp.end - 1
    stat: Stat = Stat(
        ramp.end - ramp.start - 1,
        _statistic(
            single_pixel[_local_slope, index],
            single_pixel[_var_read_noise, index],
            single_fixed[_t_bar_diff_sqr, index],
            single_fixed[_var_slope_val, index],
            slope,
            correct,
        ),
    )

    # Compute the rest of the fit statistics
    max_stat: cython.float
    single_stat: cython.float
    double_stat: cython.float
    arg_max: cython.int
    for arg_max, index in enumerate(range(ramp.start, ramp.end - 1)):
        # Compute max of single and double difference statistics
        single_stat = _statistic(
            single_pixel[_local_slope, index],
            single_pixel[_var_read_noise, index],
            single_fixed[_t_bar_diff_sqr, index],
            single_fixed[_var_slope_val, index],
            slope,
            correct,
        )
        double_stat = _statistic(
            double_pixel[_local_slope, index],
            double_pixel[_var_read_noise, index],
            double_fixed[_t_bar_diff_sqr, index],
            double_fixed[_var_slope_val, index],
            slope,
            correct,
        )
        max_stat = fmaxf(single_stat, double_stat)

        # If this is larger than the current max, update the max
        if max_stat > stat.max_stat:
            stat = Stat(arg_max, max_stat)

    return stat


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.inline
@cython.cfunc
@cython.exceptval(check=False)
def _statistic(
    local_slope: cython.float,
    var_read_noise: cython.float,
    t_bar_diff_sqr: cython.float,
    var_slope_val: cython.float,
    slope: cython.float,
    correct: cython.float,
) -> cython.float:
    """
    Compute a single fit statistic
        delta / sqrt(var + correct).

    where:
        delta = _local_slope - slope
        var = (var_read_noise + slope * var_slope_val) / t_bar_diff_sqr

        pre-computed:
            local_slope = (resultant[i + j]  - resultant[i]) / (t_bar[i + j] - t_bar[i])
            var_read_noise = read_noise ** 2 * (1/n_reads[i + j] + 1/n_reads[i])
            var_slope_coeff = tau[i + j] + tau[i] - 2 * min(t_bar[i + j], t_bar[i])
            t_bar_diff_sqr = (t_bar[i + j] - t_bar[i]) ** 2

    Parameters
    ----------
    local_slope : float
        The local slope the statistic is computed for
    var_read_noise: float
        The read noise variance for _local_slope
    t_bar_diff_sqr : float
        The square difference for the t_bar corresponding to _local_slope
    var_slope_val : float
        The slope variance coefficient for _local_slope
    slope : float
        The computed slope for the ramp
    correct : float
        The correction factor needed

    Returns
    -------
        Create a single instance of the stastic for the given parameters
    """
    delta: cython.float = local_slope - slope
    var: cython.float = (var_read_noise + slope * var_slope_val) / t_bar_diff_sqr

    return delta / sqrt(var + correct)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.inline
@cython.cfunc
@cython.exceptval(check=False)
def _correction(t_bar: cython.float[:], ramp: RampIndex, slope: cython.float) -> cython.float:
    """
    Compute the correction factor for the variance used by a statistic.

        - slope / (t_bar[end] - t_bar[start])

    Parameters
    ----------
    t_bar : float[:]
        The computed t_bar values for the ramp
    ramp : RampIndex
        Struct for start and end indices resultants for the ramp
    slope : float
        The computed slope for the ramp
    """
    diff: cython.float = t_bar[ramp.end] - t_bar[ramp.start]

    return -slope / diff


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.inline
@cython.ccall
def _fill_metadata(
    t_bar: cython.float[:],
    tau: cython.float[:],
    n_reads: cython.int[:],
    read_pattern: vector[vector[cython.int]],
    read_time: cython.float,
    n_resultants: cython.int,
) -> cython.void:
    n_read: cython.int

    i: cython.int
    j: cython.int
    resultant: vector[cython.int]
    for i in range(n_resultants):
        resultant = read_pattern[i]
        n_read = resultant.size()

        n_reads[i] = n_read
        t_bar[i] = 0
        tau[i] = 0

        for j in range(n_read):
            t_bar[i] += read_time * resultant[j]
            tau[i] += (2 * (n_read - j) - 1) * resultant[j]

        t_bar[i] /= n_read
        tau[i] *= read_time / n_read**2


_t_bar_diff = cython.declare(cython.int, FixedOffsets.t_bar_diff)
_t_bar_diff_sqr = cython.declare(cython.int, FixedOffsets.t_bar_diff_sqr)
_read_recip = cython.declare(cython.int, FixedOffsets.read_recip)
_var_slope_val = cython.declare(cython.int, FixedOffsets.var_slope_val)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.inline
@cython.ccall
def _fill_fixed_values(
    single_fixed: cython.float[:, :],
    double_fixed: cython.float[:, :],
    t_bar: cython.float[:],
    tau: cython.float[:],
    n_reads: cython.int[:],
    n_resultants: cython.int,
) -> cython.void:
    """
    Pre-compute all the values needed for jump detection which only depend on
        the read pattern.

    Parameters
    ----------
    fixed : float[:, :]
        A pre-allocated memoryview to store the pre-computed values in, its faster
        to allocate outside this function.
    t_bar : float[:]
        The average time for each resultant
    tau : float[:]
        The time variance for each resultant
    n_reads : int[:]
        The number of reads for each resultant
    n_resultants : int
        The number of resultants for the read pattern

    Returns
    -------
    [
        <t_bar[i+1] - t_bar[i]>,
        <t_bar[i+2] - t_bar[i]>,
        <t_bar[i+1] - t_bar[i]> ** 2,
        <t_bar[i+2] - t_bar[i]> ** 2,
        <(1/n_reads[i+1] + 1/n_reads[i])>,
        <(1/n_reads[i+2] + 1/n_reads[i])>,
        <(tau[i] + tau[i+1] - 2 * min(t_bar[i], t_bar[i+1]))>,
        <(tau[i] + tau[i+2] - 2 * min(t_bar[i], t_bar[i+2]))>,
    ]
    """
    # Coerce division to be using floats
    num: cython.float = 1

    i: cython.int
    for i in range(n_resultants - 1):
        single_fixed[_t_bar_diff, i] = t_bar[i + 1] - t_bar[i]
        single_fixed[_t_bar_diff_sqr, i] = single_fixed[_t_bar_diff, i] ** 2
        single_fixed[_read_recip, i] = (num / n_reads[i + 1]) + (num / n_reads[i])
        single_fixed[_var_slope_val, i] = tau[i + 1] + tau[i] - 2 * min(t_bar[i + 1], t_bar[i])

        if i < n_resultants - 2:
            double_fixed[_t_bar_diff, i] = t_bar[i + 2] - t_bar[i]
            double_fixed[_t_bar_diff_sqr, i] = double_fixed[_t_bar_diff, i] ** 2
            double_fixed[_read_recip, i] = (num / n_reads[i + 2]) + (num / n_reads[i])
            double_fixed[_var_slope_val, i] = tau[i + 2] + tau[i] - 2 * min(t_bar[i + 2], t_bar[i])


_local_slope = cython.declare(cython.int, PixelOffsets.local_slope)
_var_read_noise = cython.declare(cython.int, PixelOffsets.var_read_noise)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.inline
@cython.ccall
def _fill_pixel_values(
    single_pixel: cython.float[:, :],
    double_pixel: cython.float[:, :],
    single_fixed: cython.float[:, :],
    double_fixed: cython.float[:, :],
    resultants: cython.float[:],
    read_noise: cython.float,
    n_resultants: cython.int,
) -> cython.void:
    """
    Pre-compute all the values needed for jump detection which only depend on
        the a specific pixel (independent of the given ramp for a pixel).

    Parameters
    ----------
    pixel : float[:, :]
        A pre-allocated memoryview to store the pre-computed values in, its faster
        to allocate outside this function.
    resultants : float[:]
        The resultants for the pixel in question.
    fixed : float[:, :]
        The pre-computed fixed values for the read_pattern
    read_noise : float
        The read noise for the pixel
    n_resultants : int
        The number of resultants for the read_pattern

    Returns
    -------
    [
        <(resultants[i+1] - resultants[i])> / <(t_bar[i+1] - t_bar[i])>,
        <(resultants[i+2] - resultants[i])> / <(t_bar[i+2] - t_bar[i])>,
        read_noise**2 * <(1/n_reads[i+1] + 1/n_reads[i])>,
        read_noise**2 * <(1/n_reads[i+2] + 1/n_reads[i])>,
    ]
    """
    read_noise_sqr: cython.float = read_noise**2

    i: cython.int
    for i in range(n_resultants - 1):
        single_pixel[_local_slope, i] = (resultants[i + 1] - resultants[i]) / single_fixed[_t_bar_diff, i]
        single_pixel[_var_read_noise, i] = read_noise_sqr * single_fixed[_read_recip, i]

        if i < n_resultants - 2:
            double_pixel[_local_slope, i] = (resultants[i + 2] - resultants[i]) / double_fixed[_t_bar_diff, i]
            double_pixel[_var_read_noise, i] = read_noise_sqr * double_fixed[_read_recip, i]
