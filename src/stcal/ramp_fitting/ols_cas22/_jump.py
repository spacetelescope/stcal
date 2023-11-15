# cython: language_level=3str


"""
This module contains all the functions needed to execute jump detection for the
    Castentano+22 ramp fitting algorithm.

    The _ramp module contains the actual ramp fitting algorithm, this module
    contains a driver for the algorithm and detection of jumps/splitting ramps.

Structs
-------
Thresh : struct
    intercept - constant * log10(slope)
        - intercept : float
            The intercept of the jump threshold
        - constant : float
            The constant of the jump threshold

JumpFits : struct
    All the data on a given pixel's ramp fit with (or without) jump detection
        - average : RampFit
            The average of all the ramps fit for the pixel
        - jumps : vector[int]
            The indices of the resultants which were detected as jumps
        - fits : vector[RampFit]
            All of the fits for each ramp fit for the pixel
        - index : RampQueue
            The RampIndex representations corresponding to each fit in fits

Enums
-----
FixedOffsets : enum
    Enumerate the different pieces of information computed for jump detection
        which only depend on the read pattern.

PixelOffsets : enum
    Enumerate the different pieces of information computed for jump detection
        which only depend on the given pixel (independent of specific ramp).

JUMP_DET : value
    A the fixed value for the jump detection dq flag.

(Public) Functions
------------------
fill_fixed_values : function
    Pre-compute all the values needed for jump detection for a given read_pattern,
        this is independent of the pixel involved.

fit_jumps : function
    Compute all the ramps for a single pixel using the Casertano+22 algorithm
        with jump detection. This is a driver for the ramp fit algorithm in general
        meaning it automatically handles splitting ramps across dq flags in addition
        to splitting across detected jumps (if jump detection is turned on).
"""
import cython
from cython.cimports.libc.math import NAN, fmaxf, isnan, log10, sqrt
from cython.cimports.libcpp import bool as cpp_bool
from cython.cimports.libcpp.vector import vector
from cython.cimports.stcal.ramp_fitting.ols_cas22._jump import (
    JUMP_DET,
    FixedOffsets,
    JumpFits,
    PixelOffsets,
    Thresh,
)
from cython.cimports.stcal.ramp_fitting.ols_cas22._ramp import (
    RampFit,
    RampIndex,
    RampQueue,
    fit_ramp,
    init_ramps,
)

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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.inline
@cython.cfunc
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
@cython.cfunc
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
def fit_jumps(
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
    ramps: RampQueue = init_ramps(dq, n_resultants)

    # Initialize algorithm
    average: RampFit = RampFit(0, 0, 0)
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
        ramp_fit = fit_ramp(resultants, t_bar, tau, n_reads, read_noise, ramp)

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

            average.slope += weight * ramp_fit.slope
            average.read_var += weight**2 * ramp_fit.read_var
            average.poisson_var += weight**2 * ramp_fit.poisson_var

    # Finish computing averages using the lazy process
    average.slope /= total_weight if total_weight != 0 else 1
    average.read_var /= total_weight**2 if total_weight != 0 else 1
    average.poisson_var /= total_weight**2 if total_weight != 0 else 1

    # Multiply poisson term by flux, (no negative fluxes)
    average.poisson_var *= max(average.slope, 0)

    return JumpFits(average, jumps, fits, index)
