#! /usr/bin/env python

# !!!!!!!!!!!!!!!!!!! NOTE !!!!!!!!!!!!!!!!!!!
# Needs work.
# Also, this code makes reference to `nreads` as a the second dimension
# of the 4-D data set, while `ngroups` makes reference to the NGROUPS
# key word in the exposure metadata.  This should be changed, removing
# reference to the NGROUPS key word and using ngroups as the second
# dimension of the 4-D data set.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


import logging
from multiprocessing.pool import Pool as Pool
import numpy as np
import numpy.linalg as la
import time

from . import ramp_fit_class
from . import utils

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

DELIM = "-" * 80

BUFSIZE = 1024 * 300000  # 300Mb cache size for data section

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# This is the number of iterations that will be done with use_extra_terms
# set to False.  If this is zero, use_extra_terms will be set to True even
# for the first iteration.
# NUM_ITER_NO_EXTRA_TERMS = 1
NUM_ITER_NO_EXTRA_TERMS = 0

# These are the lower and upper limits of the number of iterations that
# will be done by determine_slope.
# MIN_ITER = NUM_ITER_NO_EXTRA_TERMS + 1
# MAX_ITER = 3
MIN_ITER = 1
MAX_ITER = 1

# This is a term to add for saturated pixels to give them low weight.
HUGE_FOR_LOW_WEIGHT = 1.e20

# This is a value to replace zero or negative values in a fit, to make
# all values of the fit positive and to give low weight where the fit was
# zero or negative.
FIT_MUST_BE_POSITIVE = 1.e10


def gls_ramp_fit(ramp_data, buffsize, save_opt, readnoise_2d, gain_2d, max_cores):
    """
    Fit a ramp using generalized least squares.

    Extended Summary
    ----------------
    Calculate the count rate for each pixel in the data ramp, for every
    integration.  Generalized least squares is used for fitting the ramp
    in order to take into account the correlation between reads.  If the
    input file contains multiple integrations, a second output file will
    be written, containing per-integration count rates.

    One additional file can optionally be written (if save_opt is True),
    containing per-integration data.

    Parameters
    ----------
    ramp_data: RampClass
        Input data needed for ramp fitting.

    buffsize : int
        Size of data section (buffer) in bytes.

    save_opt : boolean
        Calculate optional fitting results.

    readnoise_2d: ndarray
        Readnoise for all pixels.

    gain_2d: ndarray
        Gain for all pixels.

    max_cores : string
        Number of cores to use for multiprocessing. If set to 'none' (the default),
        then no multiprocessing will be done. The other allowable values are 'quarter',
        'half', and 'all'. This is the fraction of cores to use for multi-proc. The
        total number of cores includes the SMT cores (Hyper Threading for Intel).

    Returns
    -------
    image_info: tuple
        Tuple of image ndarrays computed for GLS ramp fitting.

    integ_info: tuple
        Tuple of integration ndarrays computed for GLS ramp fitting.

    gls_opt_info: tuple
        Tuple of optional product ndarrays computed for GLS ramp fitting.
    """
    number_slices = utils.compute_slices(max_cores)

    log.info(f"Number of data slices: {number_slices}")

    # Get needed sizes and shapes
    (nreads, npix, imshape, cubeshape, n_int, instrume, frame_time,
        ngroups, group_time) = utils.get_dataset_info(ramp_data)

    (group_time, frames_per_group, saturated_flag, jump_flag) = utils.get_more_info(
        ramp_data, ramp_data.flags_saturated, ramp_data.flags_jump_det)

    tstart = time.time()

    # Determine the maximum number of cosmic ray hits for any pixel.
    max_num_cr = -1                     # invalid initial value
    for num_int in range(n_int):
        i_max_num_cr = utils.get_max_num_cr(
            ramp_data.groupdq[num_int, :, :, :], jump_flag)
        max_num_cr = max(max_num_cr, i_max_num_cr)

    # Calculate effective integration time (once EFFINTIM has been populated
    #   and accessible, will use that instead), and other keywords that will
    #   needed if the pedestal calculation is requested. Note 'nframes'
    #   is the number of given by the NFRAMES keyword, and is the number of
    #   frames averaged on-board for a group, i.e., it does not include the
    #   groupgap.
    effintim, nframes, groupgap, dropframes1 = utils.get_efftim_ped(ramp_data)

    if number_slices == 1:
        image_info, integ_info, gls_opt_info = gls_fit_single(
            ramp_data, gain_2d, readnoise_2d, max_num_cr, save_opt)

    else:
        image_info, integ_info, gls_opt_info = gls_fit_multi(
            ramp_data, gain_2d, readnoise_2d, max_num_cr, save_opt, number_slices)

    tstop = time.time()

    log.info('Number of groups per integration: %d' % nreads)
    log.info('Number of integrations: %d' % n_int)

    log.debug(f"The execution time in seconds: {tstop - tstart:,}")

    return image_info, integ_info, gls_opt_info


def gls_fit_multi(
        ramp_data, gain_2d, readnoise_2d, max_num_cr, save_opt, number_slices):
    """
    ramp_data: RampClass
        The data needed to do ramp fitting.

    gain_2d: ndarray
        The 2-D gain for each pixel.

    readnoise_2d: ndarray
        The 2-D readnoise for each pixel.

    max_num_cr: int
        The maximum number of cosmic rays.

    save_opt: bool
        Option to create the optional results product.

    number_slices: int
        The number of slices/cores to use for multiprocessing.
    """
    log.info(f"Number of processors used for multiprocessing: {number_slices}")
    slices, rows_per_slice = compute_slices_for_starmap(
        ramp_data, save_opt, readnoise_2d, gain_2d, max_num_cr, number_slices)

    pool = Pool(processes=number_slices)
    pool_results = pool.starmap(gls_fit_single, slices)
    pool.close()
    pool.join()

    # Reassemble results
    image_info, integ_info, opt_res = assemble_pool_results(
        ramp_data, save_opt, pool_results, rows_per_slice)

    return image_info, integ_info, opt_res


def assemble_pool_results(ramp_data, save_opt, pool_results, rows_per_slice):
    """
    ramp_data: RampClass
        The original data used to do ramp fitting.

    save_opt: bool
        The option to save the optional results.

    pool_results: tuple
        The tuple of results to be reassembled.

    rows_per_slice: tuple
        The rows in each slice.
    """
    image_info, integ_info, opt_res = create_outputs(ramp_data)
    current_row = 0
    for k, results in enumerate(pool_results):
        nrows = rows_per_slice[k]

        image_slice, integ_slice, opt_slice = results

        reassemble_image(ramp_data, image_info, image_slice, current_row, nrows)
        reassemble_integ(ramp_data, integ_info, integ_slice, current_row, nrows)
        if save_opt:
            reassemble_opt(ramp_data, opt_res, opt_slice, current_row, nrows)

        current_row = current_row + nrows

    return image_info, integ_info, opt_res


def create_outputs(ramp_data):
    """
    Create the output arrays needed for multiprocessing reassembly.
    """
    image_info = create_output_image(ramp_data)
    integ_info = create_output_integ(ramp_data)
    opt_res = create_output_opt_res(ramp_data)

    return image_info, integ_info, opt_res


def create_output_image(ramp_data):
    """
    ramp_data: RampClass
        The original data used to do ramp fitting.
    """
    nints, ngroups, nrows, ncols = ramp_data.data.shape
    image_shape = (nrows, ncols)

    slope = np.zeros(image_shape, dtype=np.float32)
    pdq = np.zeros(image_shape, dtype=np.uint32)
    err = np.zeros(image_shape, dtype=np.float32)

    return slope, pdq, err


def create_output_integ(ramp_data):
    """
    ramp_data: RampClass
        The original data used to do ramp fitting.
    """
    nints, ngroups, nrows, ncols = ramp_data.data.shape
    image_shape = (nints, nrows, ncols)

    slope = np.zeros(image_shape, dtype=np.float32)
    pdq = np.zeros(image_shape, dtype=np.uint32)
    err = np.zeros(image_shape, dtype=np.float32)

    return slope, pdq, err


def create_output_opt_res(ramp_data):
    """
    ramp_data: RampClass
        The original data used to do ramp fitting.
    """
    # TODO Need to create the optional results output arrays.
    return None


def reassemble_image(ramp_data, image_info, image_slice, crow, nrows):
    """
    ramp_data: RampClass
        The original data used to do ramp fitting.

    image_info: tuple
        Tuple of final output arrays to be reassembled.

    image_slice: tuple
        Tuple of sliced output arrays to be used during reassembly.

    crow: int
        The current start row.

    nrows: int
        The number of rows in the current slice.
    """
    slope, pdq, err = image_slice
    srow, erow = crow, crow + nrows

    image_info[0][srow:erow, :] = slope
    image_info[1][srow:erow, :] = pdq
    image_info[2][srow:erow, :] = err


def reassemble_integ(ramp_data, integ_info, integ_slice, crow, nrows):
    """
    ramp_data: RampClass
        The original data used to do ramp fitting.

    integ_info: tuple
        Tuple of final output arrays to be reassembled.

    integ_slice: tuple
        Tuple of sliced output arrays to be used during reassembly.

    crow: int
        The current start row.

    nrows: int
        The number of rows in the current slice.
    """
    # integ_info = (slope_int, dq_int, slope_err_int)
    slope, dq, err = integ_slice
    srow, erow = crow, crow + nrows

    integ_info[0][:, srow:erow, :] = slope
    integ_info[1][:, srow:erow, :] = dq
    integ_info[2][:, srow:erow, :] = err


def reassemble_opt(ramp_data, opt_res, opt_slice, crow, nrows):
    """
    ramp_data: RampClass
        The original data used to do ramp fitting.

    opt_res: tuple
        Tuple of final output arrays to be reassembled.

    opt_slice: tuple
        Tuple of sliced output arrays to be used during reassembly.

    crow: int
        The current start row.

    nrows: int
        The number of rows in the current slice.
    """
    # TODO finish function
    # gls_opt_info = (intercept_int, intercept_err_int, pedestal_int, ampl_int, ampl_err_int)
    inter, err, pedestal, ampl, ampl_err = opt_slice
    # srow, erow = crow, crow + nrows

    log.debug(f"    ---> ({crow}, {crow + nrows})")
    log.debug(f"inter    = {inter.shape}")
    log.debug(f"err      = {err.shape}")
    log.debug(f"pedestal = {pedestal.shape}")
    log.debug(f"ampl     = {ampl.shape}")
    log.debug(f"ampl_err = {ampl_err.shape}")

    # TODO Dimension check
    '''
    opt_res[0][:, srow:erow, :] = slope
    opt_res[1][:, srow:erow, :] = err
    opt_res[2][:, srow:erow, :] = pedestal
    opt_res[3][:, srow:erow, :] = ampl
    opt_res[4][:, srow:erow, :] = ampl_err
    '''


def compute_slices_for_starmap(
        ramp_data, save_opt, readnoise_2d, gain_2d, max_num_cr, number_slices):
    """
    Creates the slices needed for each process for multiprocessing.  The slices
    for the arguments needed for ols_ramp_fit_single.

    ramp_data: RampData
        The ramp data to be sliced.

    save_opt : bool
        Whether to return the optional output model

    readnoise_2d : ndarray
        The read noise of each pixel

    gain_2d : ndarray
        The gain of each pixel

    max_num_cr: int
        The maximum number of cosmic rays in a ramp.

    number_slices: int
        The number of slices to partition the data into for multiprocessing.

    Return
    ------
    slices: list
        The list of arguments for each processor for multiprocessing.
    """
    nrows = ramp_data.data.shape[2]
    rslices = rows_per_slice(number_slices, nrows)
    slices = []
    start_row = 0
    for k in range(len(rslices)):
        ramp_slice = slice_ramp_data(ramp_data, start_row, rslices[k])
        rnoise_slice = readnoise_2d[start_row:start_row + rslices[k], :].copy()
        gain_slice = gain_2d[start_row:start_row + rslices[k], :].copy()
        slices.insert(
            k,
            (ramp_slice, rnoise_slice, gain_slice, max_num_cr, save_opt))
        start_row = start_row + rslices[k]

    return slices, rslices


def rows_per_slice(nslices, nrows):
    """
    Compute the number of rows per slice.

    Parameters
    ----------
    nslices: int
        The number of slices to partition the rows.

    nrows: int
        The number of rows to partition.

    Return
    ______
    rslices: list
        The number of rows for each slice.
    """
    quotient = nrows // nslices
    remainder = nrows % nslices

    no_inc = nslices - remainder
    if remainder > 0:
        # Ensure the number of rows per slice is no more than a
        # difference of one.
        first = [quotient + 1] * remainder
        second = [quotient] * no_inc
        rslices = first + second
    else:
        rslices = [quotient] * nslices

    return rslices


def slice_ramp_data(ramp_data, start_row, nrows):
    """
    Slices the ramp data by rows, where the arrays contain all rows in
    [start_row, start_row+nrows).

    Parameters
    ----------
    ramp_data: RampData
        The ramp data to slice.

    start_rows: int
        The start row of the slice.

    nrows: int
        The number of rows in the slice.

    Return
    ------
    ramp_data_slice: RampData
        The slice of the ramp_data.
    """
    ramp_data_slice = ramp_fit_class.RampData()

    # Slice data by row
    data = ramp_data.data[:, :, start_row:start_row + nrows, :].copy()
    err = ramp_data.err[:, :, start_row:start_row + nrows, :].copy()
    groupdq = ramp_data.groupdq[:, :, start_row:start_row + nrows, :].copy()
    pixeldq = ramp_data.pixeldq[start_row:start_row + nrows, :].copy()

    ramp_data_slice.set_arrays(data, err, groupdq, pixeldq)

    # Carry over meta data.
    ramp_data_slice.set_meta(
        name=ramp_data.instrument_name,
        frame_time=ramp_data.frame_time,
        group_time=ramp_data.group_time,
        groupgap=ramp_data.groupgap,
        nframes=ramp_data.nframes,
        drop_frames1=ramp_data.drop_frames1)

    # Carry over DQ flags.
    ramp_data_slice.flags_do_not_use = ramp_data.flags_do_not_use
    ramp_data_slice.flags_jump_det = ramp_data.flags_jump_det
    ramp_data_slice.flags_saturated = ramp_data.flags_saturated
    ramp_data_slice.flags_no_gain_val = ramp_data.flags_no_gain_val
    ramp_data_slice.flags_unreliable_slope = ramp_data.flags_unreliable_slope

    # Slice info
    ramp_data_slice.start_row = start_row
    ramp_data_slice.num_rows = nrows

    return ramp_data_slice


def gls_fit_single(ramp_data, gain_2d, readnoise_2d, max_num_cr, save_opt):
    """
    Single processor ramp fit.
    This method will fit the rate for all pixels and all integrations using
    the Generalized Least Squares (GLS) method.

    Parameters
    ----------
    ramp_data: RampClass
        Information needed to do ramp fitting.

    gain_2d: ndarray
        Gain noise 2-D array.

    readnoise_2d: ndarray
        Read noise 2-D array.

    max_num_cr: int
        The maximum number of cosmic rays in the data.

    save_opt: bool
        Save optional product.

    Returns
    --------
    image_info: tuple
        Tuple of ndarrays computed for the primary product for ramp fitting.

    integ_info: tuple
        Tuple of the ndarrays computed for each integration from ramp fitting.

    gls_opt_info: tuple
        Tuple of the ndarrays computed for the optional results product.
    """
    # START

    frame_time = ramp_data.frame_time
    group_time = ramp_data.group_time
    nframes_used = ramp_data.nframes

    gdq = ramp_data.groupdq
    data = ramp_data.data
    pixeldq = ramp_data.pixeldq

    jump_flag = ramp_data.flags_jump_det
    saturated_flag = ramp_data.flags_saturated

    number_ints = data.shape[0]
    ngroups = data.shape[1]

    slope_int, slope_err_int, dq_int, temp_dq, slopes, sum_weight = \
        create_integration_arrays(data.shape)

    # REFAC
    (intercept_int, intercept_err_int, pedestal_int, first_group, shape_ampl,
        ampl_int, ampl_err_int) = create_opt_res(save_opt, data.shape, max_num_cr)

    pixeldq = utils.reset_bad_gain(ramp_data, pixeldq, gain_2d)  # Flag bad pixels in gain

    med_rates = None
    if ngroups == 1:
        med_rates = utils.compute_median_rates(ramp_data)

    # We'll propagate error estimates from previous steps to the
    # current step by using the variance.
    input_var = ramp_data.err**2

    # Convert the data section from DN to electrons.
    data *= gain_2d

    for num_int in range(number_ints):
        ramp_data.current_integ = num_int
        gdq_cube = gdq[num_int, :, :, :]
        data_cube = data[num_int, :, :, :]
        input_var_sect = input_var[num_int, :, :, :]

        if save_opt:
            first_group[:, :] = data[num_int, 0, :, :].copy()

        (intercept_sect, intercept_var_sect, slope_sect,
         slope_var_sect, cr_sect, cr_var_sect) = determine_slope(
             ramp_data,
             data_cube, input_var_sect, gdq_cube,
             readnoise_2d, gain_2d, frame_time, group_time,
             nframes_used, max_num_cr, saturated_flag, jump_flag, med_rates)

        slope_int[num_int, :, :] = slope_sect.copy()
        v_mask = (slope_var_sect <= 0.)
        if v_mask.any():
            # Replace negative or zero variances with a large value.
            slope_var_sect[v_mask] = utils.LARGE_VARIANCE

            # Also set a flag in the pixel dq array.
            temp_dq[v_mask] = ramp_data.flags_unreliable_slope
            del v_mask

        # If a pixel was flagged (by an earlier step) as saturated in
        # the first group, flag the pixel as bad.
        # Note:  save s_mask until after the call to utils.gls_pedestal.
        s_mask = (gdq_cube[0] == saturated_flag)
        if s_mask.any():
            temp_dq[s_mask] = ramp_data.flags_do_not_use
        slope_err_int[num_int, :, :] = np.sqrt(slope_var_sect)

        # We need to take a weighted average if (and only if) number_ints > 1.
        # Accumulate sum of slopes and sum of weights.
        if number_ints > 1:
            weight = 1. / slope_var_sect
            slopes[:, :] += (slope_sect * weight)
            sum_weight[:, :] += weight

        if save_opt:
            # Save the intercepts and cosmic-ray amplitudes for the
            # current integration.
            intercept_int[num_int, :, :] = intercept_sect.copy()
            intercept_err_int[num_int, :, :] = np.sqrt(np.abs(intercept_var_sect))
            pedestal_int[num_int, :, :] = utils.gls_pedestal(
                first_group[:, :], slope_int[num_int, :, :],
                s_mask, frame_time, nframes_used)
            del s_mask

            ampl_int[num_int, :, :, :] = cr_sect.copy()
            ampl_err_int[num_int, :, :, :] = np.sqrt(np.abs(cr_var_sect))

        # Compress 4D->2D dq arrays for saturated and jump-detected
        # pixels
        pixeldq_sect = pixeldq[:, :].copy()
        dq_int[num_int, :, :] = utils.dq_compress_sect(
            ramp_data, num_int, gdq_cube, pixeldq_sect).copy()

        dq_int[num_int, :, :] |= temp_dq
        temp_dq[:, :] = 0  # initialize for next integration

    # Average the slopes over all integrations.
    if number_ints > 1:
        sum_weight = np.where(sum_weight <= 0., 1., sum_weight)
        recip_sum_weight = 1. / sum_weight
        slopes *= recip_sum_weight
        gls_err = np.sqrt(recip_sum_weight)

    # Convert back from electrons to DN.
    slope_int /= gain_2d

    slope_err_int /= gain_2d
    if number_ints > 1:
        slopes /= gain_2d
        gls_err /= gain_2d
    if save_opt:
        intercept_int /= gain_2d
        intercept_err_int /= gain_2d
        pedestal_int /= gain_2d
        gain_shape = gain_2d.shape
        gain_4d = gain_2d.reshape((1, gain_shape[0], gain_shape[1], 1))
        ampl_int /= gain_4d
        ampl_err_int /= gain_4d
        del gain_4d

    # Compress all integration's dq arrays to create 2D PIXELDDQ array for
    #   primary output
    final_pixeldq = utils.dq_compress_final(dq_int, ramp_data)

    integ_info = (slope_int, dq_int, slope_err_int)

    if save_opt:  # collect optional results for output
        # Get the zero-point intercepts and the cosmic-ray amplitudes for
        # each integration (even if there's only one integration).
        gls_opt_info = (intercept_int, intercept_err_int,
                        pedestal_int, ampl_int, ampl_err_int)
    else:
        gls_opt_info = None

    # Get output image information
    if number_ints > 1:
        fslope, ferr = (slopes.astype(np.float32), gls_err.astype(np.float32))
    else:
        fslope, ferr = (slope_int[0], slope_err_int[0])

    wh_nan = np.isnan(fslope)
    fslope[wh_nan] = 0.0
    final_pixeldq[wh_nan] |= ramp_data.flags_do_not_use

    image_info = (fslope, final_pixeldq, ferr)

    return image_info, integ_info, gls_opt_info


def create_integration_arrays(dims):
    """
    Parameter
    ---------
    dims: tuple
        Dimensions of the 4-D array.
    """
    number_ints, ngroups, number_rows, number_cols = dims

    # Integration cubes
    slope_int = np.zeros((number_ints, number_rows, number_cols), dtype=np.float32)
    slope_err_int = np.zeros((number_ints, number_rows, number_cols), dtype=np.float32)
    dq_int = np.zeros((number_ints, number_rows, number_cols), dtype=np.uint32)

    # Image size
    temp_dq = np.zeros((number_rows, number_cols), dtype=np.uint32)
    slopes = np.zeros((number_rows, number_cols), dtype=np.float32)
    sum_weight = np.zeros((number_rows, number_cols), dtype=np.float32)

    return slope_int, slope_err_int, dq_int, temp_dq, slopes, sum_weight


# REFAC
def create_opt_res(save_opt, dims, max_num_cr):
    """
    Parameter
    ---------
    dims: tuple
        Dimensions of the 4-D array.
    """
    number_ints, number_groups, number_rows, number_cols = dims
    imshape = (number_rows, number_cols)

    if save_opt:
        # Create arrays for the fitted values of zero-point intercept and
        # cosmic-ray amplitudes, and their errors.
        intercept_int = np.zeros((number_ints,) + imshape, dtype=np.float32)
        intercept_err_int = np.zeros((number_ints,) + imshape, dtype=np.float32)

        # The pedestal is the extrapolation of the first group back to zero
        # time, for each integration.
        pedestal_int = np.zeros((number_ints,) + imshape, dtype=np.float32)

        # The first group, for calculating the pedestal.  (This only needs
        # to be nrows high, but we don't have nrows yet.  xxx)
        first_group = np.zeros(imshape, dtype=np.float32)

        # If there are no cosmic rays, set the last axis length to 1.
        shape_ampl = (number_ints, imshape[0], imshape[1], max(1, max_num_cr))
        ampl_int = np.zeros(shape_ampl, dtype=np.float32)
        ampl_err_int = np.zeros(shape_ampl, dtype=np.float32)
    else:
        intercept_int = None
        intercept_err_int = None
        pedestal_int = None
        first_group = None
        shape_ampl = None
        ampl_int = None
        ampl_err_int = None

    return (intercept_int, intercept_err_int, pedestal_int, first_group,
            shape_ampl, ampl_int, ampl_err_int)


def determine_slope(
        ramp_data, data_sect, input_var_sect, gdq_sect, readnoise_sect, gain_sect,
        frame_time, group_time, nframes_used, max_num_cr, saturated_flag,
        jump_flag, med_rates):
    """Iteratively fit a slope, intercept, and cosmic rays to a ramp.

    This function fits a ramp, possibly with discontinuities (cosmic-ray
    hits), to a 3-D data "cube" with shape (number of groups, number of
    pixels high, number of pixels wide).  The fit will be done multiple
    times, with the previous fit being used to assign weights (via the
    covariance matrix) for the current fit.  The iterations stop either
    when the maximum number of iterations has been reached or when the
    maximum difference between the previous fit and the current fit is
    below a cutoff.  This function calls compute_slope and evaluate_fit.

    compute_slope creates arrays for the slope, intercept, and cosmic-ray
    amplitudes (the arrays that will be returned by determine_slope).  Then
    it loops over the number of cosmic rays, from 0 to max_num_cr
    inclusive.  Within this loop, compute_slope copies to temporary arrays
    the ramp data for all the pixels that have the current number of cosmic
    ray hits, calls gls_fit to compute the fit, then copies the results
    of the fit (slope, etc.) to the output arrays for just those pixels.

    The input to gls_fit is ramp data for a subset of pixels (nz in number)
    that all have the same number of cosmic-ray hits.  gls_fit solves
    matrix equations (one for each of the nz pixels) of the form:

        y = x * p

    where y is a column vector containing the observed data values in
    electrons for each group (the length of y is ngroups, the number of
    groups); x is a matrix with ngroups rows and 2 + num_cr columns, where
    num_cr is the number of cosmic rays being included in the fit; and p
    is the solution, a column vector containing the intercept, slope, and
    the amplitude of each of the num_cr cosmic rays.  The first column of
    x is all ones, for fitting to the intercept.  The second column of x
    is the time (seconds) at the beginning of each group.  The remaining
    num_cr columns (if num_cr > 0) are Heaviside functions, 0 for the
    first rows and 1 for all rows at and following the group containing a
    cosmic-ray hit (each row corresponds to a group).  There will be one
    such column for each cosmic ray, so that the cosmic rays will be fit
    independently of each other.  Whether a cosmic ray hit the detector
    during a particular group was determined by a previous step, and the
    affected groups are flagged in the group data quality array.  In order
    to account for the variance of each observed value and the covariance
    between them (since they're measurements along a ramp), the solution
    is computed in this form (the @ sign represents matrix multiplication):

        (xT @ C^-1 @ x)^-1 @ [xT @ C^-1 @ y]

    where C is the ngroups by ngroups covariance matrix, ^-1 means matrix
    inverse, and xT is the transpose of x.

    Summary of the notation:

    data_sect is 3-D, (ngroups, ny, nx); this is the ramp of science data.
    cr_flagged is 3-D, (ngroups, ny, nx); 1 indicates a cosmic ray, e.g.:
        cr_flagged = np.where(np.bitwise_and(gdq_sect, jump_flag), 1, 0)
    cr_flagged_2d is 2-D, (ngroups, nz); this gives the location within
        the ramp of each cosmic ray, for the subset of pixels (nz of them)
        that have a total of num_cr cosmic ray hits at each pixel.  This
        is passed to gls_fit(), which fits a slope to the ramp.

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    XXX TODO - not sure what 'ramp_data' here refers to.  This is NOT the
               RampData class passed into this function.  This is an old
               comment held over from before refactoring.
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ramp_data has shape (ngroups, nz); this will be a ramp with a 1-D
    array of pixels copied out of data_sect.  The pixels will be those
    that have a particular number of cosmic-ray hits, somewhere within
    the ramp.

    Sum cr_flagged over groups to get an (ny, nx) image of the number of
    cosmic rays (i.e. accumulated over the ramp) in each pixel.
    sum_flagged = cr_flagged.sum(axis=0, ...)
    sum_flagged is used to extract the nz pixels from (ny, nx) that have a
    specified number of cosmic ray hits, e.g.:
        for num_cr in range(max_num_cr + 1):
            ncr_mask = (sum_flagged == num_cr)
            nz = ncr_mask.sum(dtype=np.int32)
            for k in range(ngroups):
                ramp_data[k] = data_sect[k][ncr_mask]
                cr_flagged_2d[k] = cr_flagged[k][ncr_mask]

    gls_fit is called for the subset of pixels (nz of them) that have
    num_cr cosmic ray hits within the ramp, the same number for every
    pixel.

    Parameters
    ----------
    ramp_data : RampData
        The class containing all metadata needed for slope computations.

    data_sect : 3-D ndarray, shape (ngroups, ny, nx)
        The ramp data for one integration.  This may be a subarray in
        detector coordinates, but covering all groups.  ngroups is the
        number of groups; ny is the number of pixels in the Y direction;
        nx is the number of pixels in the X (more rapidly varying)
        direction.  The units should be electrons.

    input_var_sect : 3-D ndarray, shape (ngroups, ny, nx)
        The square of the input ERR array, matching data_sect.

    gdq_sect : 3-D ndarray, shape (ngroups, ny, nx)
        The group data quality array.  This may be a subarray, matching
        data_sect.

    readnoise_sect : 2-D ndarray, shape (ny, nx)
        The read noise in electrons at each detector pixel (i.e. not a
        ramp).  This may be a subarray, similar to data_sect.

    gain_sect : 2-D ndarray, or None, shape (ny, nx)
        The gain in electrons per DN at each detector pixel (i.e. not a
        ramp).  This may be a subarray, matching readnoise_sect.  If
        gain_sect is None, a value of 1 will be assumed.

    frame_time : float
        The time to read one frame, in seconds (e.g. 10.6 s).

    group_time : float
        Time increment between groups, in seconds.

    nframes_used : int
        Number of frames that were averaged together to make a group.
        Note that this value does not include the number (if any) of
        skipped frames.

    max_num_cr : non-negative int
        The maximum number of cosmic rays that should be handled.  This
        must be specified by the caller, because determine_slope may be
        called for different sections of the input data, and those sections
        may have differing maximum numbers of cosmic rays.

    saturated_flag : int
        The saturation flag.

    jump_flag : int
        The jump detection flag.

    med_rates : ndarray
        A 2D array for the median rate for each pixel.

    Returns
    -------
    tuple :  (intercept_sect, int_var_sect, slope_sect, slope_var_sect,
             cr_sect, cr_var_sect)
        intercept_sect : 2-D ndarray, shape (ny, nx)
            The intercept of the ramp at each pixel.
        int_var_sect : 2-D ndarray, shape (ny, nx)
            The variance of the intercept at each pixel.
        slope_sect : 2-D ndarray, shape (ny, nx)
            The ramp slope at each pixel of data_sect.
        slope_var_sect : 2-D ndarray, shape (ny, nx)
            The variance of the slope at each pixel.
        cr_sect : 3-D ndarray, shape (ny, nx, cr_dimen)
            The amplitude of each cosmic ray at each pixel.  cr_dimen is
            max_num_cr or 1, whichever is larger.
        cr_var_sect : 3-D ndarray, shape (ny, nx, cr_dimen)
            The variance of each cosmic-ray amplitude.
    """
    ngroups, nrows, ncols = data_sect.shape
    if ngroups == 1:
        return determine_slope_one_group(
            ramp_data, data_sect, input_var_sect, gdq_sect, readnoise_sect,
            gain_sect, frame_time, group_time, nframes_used, max_num_cr,
            saturated_flag, jump_flag, med_rates)

    slope_diff_cutoff = 1.e-5

    # These will be updated in the loop.
    # TODO The next line assumes more than one group
    prev_slope_sect = (data_sect[1, :, :] - data_sect[0, :, :]) / group_time
    prev_fit = data_sect.copy()

    use_extra_terms = True

    iter = 0
    done = False
    if NUM_ITER_NO_EXTRA_TERMS <= 0:
        # Even the first iteration uses the extra terms.
        temp_use_extra_terms = True
    else:
        temp_use_extra_terms = False

    while not done:
        (intercept_sect, int_var_sect, slope_sect,
         slope_var_sect, cr_sect, cr_var_sect) = compute_slope(
             data_sect, input_var_sect, gdq_sect, readnoise_sect, gain_sect,
             prev_fit, prev_slope_sect, frame_time, group_time, nframes_used,
             max_num_cr, saturated_flag, jump_flag, temp_use_extra_terms)

        iter += 1
        if iter == NUM_ITER_NO_EXTRA_TERMS:
            temp_use_extra_terms = use_extra_terms

        if iter >= MAX_ITER:
            done = True
        else:
            # If there are pixels with zero or negative variance, ignore
            # them when taking the difference between the slopes computed
            # in the current and previous iterations.
            slope_diff = np.where(
                slope_var_sect > 0., prev_slope_sect - slope_sect, 0.)

            max_slope_diff = np.abs(slope_diff).max()
            if iter >= MIN_ITER and max_slope_diff < slope_diff_cutoff:
                done = True

            current_fit = evaluate_fit(
                intercept_sect, slope_sect, cr_sect, frame_time,
                group_time, gdq_sect, jump_flag)

            prev_fit = positive_fit(current_fit)      # use for next iteration
            del current_fit
            prev_slope_sect = slope_sect.copy()

    return (intercept_sect, int_var_sect, slope_sect,
            slope_var_sect, cr_sect, cr_var_sect)


def determine_slope_one_group(
        ramp_data, data_sect, input_var_sect, gdq_sect, readnoise_sect, gain_sect,
        frame_time, group_time, nframes_used, max_num_cr, saturated_flag,
        jump_flag, med_rates):
    """
    The special case where an integration has only one group.

    Parameters
    ----------
    ramp_data : RampData
        The ramp data class containing all metadata needed for computations.

    data_sect : 3-D ndarray, shape (ngroups, ny, nx)
        The ramp data for one integration.  This may be a subarray in
        detector coordinates, but covering all groups.  ngroups is the
        number of groups; ny is the number of pixels in the Y direction;
        nx is the number of pixels in the X (more rapidly varying)
        direction.  The units should be electrons.

    input_var_sect : 3-D ndarray, shape (ngroups, ny, nx)
        The square of the input ERR array, matching data_sect.

    gdq_sect : 3-D ndarray, shape (ngroups, ny, nx)
        The group data quality array.  This may be a subarray, matching
        data_sect.

    readnoise_sect : 2-D ndarray, shape (ny, nx)
        The read noise in electrons at each detector pixel (i.e. not a
        ramp).  This may be a subarray, similar to data_sect.

    gain_sect : 2-D ndarray, or None, shape (ny, nx)
        The gain in electrons per DN at each detector pixel (i.e. not a
        ramp).  This may be a subarray, matching readnoise_sect.  If
        gain_sect is None, a value of 1 will be assumed.

    frame_time : float
        The time to read one frame, in seconds (e.g. 10.6 s).

    group_time : float
        Time increment between groups, in seconds.

    nframes_used : int
        Number of frames that were averaged together to make a group.
        Note that this value does not include the number (if any) of
        skipped frames.

    max_num_cr : non-negative int
        The maximum number of cosmic rays that should be handled.  This
        must be specified by the caller, because determine_slope may be
        called for different sections of the input data, and those sections
        may have differing maximum numbers of cosmic rays.

    saturated_flag : int
        The saturation flag.

    jump_flag : int
        The jump detection flag.

    med_rates : ndarray
        A 2D array for the median rate for each pixel.

    Returns
    -------
    tuple :  (intercept_sect, int_var_sect, slope_sect, slope_var_sect,
             cr_sect, cr_var_sect)

        intercept_sect : 2-D ndarray, shape (ny, nx)
            The intercept of the ramp at each pixel.

        int_var_sect : 2-D ndarray, shape (ny, nx)
            The variance of the intercept at each pixel.

        slope_sect : 2-D ndarray, shape (ny, nx)
            The ramp slope at each pixel of data_sect.

        slope_var_sect : 2-D ndarray, shape (ny, nx)
            The ramp slope at each pixel of data_sect.

        cr_sect : 3-D ndarray, shape (ny, nx, cr_dimen)
            The amplitude of each cosmic ray at each pixel.  cr_dimen is
            max_num_cr or 1, whichever is larger.

        cr_var_sect : 3-D ndarray, shape (ny, nx, cr_dimen)
            The variance of each cosmic-ray amplitude.
    """
    ngroups, nrows, ncols = data_sect.shape

    imshape = (nrows, ncols)
    intercept_sect = np.zeros(imshape, dtype=np.float32)
    int_var_sect = np.zeros(imshape, dtype=np.float32)

    slope_sect = data_sect[0, :, :] / group_time
    slope_var_sect = np.zeros(imshape, dtype=np.float32)
    var_r = 12. * (readnoise_sect / group_time)**2
    var_p = med_rates / (group_time * gain_sect)

    # Handle ZEROFRAME
    if ramp_data.zframe_locs:
        for pix in ramp_data.zframe_locs[ramp_data.current_integ]:
            row, col = pix
            slope_sect = data_sect[0, row, col] / frame_time
            var_r[row, col] = 12. * (readnoise_sect[row, col] / frame_time)**2.
            var_p[row, col] = med_rates[row, col] / (frame_time * gain_sect[row, col])
    slope_var_sect = var_r + var_p

    cubeshape = (1, nrows, ncols)
    cr_sect = np.zeros(cubeshape, dtype=np.float32)  # Not sure what this is
    cr_var_sect = np.zeros(cubeshape, dtype=np.float32)  # Not sure what this is

    return (intercept_sect, int_var_sect, slope_sect,
            slope_var_sect, cr_sect, cr_var_sect)


def evaluate_fit(
        intercept_sect, slope_sect, cr_sect, frame_time, group_time, gdq_sect, jump_flag):
    """Evaluate the fit (intercept, slope, cosmic-ray amplitudes).

    Parameters
    ----------
    intercept_sect : 2-D ndarray
        The intercept of the ramp at each pixel of data_sect (one of the
        arguments to determine_slope).

    slope_sect : 2-D ndarray
        The ramp slope at each pixel of data_sect.

    cr_sect : 3-D ndarray
        The amplitude of each cosmic ray at each pixel of data_sect.

    frame_time : float
        The time to read one frame, in seconds (e.g. 10.6 s).

    group_time : float
        Time increment between groups, in seconds.

    gdq_sect : 3-D ndarray; indices:  group, y, x
        The group data quality array.  This may be a subarray, matching
        data_sect.

    jump_flag : int
        The jump detection flag.

    Returns
    -------
    fit_model : 3-D ndarray, shape (ngroups, ny, nx)
        This is the same shape as data_sect, and if the fit is good,
        fit_model and data_sect should not differ by much.
    """

    shape_3d = gdq_sect.shape           # the ramp, (ngroups, ny, nx)
    ngroups = gdq_sect.shape[0]

    # This array is also created in function compute_slope.
    cr_flagged = np.empty(shape_3d, dtype=np.uint8)
    cr_flagged[:] = np.where(np.bitwise_and(gdq_sect, jump_flag), 1, 0)

    sum_flagged = cr_flagged.sum(axis=0, dtype=np.int32)

    # local_max_num_cr is local to this function.  It may be smaller than
    # the max_num_cr that's an argument to determine_slope, and it can even
    # be zero.
    local_max_num_cr = sum_flagged.max()
    del sum_flagged

    # The independent variable, in seconds at each image pixel.
    ind_var = np.zeros(shape_3d, dtype=np.float64)
    M = round(group_time / frame_time)
    iv = np.arange(ngroups, dtype=np.float64) * group_time + \
        frame_time * (M + 1.) / 2.
    iv = iv.reshape((ngroups, 1, 1))
    ind_var += iv

    # No cosmic rays yet; these will be accounted for below.
    # ind_var has a different shape (ngroups, ny, nx) from slope_sect and
    # intercept_sect, but their last dimensions are the same.
    fit_model = ind_var * slope_sect + intercept_sect

    # heaviside and cr_flagged have shape (ngroups, ny, nx).
    heaviside = np.zeros(shape_3d, dtype=np.float64)
    cr_cumsum = cr_flagged.cumsum(axis=0, dtype=np.int16)

    # Add an offset for each cosmic ray.
    for n in range(local_max_num_cr):
        heaviside[:] = np.where(cr_cumsum > n, 1., 0.)
        fit_model += (heaviside * cr_sect[:, :, n])

    return fit_model


def positive_fit(current_fit):
    """Replace zero and negative values with a positive number.

    Ramp data should be positive, since they are based on counts.  The
    fit to a ramp can go negative, however, due e.g. to extrapolation
    beyond where the data are saturated.  To avoid negative elements in
    the covariance matrix (populated in part with the fit to the ramp),
    this function replaces zero or negative values in the fit with a
    positive number.

    Parameters
    ----------
    current_fit : 3-D ndarray, shape (ngroups, ny, nx)
        The fit returned by evaluate_fit.

    Returns
    -------
    current_fit : 3-D ndarray, shape (ngroups, ny, nx)
        This is the same as the input current_fit, except that zero and
        negative values will have been replaced by a positive value.
    """

    return np.where(current_fit <= 0., FIT_MUST_BE_POSITIVE, current_fit)


def compute_slope(
        data_sect, input_var_sect, gdq_sect, readnoise_sect, gain_sect,
        prev_fit, prev_slope_sect, frame_time, group_time, nframes_used,
        max_num_cr, saturated_flag, jump_flag, use_extra_terms):
    """Set up the call to fit a slope to ramp data.

    This loops over the number of cosmic rays (jumps).  That is, all the
    ramps with no cosmic rays are processed first, then all the ramps with
    one cosmic ray, then with two, etc.

    Parameters
    ----------
    data_sect : 3-D ndarray; shape (ngroups, ny, nx)
        The ramp data for one of the integrations in an exposure.  This
        may be a subarray in detector coordinates, but covering all groups.

    input_var_sect : 3-D ndarray, shape (ngroups, ny, nx)
        The square of the input ERR array, matching data_sect.

    gdq_sect : 3-D ndarray; shape (ngroups, ny, nx)
        The group data quality array.  This may be a subarray, matching
        data_sect.

    readnoise_sect : 2-D ndarray; shape (ny, nx)
        The read noise in electrons at each detector pixel (i.e. not a
        ramp).  This may be a subarray, similar to data_sect.

    gain_sect : 2-D ndarray, or None; shape (ny, nx)
        The gain in electrons per DN at each detector pixel (i.e. not a
        ramp).  This may be a subarray, matching readnoise_sect.  If
        gain_sect is None, a value of 1 will be assumed.

    prev_fit : 3-D ndarray; shape (ngroups, ny, nx)
        The previous fit (intercept, slope, cosmic-ray amplitudes)
        evaluated for each pixel in the subarray.  data_sect itself may be
        used for the first iteration.

    prev_slope_sect : 2-D ndarray; shape (ny, nx)
        An estimate (e.g. from a previous iteration) of the slope at each
        pixel, in electrons per second.  This may be a subarray, similar to
        data_sect.

    frame_time : float
        The time to read one frame, in seconds (e.g. 10.6 s).

    group_time : float
        Time increment between groups, in seconds.

    nframes_used : int
        Number of frames that were averaged together to make a group.
        This value does not include the number (if any) of skipped frames.

    max_num_cr : non-negative int
        The maximum number of cosmic rays that should be handled.

    saturated_flag : int
        The saturation flag.

    jump_flag : int
        The jump detection flag.

    use_extra_terms : bool
        True if we should include Massimo Robberto's terms in the
        covariance matrix.
        See JWST-STScI-003193.pdf

    Returns
    -------
    tuple :  (intercept_sect, int_var_sect, slope_sect, slope_var_sect,
             cr_sect, cr_var_sect)
        intercept_sect is a 2-D ndarray, the intercept of the ramp at each
        pixel of data_sect.
        int_var_sect is a 2-D ndarray, the variance of the intercept at
        each pixel of data_sect.
        slope_sect is a 2-D ndarray, the ramp slope at each pixel of
        data_sect.
        slope_var_sect is a 2-D ndarray, the variance of the slope at each
        pixel of data_sect.
        cr_sect is a 3-D ndarray, shape (ny, nx, cr_dimen), the amplitude
        of each cosmic ray at each pixel of data_sect.  cr_dimen is
        max_num_cr or 1, whichever is larger.
        cr_var_sect is a 3-D ndarray, the variance of each cosmic ray
        amplitude.
    """
    cr_flagged = np.empty(data_sect.shape, dtype=np.uint8)
    cr_flagged[:] = np.where(np.bitwise_and(gdq_sect, jump_flag), 1, 0)

    # If a pixel is flagged as a jump in the first group, we can't fit to
    # the ramp, because a matrix that we need to invert would be singular.
    # If there's only one group, we can't fit a ramp to it anyway, so
    # at this point we wouldn't need to be concerned about a jump.  If
    # there is more than one group, just ignore any jump the first group.
    if data_sect.shape[0] > 1:
        cr_flagged[0, :, :] = 0

    # Sum over groups to get an (ny, nx) image of the number of cosmic
    # rays in each pixel, accumulated over the ramp.
    sum_flagged = cr_flagged.sum(axis=0, dtype=np.int32)

    # If a pixel is flagged as saturated in the first or second group, we
    # don't want to even attempt to fit a slope to the ramp for that pixel.
    # Handle this case by setting the corresponding pixel in sum_flagged to
    # a negative number.  The test `ncr_mask = (sum_flagged == num_cr)`
    # will therefore never match, since num_cr is zero or larger, and the
    # pixel will not be included in any ncr_mask.
    mask1 = (gdq_sect[0, :, :] == saturated_flag)
    sum_flagged[mask1] = -1

    # one_group_mask flags pixels that are not saturated in the first
    # group but are saturated in the second group (if there is a second
    # group).  For these pixels, we will assign a value to the slope
    # image by just dividing the value in the first group by group_time.
    if len(gdq_sect) > 1:
        mask2 = (gdq_sect[1, :, :] == saturated_flag)
        sum_flagged[mask2] = -1
        one_group_mask = np.bitwise_and(mask2, np.bitwise_not(mask1))
        del mask2
    else:
        one_group_mask = np.bitwise_not(mask1)
    del mask1

    # Set elements of this array to a huge value if the corresponding
    # pixels are saturated.  This is not a flag, it's a value to be
    # added to the diagonal of the covariance matrix.
    saturated = np.empty(data_sect.shape, dtype=np.float64)
    saturated[:] = np.where(
        np.bitwise_and(gdq_sect, saturated_flag), HUGE_FOR_LOW_WEIGHT, 0.)

    # Create arrays to be populated and then returned.
    shape = data_sect.shape

    # Lower limit of one, in case there are no cosmic rays at all.
    cr_dimen = max(1, max_num_cr)
    intercept_sect = np.zeros((shape[1], shape[2]), dtype=data_sect.dtype)
    slope_sect = np.zeros((shape[1], shape[2]), dtype=data_sect.dtype)
    cr_sect = np.zeros((shape[1], shape[2], cr_dimen), dtype=data_sect.dtype)

    int_var_sect = np.zeros((shape[1], shape[2]), dtype=data_sect.dtype)
    slope_var_sect = np.zeros((shape[1], shape[2]), dtype=data_sect.dtype)
    cr_var_sect = np.zeros((shape[1], shape[2], cr_dimen), dtype=data_sect.dtype)

    # This takes care of the case that there's only one group, as well as
    # pixels that are saturated in the second but not the first group of a
    # multi-group file
    if one_group_mask.any():
        slope_sect[one_group_mask] = data_sect[0, one_group_mask] / group_time
    del one_group_mask

    # Fit slopes for all pixels that have no cosmic ray hits anywhere in
    # the ramp, then fit slopes with one CR hit, then with two, etc.
    for num_cr in range(max_num_cr + 1):
        ngroups = len(data_sect)
        ncr_mask = (sum_flagged == num_cr)

        # Number of detector pixels flagged with num_cr CRs within the ramp.
        nz = ncr_mask.sum(dtype=np.int32)
        if nz <= 0:
            continue

        # ramp_data will be a ramp with a 1-D array of pixels copied out
        # of data_sect.
        ramp_data = np.empty((ngroups, nz), dtype=data_sect.dtype)
        input_var_data = np.empty((ngroups, nz), dtype=data_sect.dtype)
        prev_fit_data = np.empty((ngroups, nz), dtype=prev_fit.dtype)
        prev_slope_data = np.empty(nz, dtype=prev_slope_sect.dtype)
        prev_slope_data[:] = prev_slope_sect[ncr_mask]
        readnoise = np.empty(nz, dtype=readnoise_sect.dtype)
        readnoise[:] = readnoise_sect[ncr_mask]

        if gain_sect is None:
            gain = None
        else:
            gain = np.empty(nz, dtype=gain_sect.dtype)
            gain[:] = gain_sect[ncr_mask]

        cr_flagged_2d = np.empty((ngroups, nz), dtype=cr_flagged.dtype)
        saturated_data = np.empty((ngroups, nz), dtype=prev_fit.dtype)
        for k in range(ngroups):
            ramp_data[k] = data_sect[k][ncr_mask]
            input_var_data[k] = input_var_sect[k][ncr_mask]
            prev_fit_data[k] = prev_fit[k][ncr_mask]
            cr_flagged_2d[k] = cr_flagged[k][ncr_mask]
            # This is for clobbering saturated pixels.
            saturated_data[k] = saturated[k][ncr_mask]

        result, variances = gls_fit(
            ramp_data, prev_fit_data, prev_slope_data, readnoise, gain, frame_time,
            group_time, nframes_used, num_cr, cr_flagged_2d, saturated_data)

        # Copy the intercept, slope, and cosmic-ray amplitudes and their
        # variances to the arrays to be returned.
        # ncr_mask is a mask array that is True for each pixel that has the
        # current number (num_cr) of cosmic rays.  Thus, the output arrays
        # are being populated here in sets, a different set of pixels with
        # each iteration of this loop.
        intercept_sect[ncr_mask] = result[:, 0].copy()
        int_var_sect[ncr_mask] = variances[:, 0].copy()
        slope_sect[ncr_mask] = result[:, 1].copy()
        slope_var_sect[ncr_mask] = variances[:, 1].copy()

        # In this loop, i is just an index.  cr_sect is populated for
        # number of cosmic rays = 1 to num_cr, inclusive.
        for i in range(num_cr):
            cr_sect[ncr_mask, i] = result[:, 2 + i].copy()
            cr_var_sect[ncr_mask, i] = variances[:, 2 + i].copy()

    return (intercept_sect, int_var_sect, slope_sect, slope_var_sect,
            cr_sect, cr_var_sect)


def gls_fit(ramp_data, prev_fit_data, prev_slope_data, readnoise, gain, frame_time,
            group_time, nframes_used, num_cr, cr_flagged_2d, saturated_data):
    """Generalized least squares linear fit.

    It is assumed that every input pixel has num_cr cosmic-ray hits
    somewhere within the ramp.  This function should be called separately
    for different values of num_cr.

    Notes
    -----
    Curently the noise model is assumed to be a combination of
    read and photon noise alone.
    Same technique could be used with more complex noise models, but then
    the ramp covariance matrix should be input.

    Parameters
    ----------
    ramp_data : 2-D ndarray; indices:  group, pixel number
        The ramp data for one of the integrations in an exposure.  This
        may be a subset in detector coordinates, but covering all groups.
        The shape is (ngroups, nz), where ngroups is the length of the
        ramp, and nz is the number of pixels in the current subset.

    prev_fit_data : 2-D ndarray, shape (ngroups, nz)
        The fit to ramp_data, based on applying the values of intercept,
        slope, and cosmic-ray amplitudes that were determined in a previous
        call to gls_fit.  This array is only used for setting up the
        covariance matrix.

    prev_slope_data : 1-D ndarray, length nz.
        An estimate (e.g. from a previous iteration) of the slope at each
        pixel, in electrons per second.

    readnoise : 1-D ndarray, length nz.
        The read noise in electrons at each detector pixel.

    gain : 1-D ndarray, shape (nz,)
        The analog-to-digital gain (electrons per dn) at each detector
        pixel.

    frame_time : float
        The time to read one frame, in seconds (e.g. 10.6 s).

    group_time : float
        Time increment between groups, in seconds.

    nframes_used : int
        Number of frames that were averaged together to make a group.
        Note that this value does not include the number (if any) of
        skipped frames.

    num_cr : int
        The number of cosmic rays that will be handled.  All pixels in the
        current set (ramp_data) are assumed to have this many cosmic ray
        hits somewhere within the ramp.

    cr_flagged_2d : 2-D ndarray, shape (ngroups, nz)
        The values should be 0 or 1; 1 indicates that a cosmic ray was
        detected (by another step) at that point.

    saturated_data : 2-D ndarray, shape (ngroups, nz)
        Normal values are zero; the value will be a huge number for
        saturated pixels.  This will be added to the main diagonal of the
        inverse of the weight matrix to greatly reduce the weight for
        saturated pixels.

    Returns
    -------
    tuple :  (result2d, variances)
        result2d is a 2-D ndarray; shape (nz, 2 + num_cr)
        The computed values of intercept, slope, and cosmic-ray amplitudes
        (there will be num_cr cosmic-ray amplitudes) for each of the nz
        pixels.

        variances is a 2-D ndarray; shape (nz, 2 + num_cr)
        The variance for the intercept, slope, and for the amplitude of
        each cosmic ray that was detected.
    """

    M = float(nframes_used)

    ngroups = ramp_data.shape[0]
    nz = ramp_data.shape[1]
    num_cr = int(num_cr)

    # x is an array (length nz) of matrices, each of which is the
    # independent variable of a linear equation.  Each such matrix
    # has ngroups rows and 2 + num_cr columns.  The first column is set
    # to 1, for finding the intercept.  The second column is the time at
    # each group, for finding the slope.  The remaining columns (if any),
    # are 0 for all rows prior to a certain point, then 1 for all
    # subsequent rows (i.e. the Heaviside function).  The transition from
    # 0 to 1 is the location of a cosmic ray hit; the first 1 in a column
    # corresponds to the value in cr_flagged_2d being 1.
    x = np.zeros((nz, ngroups, 2 + num_cr), dtype=np.float64)
    x[:, :, 0] = 1.
    x[:, :, 1] = np.arange(ngroups, dtype=np.float64) * group_time + \
        frame_time * (M + 1.) / 2.

    if num_cr > 0:
        sum_crs = cr_flagged_2d.cumsum(axis=0)
        for k in range(ngroups):
            s = slice(k, ngroups)
            for n in range(1, num_cr + 1):
                temp = np.where(np.logical_and(cr_flagged_2d[k] == 1,
                                               sum_crs[k] == n))
                if len(temp[0]) > 0:
                    index = (temp[0], s, n + 1)
                    x[index] = 1
        del temp, index

    y = np.transpose(ramp_data, (1, 0)).reshape((nz, ngroups, 1))

    # ramp_cov is an array of nz matrices, each ngroups x ngroups.
    # each matrix gives the covariance of that pixel's ramp data
    ramp_cov = np.ones((nz, ngroups, ngroups), dtype=np.float64)

    # Use the previous fit to the data to populate the covariance matrix,
    # for each of the nz pixels.  prev_fit_data has shape (ngroups, nz),
    # similar to the ramp data, but we want the nz axis to be the first
    # (we're constructing an array of nz matrix equations), so transpose
    # prev_fit_data.
    prev_fit_T = np.transpose(prev_fit_data, (1, 0))
    for k in range(ngroups):
        # Populate the upper right, row by row.
        ramp_cov[:, k, k:ngroups] = prev_fit_T[:, k:k + 1]
        # Populate the lower left, column by column.
        ramp_cov[:, k:ngroups, k] = prev_fit_T[:, k:k + 1]
        # Give saturated pixels a very high high variance (hence a low weight)
        ramp_cov[:, k, k] += saturated_data[k, :]
    del prev_fit_T

    # iden is 2-D, but it can broadcast to 4-D.  This is used to add terms to
    # the diagonal of the covariance matrix.
    iden = np.identity(ngroups)

    rn3d = readnoise.reshape((nz, 1, 1))
    ramp_cov += (iden * rn3d**2)

    # prev_slope_data must be non-negative.
    flags = prev_slope_data < 0.
    prev_slope_data[flags] = 1.

    # The resulting fit parameters are
    #  (xT @ ramp_cov^-1 @ x)^-1 @ [xT @ ramp_cov^-1 @ y]
    #  = [y-intercept, slope, cr_amplitude_1, cr_amplitude_2, ...]
    # where @ means matrix multiplication.

    # shape of xT is (nz, 2 + num_cr, ngroups)
    xT = np.transpose(x, (0, 2, 1))

    # shape of `ramp_invcov` is (nz, ngroups, ngroups)
    iden = iden.reshape((1, ngroups, ngroups))
    ramp_invcov = la.solve(ramp_cov, iden)

    del iden

    # temp1 = xT @ ramp_invcov
    # np.einsum use is equivalent to matrix multiplication
    # shape of temp1 is (nz, 2 + num_cr, ngroups)
    temp1 = np.einsum('...ij,...jk->...ik', xT, ramp_invcov)

    # temp_var = xT @ ramp_invcov @ x
    # shape of temp_var is (nz, 2 + num_cr, 2 + num_cr)
    temp_var = np.einsum('...ij,...jk->...ik', temp1, x)

    # `fitparam_cov` is an array of nz covariance matrices.
    # fitparam_cov = (xT @ ramp_invcov @ x)^-1
    # shape of fitparam_covar is (nz, 2 + num_cr, 2 + num_cr)
    I_2 = np.eye(2 + num_cr).reshape((1, 2 + num_cr, 2 + num_cr))
    try:
        # inverse of temp_var
        fitparam_cov = la.solve(temp_var, I_2)
    except la.LinAlgError:
        # find the pixel with the singular matrix
        for z in range(nz):
            try:
                la.solve(temp_var[z], I_2)
            except la.LinAlgError as msg2:
                log.warning("singular matrix, z = %d" % z)
                raise la.LinAlgError(msg2)
    del I_2

    # [xT @ ramp_invcov @ y]
    # shape of temp2 is (nz, 2 + num_cr, 1)
    temp2 = np.einsum('...ij,...jk->...ik', temp1, y)

    # shape of fitparam is (nz, 2 + num_cr, 1)
    fitparam = np.einsum('...ij,...jk->...ik', fitparam_cov, temp2)
    r_shape = fitparam.shape
    fitparam2d = fitparam.reshape((r_shape[0], r_shape[1]))
    del fitparam

    # shape of both result2d and variances is (nz, 2 + num_cr)
    fitparam_uncs = fitparam_cov.diagonal(axis1=1, axis2=2).copy()

    return fitparam2d, fitparam_uncs
