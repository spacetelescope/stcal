#! /usr/bin/env python

import logging
from multiprocessing.pool import Pool as Pool
import numpy as np
import time

import warnings

from . import ramp_fit_class
from . import utils


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

BUFSIZE = 1024 * 300000  # 300Mb cache size for data section

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def ols_ramp_fit_multi(
        ramp_data, buffsize, save_opt, readnoise_2d, gain_2d, weighting, max_cores):
    """
    Setup the inputs to ols_ramp_fit with and without multiprocessing. The
    inputs will be sliced into the number of cores that are being used for
    multiprocessing. Because the data models cannot be pickled, only numpy
    arrays are passed and returned as parameters to ols_ramp_fit.

    Parameters
    ----------
    ramp_data : RampData
        Input data necessary for computing ramp fitting.

    buffsize : int
        size of data section (buffer) in bytes (not used)

    save_opt : bool
       calculate optional fitting results

    readnoise_2d : ndarray
        readnoise for all pixels

    gain_2d : ndarray
        gain for all pixels

    algorithm : str
        'OLS' specifies that ordinary least squares should be used;
        'GLS' specifies that generalized least squares should be used.

    weighting : str
        'optimal' specifies that optimal weighting should be used;
         currently the only weighting supported.

    max_cores : str
        Number of cores to use for multiprocessing. If set to 'none' (the default),
        then no multiprocessing will be done. The other allowable values are 'quarter',
        'half', and 'all'. This is the fraction of cores to use for multi-proc. The
        total number of cores includes the SMT cores (Hyper Threading for Intel).

    Returns
    -------
    image_info : tuple
        The tuple of computed ramp fitting arrays.

    integ_info : tuple
        The tuple of computed integration fitting arrays.

    opt_info : tuple
        The tuple of computed optional results arrays for fitting.
    """

    # Determine number of slices to use for multi-processor computations
    number_slices = utils.compute_slices(max_cores)

    # For MIRI datasets having >1 group, if all pixels in the final group are
    #   flagged as DO_NOT_USE, resize the input model arrays to exclude the
    #   final group.  Similarly, if leading groups 1 though N have all pixels
    #   flagged as DO_NOT_USE, those groups will be ignored by ramp fitting, and
    #   the input model arrays will be resized appropriately. If all pixels in
    #   all groups are flagged, return None for the models.
    if ramp_data.instrument_name == 'MIRI' and ramp_data.data.shape[1] > 1:
        miri_ans = discard_miri_groups(ramp_data)
        # The function returns False if the removed groups leaves no data to be
        # processed.  If this is the case, return None for all expected variables
        # returned by ramp_fit
        if miri_ans is not True:
            return [None] * 3

    # There is nothing to do if all ramps in all integrations are saturated.
    first_gdq = ramp_data.groupdq[:, 0, :, :]
    if np.all(np.bitwise_and(first_gdq, ramp_data.flags_saturated)):
        return None, None, None

    # Call ramp fitting for the single processor (1 data slice) case
    if number_slices == 1:
        # Single threaded computation
        image_info, integ_info, opt_info = ols_ramp_fit_single(
            ramp_data, buffsize, save_opt, readnoise_2d, gain_2d, weighting)
        if image_info is None or integ_info is None:
            return None, None, None

        return image_info, integ_info, opt_info

    # Call ramp fitting for multi-processor (multiple data slices) case
    else:
        image_info, integ_info, opt_info = ols_ramp_fit_multiprocessing(
            ramp_data, buffsize, save_opt,
            readnoise_2d, gain_2d, weighting, number_slices)

        return image_info, integ_info, opt_info


def ols_ramp_fit_multiprocessing(
        ramp_data, buffsize, save_opt,
        readnoise_2d, gain_2d, weighting, number_slices):
    """
    Fit a ramp using ordinary least squares. Calculate the count rate for each
    pixel in all data cube sections and all integrations, equal to the weighted
    slope for all sections (intervals between cosmic rays) of the pixel's ramp
    divided by the effective integration time.  The data is spread across the
    desired number of processors (>1).

    Parameters
    ----------
    ramp_data: RampData
        Input data necessary for computing ramp fitting.

    buffsize : int
        The working buffer size

    save_opt : bool
        Whether to return the optional output model

    readnoise_2d : ndarray
        The read noise of each pixel

    gain_2d : ndarray
        The gain of each pixel

    weighting : str
        'optimal' is the only valid value

    number_slices: int
        The number of slices to partition the data into for multiprocessing.

    Return
    ------
    image_info: tuple
        The tuple of computed ramp fitting arrays.

    integ_info: tuple
        The tuple of computed integration fitting arrays.

    opt_info: tuple
        The tuple of computed optional results arrays for fitting.
    """
    log.info(f"Number of processors used for multiprocessing: {number_slices}")
    slices, rows_per_slice = compute_slices_for_starmap(
        ramp_data, buffsize, save_opt,
        readnoise_2d, gain_2d, weighting, number_slices)

    pool = Pool(processes=number_slices)
    pool_results = pool.starmap(ols_ramp_fit_single, slices)
    pool.close()
    pool.join()

    # Reassemble results
    image_info, integ_info, opt_info = assemble_pool_results(
        ramp_data, save_opt, pool_results, rows_per_slice)

    return image_info, integ_info, opt_info


def assemble_pool_results(ramp_data, save_opt, pool_results, rows_per_slice):
    """
    Takes the list of results from the starmap pool method and assembles the
    slices into primary tuples to be returned by `ramp_fit`.

    Parameters
    ----------
    ramp_data: RampData
        The data needed for ramp fitting.

    save_opt: bool
        The option to save the optional results.

    pool_results: list
        The list of return values from ols_ramp_fit_single for each slice.
        Each slice is run through ols_ramp_fit_single, which returns three
        tuples of ndarrays, so pool_results is a list of tuples.  Each tuple
        contains:
            image_info, integ_info, opt_info

    rows_per_slice: list
        The number of rows in each slice.

    Return
    ------
    image_info: tuple
        The tuple of computed ramp fitting arrays.

    integ_info: tuple
        The tuple of computed integration fitting arrays.

    opt_info: tuple
        The tuple of computed optional results arrays for fitting.
    """
    # Create output arrays for each output tuple.  The input ramp data and
    # slices are needed for this.
    image_info, integ_info, opt_info = create_output_info(
        ramp_data, pool_results, save_opt)

    # Loop over the slices and assemble each slice into the main return arrays.
    current_row_start = 0
    for k, result in enumerate(pool_results):
        image_slice, integ_slice, opt_slice = result
        nrows = rows_per_slice[k]

        get_image_slice(image_info, image_slice, current_row_start, nrows)
        get_integ_slice(integ_info, integ_slice, current_row_start, nrows)
        if save_opt:
            get_opt_slice(opt_info, opt_slice, current_row_start, nrows)
        current_row_start = current_row_start + nrows

    # Handle integration times
    return image_info, integ_info, opt_info


def get_image_slice(image_info, image_slice, row_start, nrows):
    """
    Populates the image output information from each slice.

    image_info: tuple
        The output image information to populate from the slice.

    image_slice: tuple
        The output slice used to populate the output arrays.

    row_start: int
        The start row the current slice at which starts.

    nrows: int
        The number of rows int the current slice.
    """
    data, dq, var_poisson, var_rnoise, err = image_info
    sdata, sdq, svar_poisson, svar_rnoise, serr = image_slice

    srow, erow = row_start, row_start + nrows

    data[srow:erow, :] = sdata
    dq[srow:erow, :] = sdq
    var_poisson[srow:erow, :] = svar_poisson
    var_rnoise[srow:erow, :] = svar_rnoise
    err[srow:erow, :] = serr


def get_integ_slice(integ_info, integ_slice, row_start, nrows):
    """
    Populates the integration output information from each slice.

    integ_info: tuple
        The output integration information to populate from the slice.

    integ_slice: tuple
        The output slice used to populate the output arrays.

    row_start: int
        The start row the current slice at which starts.

    nrows: int
        The number of rows int the current slice.
    """
    data, dq, var_poisson, var_rnoise, err = integ_info
    idata, idq, ivar_poisson, ivar_rnoise, ierr = integ_slice

    srow, erow = row_start, row_start + nrows

    data[:, srow:erow, :] = idata
    dq[:, srow:erow, :] = idq
    var_poisson[:, srow:erow, :] = ivar_poisson
    var_rnoise[:, srow:erow, :] = ivar_rnoise
    err[:, srow:erow, :] = ierr


def get_opt_slice(opt_info, opt_slice, row_start, nrows):
    """
    Populates the optional output information from each slice.

    opt_info: tuple
        The output optional information to populate from the slice.

    opt_slice: tuple
        The output slice used to populate the output arrays.

    row_start: int
        The start row the current slice at which starts.

    nrows: int
        The number of rows int the current slice.
    """
    (slope, sigslope, var_poisson, var_rnoise,
        yint, sigyint, pedestal, weights, crmag) = opt_info
    (oslope, osigslope, ovar_poisson, ovar_rnoise,
        oyint, osigyint, opedestal, oweights, ocrmag) = opt_slice

    srow, erow = row_start, row_start + nrows

    # The optional results product is of variable size in its second dimension.
    # The number of segments/cosmic rays determine the final products size.
    # Because each slice is computed indpendently, the number of segments may
    # differ from segment to segment.  The final output product is created
    # using the max size for this dimension.  To ensure correct assignment is
    # done during this step, the second dimension, as well as the row
    # dimension, must be specified.
    slope[:, :oslope.shape[1], srow:erow, :] = oslope
    sigslope[:, :osigslope.shape[1], srow:erow, :] = osigslope
    var_poisson[:, :ovar_poisson.shape[1], srow:erow, :] = ovar_poisson
    var_rnoise[:, :ovar_rnoise.shape[1], srow:erow, :] = ovar_rnoise
    yint[:, :oyint.shape[1], srow:erow, :] = oyint
    sigyint[:, :osigyint.shape[1], srow:erow, :] = osigyint
    weights[:, :oweights.shape[1], srow:erow, :] = oweights
    crmag[:, :ocrmag.shape[1], srow:erow, :] = ocrmag

    pedestal[:, srow:erow, :] = opedestal  # Different shape (3-D, not 4-D)


def create_output_info(ramp_data, pool_results, save_opt):
    """
    Creates the output arrays and tuples for ramp fitting reassembly for
    mulitprocessing.

    Parameters
    ----------
    ramp_data: RampData
        The original ramp fitting data.

    pool_results: list
        The list of results for each slice from multiprocessing.

    save_opt: bool
        The option to save optional results.
    """
    tot_ints, tot_ngroups, tot_rows, tot_cols = ramp_data.data.shape

    imshape = (tot_rows, tot_cols)
    integ_shape = (tot_ints, tot_rows, tot_cols)

    # Create the primary product
    data = np.zeros(imshape, dtype=np.float32)
    dq = np.zeros(imshape, dtype=np.uint32)
    var_poisson = np.zeros(imshape, dtype=np.float32)
    var_rnoise = np.zeros(imshape, dtype=np.float32)
    err = np.zeros(imshape, dtype=np.float32)

    image_info = (data, dq, var_poisson, var_rnoise, err)

    # Create the integration products
    idata = np.zeros(integ_shape, dtype=np.float32)
    idq = np.zeros(integ_shape, dtype=np.uint32)
    ivar_poisson = np.zeros(integ_shape, dtype=np.float32)
    ivar_rnoise = np.zeros(integ_shape, dtype=np.float32)
    ierr = np.zeros(integ_shape, dtype=np.float32)

    integ_info = (idata, idq, ivar_poisson, ivar_rnoise, ierr)

    # Create the optional results product
    if save_opt:
        max_segs, max_crs = get_max_segs_crs(pool_results)
        opt_shape = (tot_ints, max_segs, tot_rows, tot_cols)
        crmag_shape = (tot_ints, max_crs, tot_rows, tot_cols)

        oslope = np.zeros(opt_shape, dtype=np.float32)
        osigslope = np.zeros(opt_shape, dtype=np.float32)
        ovar_poisson = np.zeros(opt_shape, dtype=np.float32)
        ovar_rnoise = np.zeros(opt_shape, dtype=np.float32)
        oyint = np.zeros(opt_shape, dtype=np.float32)
        osigyint = np.zeros(opt_shape, dtype=np.float32)
        oweights = np.zeros(opt_shape, dtype=np.float32)

        # Different shape
        opedestal = np.zeros(integ_shape, dtype=np.float32)
        ocrmag = np.zeros(crmag_shape, dtype=np.float32)

        opt_info = (oslope, osigslope, ovar_poisson, ovar_rnoise,
                    oyint, osigyint, opedestal, oweights, ocrmag)
    else:
        opt_info = None

    return image_info, integ_info, opt_info


def get_max_segs_crs(pool_results):
    """
    Computes the max number of segments computed needed for the second
    dimension of the optional results output.

    Parameter
    ---------
    pool_results: list
        The list of results for each slice from multiprocessing.

    Return
    ------
    seg_max : int
        The maximum segment computed over all slices.

    crs_max : int
        The maximum CRs computed over all slices.
    """
    seg_max = 1
    crs_max = 0
    for result in pool_results:
        image_slice, integ_slice, opt_slice = result
        oslice_slope = opt_slice[0]
        nsegs = oslice_slope.shape[1]
        if nsegs > seg_max:
            seg_max = nsegs

        olice_crmag = opt_slice[-1]
        ncrs = olice_crmag.shape[1]
        if ncrs > crs_max:
            crs_max = ncrs
    return seg_max, crs_max


def compute_slices_for_starmap(
        ramp_data, buffsize, save_opt,
        readnoise_2d, gain_2d, weighting, number_slices):
    """
    Creates the slices needed for each process for multiprocessing.  The slices
    for the arguments needed for ols_ramp_fit_single.

    ramp_data: RampData
        The ramp data to be sliced.

    buffsize : int
        The working buffer size

    save_opt : bool
        Whether to return the optional output model

    readnoise_2d : ndarray
        The read noise of each pixel

    gain_2d : ndarray
        The gain of each pixel

    weighting : str
        'optimal' is the only valid value

    number_slices: int
        The number of slices to partition the data into for multiprocessing.

    Return
    ------
    slices : list
        The list of arguments for each processor for multiprocessing.

    rslices : list
        The list of the number of rows in each slice.
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
            (ramp_slice, buffsize, save_opt,
             rnoise_slice, gain_slice, weighting))
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

    ramp_data_slice.set_arrays(
        data, err, groupdq, pixeldq)

    if ramp_data.zeroframe is not None:
        ramp_data_slice.zeroframe = ramp_data.zeroframe[:, start_row:start_row + nrows, :].copy()

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


def ols_ramp_fit_single(
        ramp_data, buffsize, save_opt, readnoise_2d, gain_2d, weighting):
    """
    Fit a ramp using ordinary least squares. Calculate the count rate for each
    pixel in all data cube sections and all integrations, equal to the weighted
    slope for all sections (intervals between cosmic rays) of the pixel's ramp
    divided by the effective integration time.

    Parameters
    ----------
    ramp_data : RampData
        Input data necessary for computing ramp fitting.

    buffsize : int
        The working buffer size

    save_opt : bool
        Whether to return the optional output model

    readnoise_2d : ndarray
        The read noise of each pixel

    gain_2d : ndarray
        The gain of each pixel

    weighting : str
        'optimal' is the only valid value

    Return
    ------
    image_info : tuple
        The tuple of computed ramp fitting arrays.

    integ_info : tuple
        The tuple of computed integration fitting arrays.

    opt_info : tuple
        The tuple of computed optional results arrays for fitting.
    """
    tstart = time.time()

    if not ramp_data.suppress_one_group_ramps and ramp_data.zeroframe is not None:
        zframe_locs, cnt = utils.use_zeroframe_for_saturated_ramps(ramp_data)
        ramp_data.zframe_locs = zframe_locs
        ramp_data.cnt = cnt

    # Save original shapes for writing to log file, as these may change for MIRI
    n_int, ngroups, nrows, ncols = ramp_data.data.shape
    orig_ngroups = ngroups
    orig_cubeshape = (ngroups, nrows, ncols)

    if ngroups == 1:
        log.warning('Dataset has NGROUPS=1, so count rates for each integration ')
        log.warning('will be calculated as the value of that 1 group divided by ')
        log.warning('the group exposure time.')

    # In this 'First Pass' over the data, loop over integrations and data
    #   sections to calculate the estimated median slopes, which will be used
    #   to calculate the variances. This is the same method to estimate slopes
    #   as is done in the jump detection step, except here CR-affected and
    #   saturated groups have already been flagged. The actual, fit, slopes for
    #   each segment are also calculated here.
    fit_slopes_ans = ramp_fit_slopes(
        ramp_data, gain_2d, readnoise_2d, save_opt, weighting)
    if fit_slopes_ans[0] == "saturated":
        return fit_slopes_ans[1:]

    # In this 'Second Pass' over the data, loop over integrations and data
    #   sections to calculate the variances of the slope using the estimated
    #   median slopes from the 'First Pass'. These variances are due to Poisson
    #   noise only, read noise only, and the combination of Poisson noise and
    #   read noise. The integration-specific variances are 3D arrays, and the
    #   segment-specific variances are 4D arrays.
    variances_ans = ramp_fit_compute_variances(
        ramp_data, gain_2d, readnoise_2d, fit_slopes_ans)

    # Now that the segment-specific and integration-specific variances have
    #   been calculated, the segment-specific, integration-specific, and
    #   overall slopes will be calculated. The integration-specific slope is
    #   calculated as a weighted average of the segments in the integration:
    #     slope_int = sum_over_segs(slope_seg/var_seg)/ sum_over_segs(1/var_seg)
    #  The overall slope is calculated as a weighted average of the segments in
    #     all integrations:
    #     slope = sum_over_integs_and_segs(slope_seg/var_seg)/
    #                    sum_over_integs_and_segs(1/var_seg)
    image_info, integ_info, opt_info = ramp_fit_overall(
        ramp_data, orig_cubeshape, orig_ngroups, buffsize, fit_slopes_ans,
        variances_ans, save_opt, tstart)

    return image_info, integ_info, opt_info


def discard_miri_groups(ramp_data):
    """
    For MIRI datasets having >1 group, if all pixels in the final group are
    flagged as DO_NOT_USE, resize the input model arrays to exclude the
    final group.  Similarly, if leading groups 1 though N have all pixels
    flagged as DO_NOT_USE, those groups will be ignored by ramp fitting, and
    the input model arrays will be resized appropriately. If all pixels in
    all groups are flagged, return None for the models.

    Parameters
    ----------
    ramp_data : RampData
        Input data necessary for computing ramp fitting.

    Returns
    -------
    bool :
        False if no data to process after discarding unusable data.
        True if useable data available for further processing.
    """
    data = ramp_data.data
    err = ramp_data.err
    groupdq = ramp_data.groupdq

    n_int, ngroups, nrows, ncols = data.shape

    num_bad_slices = 0  # number of initial groups that are all DO_NOT_USE

    while np.all(np.bitwise_and(groupdq[:, 0, :, :], ramp_data.flags_do_not_use)):
        num_bad_slices += 1
        ngroups -= 1

        # Check if there are remaining groups before accessing data
        if ngroups < 1:  # no usable data
            log.error('1. All groups have all pixels flagged as DO_NOT_USE,')
            log.error('  so will not process this dataset.')
            return False

        groupdq = groupdq[:, 1:, :, :]

        # Where the initial group of the just-truncated data is a cosmic ray,
        #   remove the JUMP_DET flag from the group dq for those pixels so
        #   that those groups will be included in the fit.
        wh_cr = np.where(np.bitwise_and(groupdq[:, 0, :, :], ramp_data.flags_jump_det))
        num_cr_1st = len(wh_cr[0])

        for ii in range(num_cr_1st):
            groupdq[wh_cr[0][ii], 0, wh_cr[1][ii], wh_cr[2][ii]] -= ramp_data.flags_jump_det

    if num_bad_slices > 0:
        data = data[:, num_bad_slices:, :, :]
        err = err[:, num_bad_slices:, :, :]

    log.info('Number of leading groups that are flagged as DO_NOT_USE: %s', num_bad_slices)

    # If all groups were flagged, the final group would have been picked up
    #   in the while loop above, ngroups would have been set to 0, and Nones
    #   would have been returned.  If execution has gotten here, there must
    #   be at least 1 remaining group that is not all flagged.
    if np.all(np.bitwise_and(groupdq[:, -1, :, :], ramp_data.flags_do_not_use)):
        ngroups -= 1

        # Check if there are remaining groups before accessing data
        if ngroups < 1:  # no usable data
            log.error('2. All groups have all pixels flagged as DO_NOT_USE,')
            log.error('  so will not process this dataset.')
            return False

        data = data[:, :-1, :, :]
        err = err[:, :-1, :, :]
        groupdq = groupdq[:, :-1, :, :]

        log.info('MIRI dataset has all pixels in the final group flagged as DO_NOT_USE.')

    # Next block is to satisfy github issue 1681:
    # "MIRI FirstFrame and LastFrame minimum number of groups"
    if ngroups < 2:
        log.warning('MIRI datasets require at least 2 groups/integration')
        log.warning('(NGROUPS), so will not process this dataset.')
        return False

    ramp_data.data = data
    ramp_data.err = err
    ramp_data.groupdq = groupdq

    return True


def ramp_fit_slopes(ramp_data, gain_2d, readnoise_2d, save_opt, weighting):
    """
    Calculate effective integration time (once EFFINTIM has been populated accessible, will
    use that instead), and other keywords that will needed if the pedestal calculation is
    requested. Note 'nframes' is the number of given by the NFRAMES keyword, and is the
    number of frames averaged on-board for a group, i.e., it does not include the groupgap.

    Parameters
    ----------
    ramp_data : RampData
        Input data necessary for computing ramp fitting.

    gain_2d : ndarrays
        gain for all pixels

    readnoise_2d : ndarrays
        readnoise for all pixels

    save_opt : bool
       calculate optional fitting results

    weighting : str
        'optimal' specifies that optimal weighting should be used;
         currently the only weighting supported.

    Return
    ------
    max_seg : int
        Maximum possible number of segments over all groups and segments

    gdq_cube_shape : ndarray
        Group DQ dimensions

    effintim : float
        effective integration time for a single group

    f_max_seg : int
        Actual maximum number of segments over all groups and segments

    dq_int : ndarray
        The pixel dq for each integration for each pixel

    num_seg_per_int : ndarray
        Cube of numbers of segments for all integrations and pixels, 3-D int

    sat_0th_group_int : ndarray
        Integration-specific slice whose value for a pixel is 1 if the initial
        group of the ramp is saturated, 3-D uint8

    opt_res : OptRes
        Object to hold optional results for all good pixels.

    pixeldq : ndarray
        The input 2-D pixel DQ flags

    inv_var : ndarray
        values of 1/variance for good pixels, 1-D float

    med_rates : ndarray
        Rate array
    """
    # Get image data information
    data = ramp_data.data
    err = ramp_data.err
    groupdq = ramp_data.groupdq
    inpixeldq = ramp_data.pixeldq

    # Get instrument and exposure data
    frame_time = ramp_data.frame_time
    groupgap = ramp_data.groupgap
    nframes = ramp_data.nframes

    # Get needed sizes and shapes
    n_int, ngroups, nrows, ncols = data.shape
    imshape = (nrows, ncols)
    cubeshape = (ngroups,) + imshape

    # Calculate effective integration time (once EFFINTIM has been populated
    #   and accessible, will use that instead), and other keywords that will
    #   needed if the pedestal calculation is requested. Note 'nframes'
    #   is the number of given by the NFRAMES keyword, and is the number of
    #   frames averaged on-board for a group, i.e., it does not include the
    #   groupgap.
    effintim = (nframes + groupgap) * frame_time

    # Get GROUP DQ and ERR arrays from input file
    gdq_cube = groupdq
    gdq_cube_shape = gdq_cube.shape

    # Get max number of segments fit in all integrations
    max_seg, num_CRs = calc_num_seg(
        gdq_cube, n_int, ramp_data.flags_jump_det, ramp_data.flags_do_not_use)
    del gdq_cube

    f_max_seg = 0  # final number to use, usually overwritten by actual value

    dq_int, num_seg_per_int, sat_0th_group_int =\
        utils.alloc_arrays_1(n_int, imshape)

    opt_res = utils.OptRes(n_int, imshape, max_seg, ngroups, save_opt)

    # Get Pixel DQ array from input file. The incoming RampModel has uint32
    #   PIXELDQ, but ramp fitting will update this array here by flagging
    #   the 2D PIXELDQ locations where the ramp data has been previously
    #   flagged as jump-detected or saturated. These additional bit values
    #   require this local variable to be uint16, and it will be used as the
    #   (uint16) PIXELDQ in the outgoing ImageModel.
    pixeldq = inpixeldq.copy()
    pixeldq = utils.reset_bad_gain(ramp_data, pixeldq, gain_2d)  # Flag bad pixels in gain

    # In this 'First Pass' over the data, loop over integrations and data
    #   sections to calculate the estimated median slopes, which will be used
    #   to calculate the variances. This is the same method to estimate slopes
    #   as is done in the jump detection step, except here CR-affected and
    #   saturated groups have already been flagged. The actual, fit, slopes for
    #   each segment are also calculated here.

    med_rates = utils.compute_median_rates(ramp_data)

    # Loop over data integrations:
    for num_int in range(0, n_int):
        # Loop over data sections
        for rlo in range(0, cubeshape[1], nrows):
            rhi = rlo + nrows

            if rhi > cubeshape[1]:
                rhi = cubeshape[1]

            # Skip data section if it is all NaNs
            # data_sect = np.float32(data[num_int, :, :, :])
            data_sect = data[num_int, :, :, :]
            if np.all(np.isnan(data_sect)):
                log.error('Current data section is all nans, so not processing the section.')
                continue

            # first frame section for 1st group of current integration
            ff_sect = data[num_int, 0, rlo:rhi, :]

            # Get appropriate sections
            gdq_sect = groupdq[num_int, :, :, :]
            rn_sect = readnoise_2d[rlo:rhi, :]
            gain_sect = gain_2d[rlo:rhi, :]

            # Reset all saturated groups in the input data array to NaN
            where_sat = np.where(np.bitwise_and(gdq_sect, ramp_data.flags_saturated))

            data_sect[where_sat] = np.NaN
            del where_sat

            # Calculate the slope of each segment
            # note that the name "opt_res", which stands for "optional results",
            # is deceiving; this in fact contains all the per-integration and
            # per-segment results that will eventually be used to compute the
            # final slopes, sigmas, etc. for the main (non-optional) products
            t_dq_cube, inv_var, opt_res, f_max_seg, num_seg = \
                calc_slope(data_sect, gdq_sect, frame_time, opt_res, save_opt, rn_sect,
                           gain_sect, max_seg, ngroups, weighting, f_max_seg, ramp_data)

            del gain_sect

            # Populate 3D num_seg { integ, y, x } with 2D num_seg for this data
            #  section (y,x) and integration (num_int)
            sect_shape = data_sect.shape[-2:]
            num_seg_per_int[num_int, rlo:rhi, :] = num_seg.reshape(sect_shape)

            # Populate integ-spec slice which is set if 0th group has SAT
            wh_sat0 = np.where(np.bitwise_and(gdq_sect[0, :, :], ramp_data.flags_saturated))
            if len(wh_sat0[0]) > 0:
                sat_0th_group_int[num_int, rlo:rhi, :][wh_sat0] = 1

            del wh_sat0

            pixeldq_sect = pixeldq[rlo:rhi, :].copy()
            dq_int[num_int, rlo:rhi, :] = utils.dq_compress_sect(
                ramp_data, num_int, t_dq_cube, pixeldq_sect).copy()

            del t_dq_cube

            # Loop over the segments and copy the reshaped 2D segment-specific
            #   results for the current data section to the 4D output arrays.
            opt_res.reshape_res(num_int, rlo, rhi, sect_shape, ff_sect, save_opt)

            if save_opt:
                # Calculate difference between each slice and the previous slice
                #   as approximation to cosmic ray amplitude for those pixels
                #   having their DQ set for cosmic rays
                data_diff = data_sect - utils.shift_z(data_sect, -1)
                dq_cr = np.bitwise_and(ramp_data.flags_jump_det, gdq_sect)

                opt_res.cr_mag_seg[num_int, :, rlo:rhi, :] = data_diff * (dq_cr != 0)

                del data_diff

            del data_sect
            del ff_sect
            del gdq_sect

    if pixeldq_sect is not None:
        del pixeldq_sect

    ramp_data.data = data
    ramp_data.err = err
    ramp_data.groupdq = groupdq
    ramp_data.pixeldq = inpixeldq

    return max_seg, gdq_cube_shape, effintim, f_max_seg, dq_int, num_seg_per_int,\
        sat_0th_group_int, opt_res, pixeldq, inv_var, med_rates


def ramp_fit_compute_variances(ramp_data, gain_2d, readnoise_2d, fit_slopes_ans):
    """
    In this 'Second Pass' over the data, loop over integrations and data
    sections to calculate the variances of the slope using the estimated
    median slopes from the 'First Pass'. These variances are due to Poisson
    noise only, read noise only, and the combination of Poisson noise and
    read noise. The integration-specific variances are 3D arrays, and the
    segment-specific variances are 4D arrays.

    The naming convention for the arrays:
        'var': a variance
        'p3': intermediate 3D array for variance due to Poisson noise
        'r4': intermediate 4D array for variance due to read noise
        'both4': intermediate 4D array for combined variance due to both Poisson and read noise
        'inv_<X>': intermediate array = 1/<X>
        's_inv_<X>': intermediate array = 1/<X>, summed over integrations


    Parameters
    ----------
    ramp_data : ramp_fit_class.RampData
        Input data necessary for computing ramp fitting.

    gain_2d : ndarray
        gain for all pixels

    readnoise_2d : ndarray
        The read noise for each pixel

    fit_slopes_ans : tuple
        Contains intermediate values computed in the first pass over the data.

    Return
    ------
    var_p3 : ndarray
        3-D variance based on Poisson noise

    var_r3 : ndarray
        3-D variance based on read noise

    var_p4 : ndarray
        4-D variance based on Poisson noise

    var_r4 : ndarray
        4-D variance based on read noise

    var_both4 : ndarray
        4-D array for combined variance due to both Poisson and read noise

    var_both3 : ndarray
        3-D array for combined variance due to both Poisson and read noise

    inv_var_both4 : ndarray
        1 / var_both4

    s_inv_var_p3 : ndarray
        1 / var_p3, summed over integrations

    s_inv_var_r3 : ndarray
        1 / var_r3, summed over integrations

    s_inv_var_both3 : ndarray
        1 / var_both3, summed over integrations
    """

    # Get image data information
    data = ramp_data.data
    err = ramp_data.err
    groupdq = ramp_data.groupdq
    inpixeldq = ramp_data.pixeldq

    # Get instrument and exposure data
    group_time = ramp_data.group_time

    # Get needed sizes and shapes
    n_int, ngroups, nrows, ncols = data.shape
    imshape = (nrows, ncols)
    cubeshape = (ngroups,) + imshape

    max_seg = fit_slopes_ans[0]
    num_seg_per_int = fit_slopes_ans[5]
    med_rates = fit_slopes_ans[10]

    var_p3, var_r3, var_p4, var_r4, var_both4, var_both3, \
        inv_var_both4, s_inv_var_p3, s_inv_var_r3, s_inv_var_both3, segs_4 = \
        utils.alloc_arrays_2(n_int, imshape, max_seg)

    # Loop over data integrations
    for num_int in range(n_int):
        ramp_data.current_integ = num_int

        # Loop over data sections
        for rlo in range(0, cubeshape[1], nrows):
            rhi = rlo + nrows

            if rhi > cubeshape[1]:
                rhi = cubeshape[1]

            gdq_sect = groupdq[num_int, :, rlo:rhi, :]
            rn_sect = readnoise_2d[rlo:rhi, :]
            gain_sect = gain_2d[rlo:rhi, :]

            # Calculate results needed to compute the variance arrays
            den_r3, den_p3, num_r3, segs_beg_3 = utils.calc_slope_vars(
                ramp_data, rn_sect, gain_sect, gdq_sect, group_time, max_seg)

            segs_4[num_int, :, rlo:rhi, :] = segs_beg_3

            # Suppress harmless arithmetic warnings for now
            warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
            warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)
            var_p4[num_int, :, rlo:rhi, :] = den_p3 * med_rates[rlo:rhi, :]

            # Find the segment variance due to read noise and convert back to DN
            var_r4[num_int, :, rlo:rhi, :] = num_r3 * den_r3 / gain_sect**2

            # Reset the warnings filter to its original state
            warnings.resetwarnings()

            del den_r3, den_p3, num_r3, segs_beg_3
            del gain_sect
            del gdq_sect

        # The next 4 statements zero out entries for non-existing segments, and
        #   set the variances for segments having negative slopes (the segment
        #   variance is proportional to the median estimated slope) to
        #   outrageously large values so that they will have negligible
        #   contributions.
        var_p4[num_int, :, :, :] *= (segs_4[num_int, :, :, :] > 0)

        # Suppress, then re-enable harmless arithmetic warnings
        warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
        warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)
        var_p4[var_p4 <= 0.] = utils.LARGE_VARIANCE

        var_r4[num_int, :, :, :] *= (segs_4[num_int, :, :, :] > 0)
        var_r4[var_r4 <= 0.] = utils.LARGE_VARIANCE

        # The sums of inverses of the variances are needed for later
        #   variance calculations.
        s_inv_var_p3[num_int, :, :] = (1. / var_p4[num_int, :, :, :]).sum(axis=0)
        var_p3[num_int, :, :] = 1. / s_inv_var_p3[num_int, :, :]
        s_inv_var_r3[num_int, :, :] = (1. / var_r4[num_int, :, :, :]).sum(axis=0)
        var_r3[num_int, :, :] = 1. / s_inv_var_r3[num_int, :, :]

        # Huge variances correspond to non-existing segments, so are reset to 0
        #  to nullify their contribution.
        var_p3[var_p3 > 0.1 * utils.LARGE_VARIANCE] = 0.
        var_p3[:, med_rates <= 0.] = 0.
        warnings.resetwarnings()

        var_p4[num_int, :, med_rates <= 0.] = 0.
        var_both4[num_int, :, :, :] = var_r4[num_int, :, :, :] + var_p4[num_int, :, :, :]
        inv_var_both4[num_int, :, :, :] = 1. / var_both4[num_int, :, :, :]

        # Want to retain values in the 4D arrays only for the segments that each
        #   pixel has, so will zero out values for the higher indices. Creating
        #   and manipulating intermediate arrays (views, such as var_p4_int
        #   will zero out the appropriate indices in var_p4 and var_r4.)
        # Extract the slice of 4D arrays for the current integration
        var_p4_int = var_p4[num_int, :, :, :]   # [ segment, y, x ]
        inv_var_both4_int = inv_var_both4[num_int, :, :, :]

        # Zero out non-existing segments
        var_p4_int *= (segs_4[num_int, :, :, :] > 0)
        inv_var_both4_int *= (segs_4[num_int, :, :, :] > 0)

        # reshape these arrays to simplify masking [ segment, 1D pixel ]
        var_p4_int2 = var_p4_int.reshape(
            (var_p4_int.shape[0], var_p4_int.shape[1] * var_p4_int.shape[2]))

        s_inv_var_both3[num_int, :, :] = (inv_var_both4[num_int, :, :, :]).sum(axis=0)

        # Suppress, then re-enable harmless arithmetic warnings
        warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
        warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)
        var_both3[num_int, :, :] = 1. / s_inv_var_both3[num_int, :, :]
        warnings.resetwarnings()

        del var_p4_int
        del var_p4_int2

    del gain_2d

    var_p4 *= (segs_4[:, :, :, :] > 0)  # Zero out non-existing segments
    var_r4 *= (segs_4[:, :, :, :] > 0)

    # Delete lots of arrays no longer needed
    if inv_var_both4_int is not None:
        del inv_var_both4_int

    if med_rates is not None:
        del med_rates

    if num_seg_per_int is not None:
        del num_seg_per_int

    if readnoise_2d is not None:
        del readnoise_2d

    if rn_sect is not None:
        del rn_sect

    if segs_4 is not None:
        del segs_4

    ramp_data.data = data
    ramp_data.err = err
    ramp_data.groupdq = groupdq
    ramp_data.pixeldq = inpixeldq

    return var_p3, var_r3, var_p4, var_r4, var_both4, var_both3, inv_var_both4, \
        s_inv_var_p3, s_inv_var_r3, s_inv_var_both3


def ramp_fit_overall(
        ramp_data, orig_cubeshape, orig_ngroups, buffsize, fit_slopes_ans,
        variances_ans, save_opt, tstart):
    """
    Computes the final/overall slope and variance values using the
    intermediate computations previously computed.  When computing
    integration slopes, if NaNs are computed, the corresponding DQ
    flag is set to DO_NOT_USE and the value is set to 0. per INS.


    Parameters
    ----------
    ramp_data : RampData
        Input data necessary for computing ramp fitting.

    orig_cubeshape : tuple
       Original shape cube of input dataset

    orig_ngroups : int
       Original number of groups

    buffsize : int
        Size of data section (buffer) in bytes

    fit_slopes_ans : tuple
        Return values from ramp_fit_slopes

    variances_ans : tuple
        Return values from ramp_fit_compute_variances

    save_opt : bool
        Calculate optional fitting results.

    tstart : float
        Start time.

    Return
    ------
    image_info : tuple
        The tuple of computed ramp fitting arrays.

    integ_info : tuple
        The tuple of computed integration fitting arrays.

    opt_info : tuple
        The tuple of computed optional results arrays for fitting.
    """
    # Get image data information
    data = ramp_data.data
    groupdq = ramp_data.groupdq

    # Get instrument and exposure data
    instrume = ramp_data.instrument_name
    groupgap = ramp_data.groupgap
    nframes = ramp_data.nframes
    dropframes1 = ramp_data.drop_frames1

    if dropframes1 is None:    # set to default if missing
        dropframes1 = 0
        log.debug('Missing keyword DRPFRMS1, so setting to default value of 0')

    # Get needed sizes and shapes
    n_int, ngroups, nrows, ncols = data.shape
    imshape = (nrows, ncols)

    # Unpack intermediate computations from preious steps
    max_seg, gdq_cube_shape, effintim, f_max_seg, dq_int, num_seg_per_int = fit_slopes_ans[:6]
    sat_0th_group_int, opt_res, pixeldq, inv_var, med_rates = fit_slopes_ans[6:]

    var_p3, var_r3, var_p4, var_r4, var_both4, var_both3 = variances_ans[:6]
    inv_var_both4, s_inv_var_p3, s_inv_var_r3, s_inv_var_both3 = variances_ans[6:]

    slope_by_var4 = opt_res.slope_seg.copy() / var_both4

    del var_both4

    s_slope_by_var3 = slope_by_var4.sum(axis=1)  # sum over segments (not integs)
    s_slope_by_var2 = s_slope_by_var3.sum(axis=0)  # sum over integrations
    s_inv_var_both2 = s_inv_var_both3.sum(axis=0)

    # Compute the 'dataset-averaged' slope
    # Suppress, then re-enable harmless arithmetic warnings
    warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
    warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)
    slope_dataset2 = s_slope_by_var2 / s_inv_var_both2
    warnings.resetwarnings()

    del s_slope_by_var2, s_slope_by_var3, slope_by_var4
    del s_inv_var_both2, s_inv_var_both3

    #  Replace nans in slope_dataset2 with 0 (for non-existing segments)
    slope_dataset2[np.isnan(slope_dataset2)] = 0.

    # Compute the integration-specific slope
    the_num = (opt_res.slope_seg * inv_var_both4).sum(axis=1)

    the_den = (inv_var_both4).sum(axis=1)

    # Suppress, then re-enable harmless arithmetic warnings
    warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
    warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)

    slope_int = the_num / the_den

    # Adjust DQ flags for NaNs.
    wh_nans = np.isnan(slope_int)
    dq_int[wh_nans] = np.bitwise_or(dq_int[wh_nans], ramp_data.flags_do_not_use)
    warnings.resetwarnings()

    del the_num, the_den, wh_nans

    # Clean up ramps that are SAT on their initial groups; set ramp parameters
    #   for variances and slope so they will not contribute

    var_p3, var_both3, slope_int, dq_int = utils.fix_sat_ramps(
        ramp_data, sat_0th_group_int, var_p3, var_both3, slope_int, dq_int)

    if sat_0th_group_int is not None:
        del sat_0th_group_int

    # Loop over data integrations to calculate integration-specific pedestal
    if save_opt:
        dq_slice = np.zeros((gdq_cube_shape[2], gdq_cube_shape[3]), dtype=np.uint32)

        for num_int in range(0, n_int):
            dq_slice = groupdq[num_int, 0, :, :]
            opt_res.ped_int[num_int, :, :] = \
                utils.calc_pedestal(ramp_data, num_int, slope_int, opt_res.firstf_int,
                                    dq_slice, nframes, groupgap, dropframes1)

        del dq_slice

    # Collect optional results for output
    if save_opt:
        gdq_cube = groupdq
        opt_res.shrink_crmag(n_int, gdq_cube, imshape, ngroups, ramp_data.flags_jump_det)
        del gdq_cube

        # Some contributions to these vars may be NaN as they are from ramps
        # having PIXELDQ=DO_NOT_USE
        var_p4[np.isnan(var_p4)] = 0.
        var_r4[np.isnan(var_r4)] = 0.

        # Truncate results at the maximum number of segments found
        opt_res.slope_seg = opt_res.slope_seg[:, :f_max_seg, :, :]
        opt_res.sigslope_seg = opt_res.sigslope_seg[:, :f_max_seg, :, :]
        opt_res.yint_seg = opt_res.yint_seg[:, :f_max_seg, :, :]
        opt_res.sigyint_seg = opt_res.sigyint_seg[:, :f_max_seg, :, :]
        opt_res.weights = (inv_var_both4[:, :f_max_seg, :, :])**2.
        opt_res.var_p_seg = var_p4[:, :f_max_seg, :, :]
        opt_res.var_r_seg = var_r4[:, :f_max_seg, :, :]

        opt_info = opt_res.output_optional(effintim)
    else:
        opt_info = None

    if inv_var_both4 is not None:
        del inv_var_both4

    if var_p4 is not None:
        del var_p4

    if var_r4 is not None:
        del var_r4

    if inv_var is not None:
        del inv_var

    if pixeldq is not None:
        del pixeldq

    # Output integration-specific results to separate file
    integ_info = utils.output_integ(
        ramp_data, slope_int, dq_int, effintim, var_p3, var_r3, var_both3)

    if opt_res is not None:
        del opt_res

    if slope_int is not None:
        del slope_int
    del var_p3
    del var_r3
    del var_both3

    # Divide slopes by total (summed over all integrations) effective
    #   integration time to give count rates.
    c_rates = slope_dataset2 / effintim

    # Compress all integration's dq arrays to create 2D PIXELDDQ array for
    #   primary output
    final_pixeldq = utils.dq_compress_final(dq_int, ramp_data)

    # For invalid slope calculations set to NaN.  Pixels flagged as SATURATED or
    # DO_NOT_USE have invalid data.
    invalid_data = ramp_data.flags_saturated | ramp_data.flags_do_not_use
    wh_invalid = np.where(np.bitwise_and(final_pixeldq, invalid_data))
    c_rates[wh_invalid] = np.nan

    if dq_int is not None:
        del dq_int

    tstop = time.time()

    utils.log_stats(c_rates)

    log.debug('Instrument: %s', instrume)
    log.debug('Number of pixels in 2D array: %d', nrows * ncols)
    log.debug('Shape of 2D image: (%d, %d)' % (imshape))
    log.debug('Shape of data cube: (%d, %d, %d)' % (orig_cubeshape))
    log.debug('Buffer size (bytes): %d', buffsize)
    log.debug('Number of rows per buffer: %d', nrows)
    log.info('Number of groups per integration: %d', orig_ngroups)
    log.info('Number of integrations: %d', n_int)
    log.debug('The execution time in seconds: %f', tstop - tstart)

    # Compute the 2D variances due to Poisson and read noise
    var_p2 = 1 / (s_inv_var_p3.sum(axis=0))
    var_r2 = 1 / (s_inv_var_r3.sum(axis=0))

    # Huge variances correspond to non-existing segments, so are reset to 0
    #  to nullify their contribution.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value.*", RuntimeWarning)
        var_p2[var_p2 > 0.1 * utils.LARGE_VARIANCE] = 0.
        var_r2[var_r2 > 0.1 * utils.LARGE_VARIANCE] = 0.

    # Some contributions to these vars may be NaN as they are from ramps
    # having PIXELDQ=DO_NOT_USE
    var_p2[np.isnan(var_p2)] = 0.
    var_p2[med_rates <= 0.0] = 0.
    var_r2[np.isnan(var_r2)] = 0.

    # Suppress, then re-enable, harmless arithmetic warning
    warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
    err_tot = np.sqrt(var_p2 + var_r2)
    warnings.resetwarnings()

    del s_inv_var_p3
    del s_inv_var_r3

    # Create new model for the primary output.
    data = c_rates.astype(np.float32)
    dq = final_pixeldq.astype(np.uint32)
    var_poisson = var_p2.astype(np.float32)
    var_rnoise = var_r2.astype(np.float32)
    err = err_tot.astype(np.float32)
    image_info = (data, dq, var_poisson, var_rnoise, err)

    return image_info, integ_info, opt_info


def calc_power(snr):
    """
    Using the given SNR, calculate the weighting exponent, which is from
    `Fixsen, D.J., Offenberg, J.D., Hanisch, R.J., Mather, J.C, Nieto,
    Santisteban, M.A., Sengupta, R., & Stockman, H.S., 2000, PASP, 112, 1350`.

    Parameters
    ----------
    snr : ndarray
        signal-to-noise for the ramp segments, 1-D float

    Returns
    -------
    pow_wt : ndarray
        weighting exponent, 1-D float
    """
    pow_wt = snr.copy() * 0.0
    pow_wt[np.where(snr > 5.)] = 0.4
    pow_wt[np.where(snr > 10.)] = 1.0
    pow_wt[np.where(snr > 20.)] = 3.0
    pow_wt[np.where(snr > 50.)] = 6.0
    pow_wt[np.where(snr > 100.)] = 10.0

    return pow_wt.ravel()


def interpolate_power(snr):
    """
    Using the given SNR, interpolate the weighting exponent, which is from
    `Fixsen, D.J., Offenberg, J.D., Hanisch, R.J., Mather, J.C, Nieto,
    Santisteban, M.A., Sengupta, R., & Stockman, H.S., 2000, PASP, 112, 1350`.

    Parameters
    ----------
    snr : ndarray
        signal-to-noise for the ramp segments, 1-D float

    Returns
    -------
    pow_wt : ndarray
        weighting exponent, 1-D float
    """
    pow_wt = snr.copy() * 0.0
    pow_wt[np.where(snr > 5.)] = ((snr[snr > 5] - 5) / (10 - 5)) * 0.6 + 0.4
    pow_wt[np.where(snr > 10.)] = ((snr[snr > 10] - 10) / (20 - 10)) * 2.0 + 1.0
    pow_wt[np.where(snr > 20.)] = ((snr[snr > 20] - 20)) / (50 - 20) * 3.0 + 3.0
    pow_wt[np.where(snr > 50.)] = ((snr[snr > 50] - 50)) / (100 - 50) * 4.0 + 6.0
    pow_wt[np.where(snr > 100.)] = 10.0

    return pow_wt.ravel()


def calc_slope(data_sect, gdq_sect, frame_time, opt_res, save_opt, rn_sect,
               gain_sect, i_max_seg, ngroups, weighting, f_max_seg, ramp_data):
    """
    Compute the slope of each segment for each pixel in the data cube section
    for the current integration. Each segment has its slope fit in fit_lines();
    that slope and other quantities from the fit are added to the 'optional
    result' object by append_arr() from the appropriate 'CASE' (type of segment)
    in fit_next_segment().

    Parameters
    ----------
    data_sect : ndarray
        section of input data cube array, 3-D float

    gdq_sect : ndarray
        section of GROUPDQ data quality array, 3-D int

    frame_time : float
        integration time

    opt_res : OptRes object
        Contains all quantities derived from fitting all segments in all
        pixels in all integrations, which will eventually be used to compute
        per-integration and per-exposure quantities for all pixels. It's
        also used to populate the optional product, when requested.

    save_opt : bool
       save optional fitting results

    rn_sect : ndarray
        read noise values for all pixels in data section

    gain_sect : ndarray
        gain values for all pixels in data section

    i_max_seg : int
        used for size of initial allocation of arrays for optional results;
        maximum possible number of segments within the ramp, based on the
        number of CR flags

    ngroups : int
        number of groups per integration

    weighting : str
        'optimal' specifies that optimal weighting should be used; currently
        the only weighting supported.

    f_max_seg : int
        actual maximum number of segments within a ramp, based on the fitting
        of all ramps; later used when truncating arrays before output.

    remp_data : RampClass
        The ramp data and metadata, specifically the relevant DQ flags.

    Returns
    -------
    gdq_sect : ndarray
        data quality flags for pixels in section, 3-D int

    inv_var : ndarray
        values of 1/variance for good pixels, 1-D float

    opt_res : OptRes object
        contains all quantities related to fitting for use in computing final
        slopes, variances, etc. and is used to populate the optional output

    f_max_seg : int
        actual maximum number of segments within a ramp, updated here based on
        fitting ramps in the current data section; later used when truncating
        arrays before output.

    num_seg : ndarray
        numbers of segments for good pixels, 1-D int
    """
    ngroups, nrows, ncols = data_sect.shape
    npix = nrows * ncols  # number of pixels in section of 2D array

    all_pix = np.arange(npix)
    arange_ngroups_col = np.arange(ngroups)[:, np.newaxis]
    start = np.zeros(npix, dtype=np.int32)  # lowest channel in fit

    # Highest channel in fit initialized to last read
    end = np.zeros(npix, dtype=np.int32) + (ngroups - 1)

    pixel_done = (end < 0)  # False until processing is done

    inv_var = np.zeros(npix, dtype=np.float32)  # inverse of fit variance
    num_seg = np.zeros(npix, dtype=np.int32)  # number of segments per pixel

    # End stack array - endpoints for each pixel
    # initialize with ngroups for each pixel; set 1st channel to 0
    end_st = np.zeros((ngroups + 1, npix), dtype=np.int32)
    end_st[0, :] = ngroups - 1

    # end_heads is initially a tuple populated with every pixel that is
    # either saturated or contains a cosmic ray based on the input DQ
    # array, so is sized to accomodate the maximum possible number of
    # pixels flagged. It is later compressed to be an array denoting
    # the number of endpoints per pixel.
    end_heads = np.ones(npix * ngroups, dtype=np.int32)

    # Create nominal 2D ERR array, which is 1st slice of
    #    avged_data_cube * readtime
    err_2d_array = data_sect[0, :, :] * frame_time

    # Suppress, then re-enable, harmless arithmetic warnings
    '''
    warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
    warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)
    '''
    err_2d_array[err_2d_array < 0] = 0
    warnings.resetwarnings()

    # Frames >= start and <= end will be masked. However, the first channel
    #   to be included in fit will be the read in which a cosmic ray has
    #   been flagged
    mask_2d = ((arange_ngroups_col >= start[np.newaxis, :]) &
               (arange_ngroups_col <= end[np.newaxis, :]))

    end = 0  # array no longer needed

    # Section of GROUPDQ dq section, excluding bad dq values in mask
    gdq_sect_r = np.reshape(gdq_sect, (ngroups, npix))
    mask_2d[gdq_sect_r != 0] = False  # saturated or CR-affected
    mask_2d_init = mask_2d.copy()  # initial flags for entire ramp
    start = np.argmax(mask_2d, 0)  # start with the first True value
    # Reset the initial False groups to be True so that the first False is now either a jump or sat
    # Because start was set to be the first True, the initial False values will not be included
    for pixel in range(npix):
        mask_2d[:start[pixel], pixel] = True

    wh_f = np.where(np.logical_not(mask_2d))

    these_p = wh_f[1]  # coordinates of pixels flagged as False
    these_r = wh_f[0]  # reads of pixels flagged as False

    del wh_f

    # Populate end_st to contain the set of end points for each pixel.
    # Populate end_heads to initially include every pixel that is either
    # saturated or contains a cosmic ray. Skips the duplicated final group
    # for saturated pixels. Saturated pixels resulting in a contiguous set
    # of intervals of length 1 will later be flagged as too short
    # to fit well.
    for ii, val in enumerate(these_p):
        if these_r[ii] != (ngroups - 1):
            end_st[end_heads[these_p[ii]], these_p[ii]] = these_r[ii]
            end_heads[these_p[ii]] += 1

    # Sort and reverse array to handle the order that saturated pixels
    # were added
    end_st.sort(axis=0)
    end_st = end_st[::-1]

    # Reformat to designate the number of endpoints per pixel; compress
    # to specify number of groups per pixel
    end_heads = (end_st > 0).sum(axis=0)

    # Create object to hold optional results
    opt_res.init_2d(npix, i_max_seg, save_opt)

    # LS fit until 'ngroups' iterations or all pixels in
    #    section have been processed
    for iter_num in range(ngroups):
        if pixel_done.all():
            break

        # frames >= start and <= end_st will be included in fit
        mask_2d = \
            ((arange_ngroups_col >= start)
             & (arange_ngroups_col < (end_st[end_heads[all_pix] - 1, all_pix] + 1)))

        mask_2d[gdq_sect_r != 0] = False  # RE-exclude bad group dq values

        # A segment could be created if a cosmic ray cause a JUMP_DET flag to be
        #   set.  In the above line that group would be excluded from the
        #   current segment.  If a segment is created only due to a group
        #   flagged as JUMP_DET it will be the group just prior to the 0th
        #   group in the current segement.  We want to include it as part of
        #   the current segment, but exclude all other groups with any other
        #   flag.

        # Find CRs in the ramp.
        jump_det = ramp_data.flags_jump_det
        mask_2d_jump = mask_2d.copy()
        wh_jump = np.where(gdq_sect_r == jump_det)
        mask_2d_jump[wh_jump] = True
        del wh_jump

        # Add back possible CRs at the beginning of a ramp that were excluded
        # above.
        wh_mask_2d = np.where(mask_2d)
        mask_2d[np.maximum(wh_mask_2d[0] - 1, 0), wh_mask_2d[1]] = True
        del wh_mask_2d

        mask_2d = mask_2d & mask_2d_jump

        # for all pixels, update arrays, summing slope and variance
        f_max_seg, num_seg = fit_next_segment(
            start, end_st, end_heads, pixel_done, data_sect, mask_2d, mask_2d_init,
            inv_var, num_seg, opt_res, save_opt, rn_sect, gain_sect, ngroups, weighting,
            f_max_seg, gdq_sect_r, ramp_data)

        if f_max_seg is None:
            f_max_seg = 1

    arange_ngroups_col = 0
    all_pix = 0

    return gdq_sect, inv_var, opt_res, f_max_seg, num_seg


def fit_next_segment(start, end_st, end_heads, pixel_done, data_sect, mask_2d,
                     mask_2d_init, inv_var, num_seg, opt_res, save_opt, rn_sect,
                     gain_sect, ngroups, weighting, f_max_seg, gdq_sect_r, ramp_data):
    """
    Call routine to LS fit masked data for a single segment for all pixels in
    data section. Then categorize each pixel's fitting interval based on
    interval length, and whether the interval is at the end of the array.
    Update the start array, the end stack array, the end_heads array which
    contains the number of endpoints. For pixels in which the fitting intervals
    are long enough, the resulting slope and variance are added to the
    appropriate stack arrays.  The first channel to fit in a segment is either
    the first group in the ramp, or a group in which a cosmic ray has been
    flagged.

    Parameters
    ----------
    start : ndarray
        lowest channel in fit, 1-D  int

    end_st : ndarray
        stack array of endpoints, 2-D  int

    end_heads : ndarray
        number of endpoints for each pixel, 1-D  int

    pixel_done : ndarray
        whether each pixel's calculations are completed, 1-D  bool

    data_sect : ndarray
        data cube section, 3-D float

    mask_2d : ndarray
        delineates which channels to fit for each pixel, 2-D bool

    mask_2d_init : ndarray
        copy of intial mask_2d, 2-D bool

    inv_var : ndarray
        values of 1/variance for good pixels, 1-D float

    num_seg : ndarray
        numbers of segments for good pixels, 1-D int

    opt_res : OptRes object
        all fitting quantities, used to compute final results
        and to populate optional output product

    save_opt : bool
       save optional fitting results

    rn_sect : ndarray
        read noise values for all pixels in data section

    gain_sect : ndarray
        gain values for all pixels in data section

    ngroups : int
        number of groups per integration

    weighting : str
        'optimal' specifies that optimal weighting should be used; currently
        the only weighting supported.

    f_max_seg : int
        actual maximum number of segments within a ramp, updated here based on
        fitting ramps in the current data section; later used when truncating
        arrays before output.

    gdq_sect_r : ndarray
        The section data presented as a 2-D array with dimnsions (ngroups, npix)

    ramp_data : RampData
        The ramp data needed for processing, specifically flag values.

    Returns
    -------
    f_max_seg : int
        actual maximum number of segments within a ramp, updated here based on
        fitting ramps in the current data section; later used when truncating
        arrays before output.

    num_seg : ndarray
        numbers of segments for good pixels, 1-D int
    """
    ngroups, nrows, ncols = data_sect.shape
    all_pix = np.arange(nrows * ncols)

    ramp_mask_sum = mask_2d_init.sum(axis=0)

    # Compute fit quantities for the next segment of all pixels
    # Each returned array below is 1D, for all npix pixels for current segment
    slope, intercept, variance, sig_intercept, sig_slope = fit_lines(
        data_sect, mask_2d, rn_sect, gain_sect, ngroups, weighting, gdq_sect_r, ramp_data)

    end_locs = end_st[end_heads[all_pix] - 1, all_pix]

    # Set the fitting interval length; for a segment having >1 groups, this is
    #   the number of groups-1
    l_interval = end_locs - start

    wh_done = (start == -1)  # done pixels
    l_interval[wh_done] = 0  # set interval lengths for done pixels to 0

    # Create array to set when each good pixel is classified for the current
    #   semiramp (to enable unclassified pixels to have their arrays updated)
    got_case = np.zeros((ncols * nrows), dtype=bool)

    # Special case fit with NGROUPS being 1 or 2.
    if ngroups == 1 or ngroups == 2:
        return fit_short_ngroups(
            ngroups, start, end_st, end_heads, pixel_done, all_pix,
            inv_var, num_seg, slope, intercept, variance, sig_intercept, sig_slope,
            opt_res, save_opt, mask_2d_init, ramp_mask_sum)

    # CASE: Long enough (semiramp has >2 groups), at end of ramp
    wh_check = np.where((l_interval > 1) & (end_locs == ngroups - 1) & (~pixel_done))
    if len(wh_check[0]) > 0:
        f_max_seg = fit_next_segment_long_end_of_ramp(
            wh_check, start, end_st, end_heads, pixel_done, got_case, f_max_seg,
            inv_var, num_seg, slope, intercept, variance, sig_intercept, sig_slope,
            opt_res, save_opt)

    # CASE: Long enough (semiramp has >2 groups ), not at array end (meaning
    #  final group for this semiramp is not final group of the whole ramp)
    wh_check = np.where((l_interval > 2) & (end_locs != ngroups - 1) & ~pixel_done)
    if len(wh_check[0]) > 0:
        f_max_seg = fit_next_segment_long_not_end_of_ramp(
            wh_check, start, end_st, end_heads, pixel_done, got_case, f_max_seg,
            inv_var, num_seg, slope, intercept, variance, sig_intercept, sig_slope,
            opt_res, save_opt, mask_2d_init, end_locs, ngroups)

    # CASE: interval too short to fit normally (only 2 good groups)
    #    At end of array, NGROUPS>1, but exclude NGROUPS==2 datasets
    #    as they are covered in `fit_short_ngroups`.
    wh_check = np.where((l_interval == 1) & (end_locs == ngroups - 1)
                        & (ngroups > 2) & (~pixel_done))

    if len(wh_check[0]) > 0:
        f_max_seg = fit_next_segment_short_seg_at_end(
            wh_check, start, end_st, end_heads, pixel_done, got_case, f_max_seg,
            inv_var, num_seg, slope, intercept, variance, sig_intercept, sig_slope,
            opt_res, save_opt, mask_2d_init)

    # CASE: full-length ramp has 2 good groups not at array end
    wh_check = np.where((l_interval == 2) & (ngroups > 2)
                        & (end_locs != ngroups - 1) & ~pixel_done)

    if len(wh_check[0]) > 0:
        f_max_seg = fit_next_segment_short_seg_not_at_end(
            wh_check, start, end_st, end_heads, pixel_done, got_case, f_max_seg,
            inv_var, num_seg, slope, intercept, variance, sig_intercept, sig_slope,
            opt_res, save_opt, mask_2d_init, end_locs, ngroups)

    # CASE: full-length ramp has a good group on 0th group of the entire ramp,
    #    and no later good groups. Will use single good group data as the slope.
    wh_check = np.where(
        mask_2d_init[0, :] & ~mask_2d_init[1, :] & (ramp_mask_sum == 1) & ~pixel_done)

    if len(wh_check[0]) > 0:
        f_max_seg = fit_next_segment_only_good_0th_group(
            wh_check, start, end_st, end_heads, pixel_done, got_case, f_max_seg,
            inv_var, num_seg, slope, intercept, variance, sig_intercept, sig_slope,
            opt_res, save_opt, mask_2d_init)

    # CASE: the segment has a good 0th group and a bad 1st group.
    wh_check = np.where(mask_2d_init[0, :] & ~mask_2d_init[1, :] & ~pixel_done
                        & (end_locs == 1) & (start == 0))

    if len(wh_check[0]) > 0:
        fit_next_segment_good_0th_bad_1st(
            wh_check, start, end_st, end_heads, got_case, ngroups)

    # CASE OTHER: all other types of segments not covered earlier. No segments
    #   handled here have adequate data, but the stack arrays are updated.
    wh_check = np.asarray(np.where(~pixel_done & ~got_case))
    if len(wh_check[0]) > 0:
        fit_next_segment_all_other(wh_check, start, end_st, end_heads, ngroups)

    return f_max_seg, num_seg


def fit_next_segment_all_other(wh_check, start, end_st, end_heads, ngroups):
    """
    Catch all other types of segments not covered earlier. No segments
    handled here have adequate data, but the stack arrays are updated.
        - increment start array
        - remove current end from end stack
        - decrement number of ends

    Parameters
    ----------
    wh_check : ndarray
        pixels for current segment processing and updating, 1-D

    start : ndarray
        lowest channel in fit, 1-D int

    end_st : ndarray
        stack array of endpoints, 2-D int

    end_heads : ndarray
        number of endpoints for each pixel, 1-D int

    ngroups : int
        number of groups in exposure
    """
    these_pix = wh_check[0]
    start[these_pix] += 1
    start[start > ngroups - 1] = ngroups - 1  # to keep at max level
    end_st[end_heads[these_pix] - 1, these_pix] = 0
    end_heads[these_pix] -= 1
    end_heads[end_heads < 0.] = 0.


def fit_next_segment_good_0th_bad_1st(
        wh_check, start, end_st, end_heads, got_case, ngroups):
    """
    The segment has a good 0th group and a bad 1st group. For the
    data from the 0th good group of this segment to possibly be used as a
    slope, that group must necessarily be the 0th group of the entire ramp.
    It is possible to have a single 'good' group segment after the 0th group
    of the ramp; in that case the 0th group and the 1st group would both have
    to be CRs, and the data of the 0th group would not be included as a slope.
    For a good 0th group in a ramp followed by a bad 1st group there must be
    good groups later in the segment because if there were not, the segment
    would be done in `fit_next_segment_only_good_0th_group`. In this situation,
    since here are later good groups in the segment, those later good groups
    will be used in the slope computation, and the 0th good group will not be.
    As a result, for all instances of these types of segments, the data in the
    initial good group will not be used in the slope calculation, but the
    arrays for the indices for the ramp (end_st, etc) are appropriately
    adjusted.
        - increment start array
        - remove current end from end stack
        - decrement number of ends

    Parameters
    ----------
    wh_check : ndarray
        pixels for current segment processing and updating, 1-D

    start : ndarray
        lowest channel in fit, 1-D int

    end_st : ndarray
        stack array of endpoints, 2-D int

    end_heads : ndarray
        number of endpoints for each pixel, 1-D int

    got_case : ndarray
        classification of pixel for current semiramp, 1-D

    ngroups : int
        number of groups in exposure
    """
    these_pix = wh_check[0]
    got_case[these_pix] = True
    start[these_pix] += 1
    start[start > ngroups - 1] = ngroups - 1  # to keep at max level
    end_st[end_heads[these_pix] - 1, these_pix] = 0
    end_heads[these_pix] -= 1
    end_heads[end_heads < 0.] = 0.


def fit_next_segment_only_good_0th_group(
        wh_check, start, end_st, end_heads, pixel_done, got_case, f_max_seg,
        inv_var, num_seg, slope, intercept, variance, sig_intercept, sig_slope,
        opt_res, save_opt, mask_2d_init):
    """
    Full-length ramp has a good group on 0th group of the entire ramp,
    and no later good groups. Will use single good group data as the slope.
        - set start to -1 to designate all fitting done
        - remove current end from end stack
        - set number of end to 0
        - add slopes and variances to running sums
        - set pixel_done to True to designate all fitting done

    Parameters
    ----------
    wh_check : ndarray
        pixels for current segment processing and updating, 1-D

    start : ndarray
        lowest channel in fit, 1-D int

    end_st : ndarray
        stack array of endpoints, 2-D int

    end_heads : ndarray
        number of endpoints for each pixel, 1-D int

    pixel_done : ndarray
        whether each pixel's calculations are completed, 1-D bool

    got_case : ndarray
        classification of pixel for current semiramp, 1-D

    f_max_seg : int
        actual maximum number of segments within a ramp, updated here based on
        fitting ramps in the current data section; later used when truncating
        arrays before output.

    inv_var : ndarray
        values of 1/variance for good pixels, 1-D float

    num_seg : ndarray
        numbers of segments for good pixels, 1-D int

    slope : ndarray
       weighted slope for current iteration's pixels for data section, 1-D
       float

    intercept : ndarray
       y-intercepts from fit for data section, 1-D float

    variance : ndarray
       variance of residuals for fit for data section, 1-D float

    sig_intercept : ndarray
       sigma of y-intercepts from fit for data section, 1-D float

    sig_slope : ndarray
       sigma of slopes from fit for data section (for a single segment), 1-D
       float

    opt_res : OptRes object
        all fitting quantities, used to compute final results
        and to populate optional output product

    save_opt : bool
       save optional fitting results

    mask_2d_init : ndarray
        copy of intial mask_2d, 2-D bool

    Returns
    -------
    f_max_seg : int
        actual maximum number of segments within a ramp, updated here based on
        fitting ramps in the current data section; later used when truncating
        arrays before output.
    """
    these_pix = wh_check[0]
    got_case[these_pix] = True

    start[these_pix] = -1
    end_st[end_heads[these_pix] - 1, these_pix] = 0
    end_heads[these_pix] = 0
    pixel_done[these_pix] = True  # all processing for pixel is completed
    inv_var[these_pix] += 1.0 / variance[these_pix]

    # Append results to arrays
    opt_res.append_arr(num_seg, these_pix, intercept, slope,
                       sig_intercept, sig_slope, inv_var, save_opt)

    num_seg[these_pix] += 1
    f_max_seg = max(f_max_seg, num_seg.max())

    return f_max_seg


def fit_next_segment_short_seg_not_at_end(
        wh_check, start, end_st, end_heads, pixel_done, got_case, f_max_seg,
        inv_var, num_seg, slope, intercept, variance, sig_intercept, sig_slope,
        opt_res, save_opt, mask_2d_init, end_locs, ngroups):
    """
    Special case
    Full-length ramp has 2 good groups not at array end
        - use the 2 good reads to get the slope
        - set start to -1 to designate all fitting done
        - remove current end from end stack
        - set number of end to 0
        - add slopes and variances to running sums
        - set pixel_done to True to designate all fitting done
    For segments of this type, the final good group in the segment is
    followed by a group that is flagged as a CR and/or SAT and is not the
    final group in the ramp, and the variable `l_interval` used below is
    equal to 2, which is the number of the segment's groups.

    Parameters
    ----------
    wh_check : ndarray
        pixels for current segment processing and updating, 1-D

    start : ndarray
        lowest channel in fit, 1-D int

    end_st : ndarray
        stack array of endpoints, 2-D int

    end_heads : ndarray
        number of endpoints for each pixel, 1-D int

    pixel_done : ndarray
        whether each pixel's calculations are completed, 1-D bool

    got_case : ndarray
        classification of pixel for current semiramp, 1-D

    f_max_seg : int
        actual maximum number of segments within a ramp, updated here based on
        fitting ramps in the current data section; later used when truncating
        arrays before output.

    inv_var : ndarray
        values of 1/variance for good pixels, 1-D float

    num_seg : ndarray
        numbers of segments for good pixels, 1-D int

    slope : ndarray
       weighted slope for current iteration's pixels for data section, 1-D
       float

    intercept : ndarray
       y-intercepts from fit for data section, 1-D float

    variance : ndarray
       variance of residuals for fit for data section, 1-D float

    sig_intercept : ndarray
       sigma of y-intercepts from fit for data section, 1-D float

    sig_slope : ndarray
       sigma of slopes from fit for data section (for a single segment), 1-D
       float

    opt_res : OptRes object
        all fitting quantities, used to compute final results
        and to populate optional output product

    save_opt : bool
       save optional fitting results

    mask_2d_init : ndarray
        copy of intial mask_2d, 2-D bool

    end_locs : ndarray
        end locations, 1-D

    ngroups : int
        number of groups in exposure

    Returns
    -------
    f_max_seg : int
        actual maximum number of segments within a ramp, updated here based on
        fitting ramps in the current data section; later used when truncating
        arrays before output.
    """
    # Copy mask, as will modify when calculating the number of later good groups
    c_mask_2d_init = mask_2d_init.copy()

    these_pix = wh_check[0]
    got_case[these_pix] = True

    # Suppress, then re-enable, harmless arithmetic warnings
    warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
    warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)
    inv_var[these_pix] += 1.0 / variance[these_pix]
    warnings.resetwarnings()

    # create array: 0...ngroups-1 in a column for each pixel
    arr_ind_all = np.array(
        [np.arange(ngroups), ] * c_mask_2d_init.shape[1]).transpose()
    wh_c_start_all = np.zeros(mask_2d_init.shape[1], dtype=np.uint8)
    wh_c_start_all[these_pix] = start[these_pix]

    # set to False all groups before start group
    c_mask_2d_init[arr_ind_all < wh_c_start_all] = 0
    tot_good_groups = c_mask_2d_init.sum(axis=0)

    # Select pixels having at least 2 later good groups (these later good
    #   groups are a segment whose slope will be calculated)
    wh_more = np.where(tot_good_groups[these_pix] > 1)
    pix_more = these_pix[wh_more]
    start[pix_more] = end_locs[pix_more]
    end_st[end_heads[pix_more] - 1, pix_more] = 0
    end_heads[pix_more] -= 1

    # Select pixels having less than 2 later good groups (these later good
    #   groups will not be used)
    wh_only = np.where(tot_good_groups[these_pix] <= 1)
    pix_only = these_pix[wh_only]
    start[pix_only] = -1
    end_st[end_heads[pix_only] - 1, pix_only] = 0
    end_heads[pix_only] = 0
    pixel_done[pix_only] = True  # all processing for pixel is completed
    end_heads[(end_heads < 0.)] = 0.

    # Append results to arrays
    opt_res.append_arr(num_seg, these_pix, intercept, slope,
                       sig_intercept, sig_slope, inv_var, save_opt)

    num_seg[these_pix] += 1
    f_max_seg = max(f_max_seg, num_seg.max())

    return f_max_seg


def fit_next_segment_short_seg_at_end(
        wh_check, start, end_st, end_heads, pixel_done, got_case, f_max_seg,
        inv_var, num_seg, slope, intercept, variance, sig_intercept, sig_slope,
        opt_res, save_opt, mask_2d_init):
    """
    Interval too short to fit normally (only 2 good groups)
    At end of array, NGROUPS>1, but exclude NGROUPS==2 datasets
    as they are covered in `fit_short_groups`.
        - set start to -1 to designate all fitting done
        - remove current end from end stack
        - set number of ends to 0
        - add slopes and variances to running sums
        - set pixel_done to True to designate all fitting done
    For segments of this type, the final good group is the final group in the
    ramp, and the variable `l_interval` used below = 1, and the number of
    groups in the segment = 2

    Parameters
    ----------
    wh_check : ndarray
        pixels for current segment processing and updating, 1-D

    start : ndarray
        lowest channel in fit, 1-D int

    end_st : ndarray
        stack array of endpoints, 2-D int

    end_heads : ndarray
        number of endpoints for each pixel, 1-D int

    pixel_done : ndarray
        whether each pixel's calculations are completed, 1-D bool

    got_case : ndarray
        classification of pixel for current semiramp, 1-D

    f_max_seg : int
        actual maximum number of segments within a ramp, updated here based on
        fitting ramps in the current data section; later used when truncating
        arrays before output.

    inv_var : ndarray
        values of 1/variance for good pixels, 1-D float

    num_seg : ndarray
        numbers of segments for good pixels, 1-D int

    slope : ndarray
       weighted slope for current iteration's pixels for data section, 1-D
       float

    intercept : ndarray
       y-intercepts from fit for data section, 1-D float

    variance : ndarray
       variance of residuals for fit for data section, 1-D float

    sig_intercept : ndarray
       sigma of y-intercepts from fit for data section, 1-D float

    sig_slope : ndarray
       sigma of slopes from fit for data section (for a single segment), 1-D
       float

    opt_res : OptRes object
        all fitting quantities, used to compute final results
        and to populate optional output product

    save_opt : bool
       save optional fitting results

    mask_2d_init : ndarray
        copy of intial mask_2d, 2-D bool

    Returns
    -------
    f_max_seg : int
        actual maximum number of segments within a ramp, updated here based on
        fitting ramps in the current data section; later used when truncating
        arrays before output.
    """
    # Require that pixels to be processed here have at least 1 good group out
    #   of the final 2 groups (these ramps have 2 groups and are at the end of
    #   the array).
    wh_list = []

    num_wh = len(wh_check[0])
    for ii in range(num_wh):  # locate pixels with at least 1 good group
        this_pix = wh_check[0][ii]
        sum_final_2 = mask_2d_init[start[this_pix]:, this_pix].sum()

        if sum_final_2 > 0:
            wh_list.append(wh_check[0][ii])  # add to list to be fit

    if len(wh_list) > 0:
        these_pix = np.asarray(wh_list)
        got_case[these_pix] = True

        start[these_pix] = -1
        end_st[end_heads[these_pix] - 1, these_pix] = 0
        end_heads[these_pix] = 0
        pixel_done[these_pix] = True

        g_pix = these_pix[variance[these_pix] > 0.]  # good pixels

        if len(g_pix) > 0:
            inv_var[g_pix] += 1.0 / variance[g_pix]

            # Append results to arrays
            opt_res.append_arr(num_seg, g_pix, intercept, slope,
                               sig_intercept, sig_slope, inv_var, save_opt)

            num_seg[g_pix] += 1
            f_max_seg = max(f_max_seg, num_seg.max())

    return f_max_seg


def fit_next_segment_long_not_end_of_ramp(
        wh_check, start, end_st, end_heads, pixel_done, got_case, f_max_seg,
        inv_var, num_seg, slope, intercept, variance, sig_intercept, sig_slope,
        opt_res, save_opt, mask_2d_init, end_locs, ngroups):
    """
    Special case fitting long segment at the end of ramp.
    Long enough (semiramp has >2 groups ), not at array end (meaning
    final group for this semiramp is not final group of the whole ramp)
        - remove current end from end stack
        - decrement number of ends
        - add slopes and variances to running sums
    For segments of this type, the final good group in the segment is a CR
    and/or SAT and is not the final group in the ramp, and the variable
    `l_interval` used below is equal to the number of the segment's groups.

    Parameters
    ----------
    wh_check : ndarray
        pixels for current segment processing and updating, 1-D

    start : ndarray
        lowest channel in fit, 1-D int

    end_st : ndarray
        stack array of endpoints, 2-D int

    end_heads : ndarray
        number of endpoints for each pixel, 1-D int

    pixel_done : ndarray
        whether each pixel's calculations are completed, 1-D bool

    got_case : ndarray
        classification of pixel for current semiramp, 1-D

    f_max_seg : int
        actual maximum number of segments within a ramp, updated here based on
        fitting ramps in the current data section; later used when truncating
        arrays before output.

    inv_var : ndarray
        values of 1/variance for good pixels, 1-D float

    num_seg : ndarray
        numbers of segments for good pixels, 1-D int

    slope : ndarray
       weighted slope for current iteration's pixels for data section, 1-D
       float

    intercept : ndarray
       y-intercepts from fit for data section, 1-D float

    variance : ndarray
       variance of residuals for fit for data section, 1-D float

    sig_intercept : ndarray
       sigma of y-intercepts from fit for data section, 1-D float

    sig_slope : ndarray
       sigma of slopes from fit for data section (for a single segment), 1-D
       float

    opt_res : OptRes object
        all fitting quantities, used to compute final results
        and to populate optional output product

    save_opt : bool
       save optional fitting results

    end_locs : ndarray
        end locations, 1-D

    mask_2d_init : ndarray
        copy of intial mask_2d, 2-D bool

    ngroups : int
        number of groups in exposure

    Returns
    -------
    f_max_seg : int
        actual maximum number of segments within a ramp, updated here based on
        fitting ramps in the current data section; later used when truncating
        arrays before output.
    """
    these_pix = wh_check[0]
    got_case[these_pix] = True

    start[these_pix] = end_locs[these_pix]
    end_st[end_heads[these_pix] - 1, these_pix] = 0
    end_heads[these_pix] -= 1
    end_heads[end_heads < 0.] = 0.

    g_pix = these_pix[variance[these_pix] > 0.]  # good pixels

    if len(g_pix) > 0:
        inv_var[g_pix] += 1.0 / variance[g_pix]

        # Append results to arrays
        opt_res.append_arr(num_seg, g_pix, intercept, slope, sig_intercept,
                           sig_slope, inv_var, save_opt)

        num_seg[g_pix] += 1
        f_max_seg = max(f_max_seg, num_seg.max())

        # If there are pixels with no later good groups, update stack
        #   arrays accordingly
        c_mask_2d_init = mask_2d_init.copy()

        # create array: 0...ngroups-1 in a column for each pixel
        arr_ind_all = np.array(
            [np.arange(ngroups), ] * c_mask_2d_init.shape[1]).transpose()

        wh_c_start_all = np.zeros(c_mask_2d_init.shape[1], dtype=np.uint8)
        wh_c_start_all[g_pix] = start[g_pix]

        # set to False all groups before start group
        c_mask_2d_init[arr_ind_all < wh_c_start_all] = False

        # select pixels having all groups False from start to ramp end
        wh_rest_false = np.where(c_mask_2d_init.sum(axis=0) == 0)
        if len(wh_rest_false[0]) > 0:
            pix_rest_false = wh_rest_false[0]
            start[pix_rest_false] = -1
            end_st[end_heads[pix_rest_false] - 1, pix_rest_false] = 0
            end_heads[pix_rest_false] = 0
            pixel_done[pix_rest_false] = True  # all processing is complete

    return f_max_seg


def fit_next_segment_long_end_of_ramp(
        wh_check, start, end_st, end_heads, pixel_done, got_case, f_max_seg,
        inv_var, num_seg, slope, intercept, variance, sig_intercept, sig_slope,
        opt_res, save_opt):
    """
    Long enough (semiramp has >2 groups), at end of ramp
        - set start to -1 to designate all fitting done
        - remove current end from end stack
        - set number of ends to 0
        - add slopes and variances to running sums
    For segments of this type, the final good group is the final group in the
    ramp, and the variable `l_interval` used below is equal to the number of
    the segment's groups minus 1.

    Parameters
    ----------
    wh_check : ndarray
        pixels for current segment processing and updating, 1-D

    start : ndarray
        lowest channel in fit, 1-D int

    end_st : ndarray
        stack array of endpoints, 2-D int

    end_heads : ndarray
        number of endpoints for each pixel, 1-D int

    pixel_done : ndarray
        whether each pixel's calculations are completed, 1-D bool

    got_case : ndarray
        classification of pixel for current semiramp, 1-D

    f_max_seg : int
        actual maximum number of segments within a ramp, updated here based on
        fitting ramps in the current data section; later used when truncating
        arrays before output.

    inv_var : ndarray
        values of 1/variance for good pixels, 1-D float

    num_seg : ndarray
        numbers of segments for good pixels, 1-D int

    slope : ndarray
       weighted slope for current iteration's pixels for data section, 1-D
       float

    intercept : ndarray
       y-intercepts from fit for data section, 1-D float

    variance : ndarray
       variance of residuals for fit for data section, 1-D float

    sig_intercept : ndarray
       sigma of y-intercepts from fit for data section, 1-D float

    sig_slope : ndarray
       sigma of slopes from fit for data section (for a single segment), 1-D
       float

    opt_res : OptRes object
        all fitting quantities, used to compute final results
        and to populate optional output product

    save_opt : bool
       save optional fitting results

    Returns
    -------
    f_max_seg : int
        actual maximum number of segments within a ramp, updated here based on
        fitting ramps in the current data section; later used when truncating
        arrays before output.
    """
    these_pix = wh_check[0]
    start[these_pix] = -1   # all processing for this pixel is completed
    end_st[end_heads[these_pix] - 1, these_pix] = 0
    end_heads[these_pix] = 0
    pixel_done[these_pix] = True  # all processing for pixel is completed
    got_case[these_pix] = True

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value.*", RuntimeWarning)
        g_pix = these_pix[variance[these_pix] > 0.]  # good pixels
    if len(g_pix) > 0:
        inv_var[g_pix] += 1.0 / variance[g_pix]

        # Append results to arrays
        opt_res.append_arr(num_seg, g_pix, intercept, slope, sig_intercept,
                           sig_slope, inv_var, save_opt)

        num_seg[g_pix] += 1
        f_max_seg = max(f_max_seg, num_seg.max())
    return f_max_seg


def fit_short_ngroups(
        ngroups, start, end_st, end_heads, pixel_done, all_pix,
        inv_var, num_seg, slope, intercept, variance, sig_intercept, sig_slope,
        opt_res, save_opt, mask_2d_init, ramp_mask_sum):
    """
    Special case fitting for short ngroups fit.

    Parameters
    ----------
    ngroups : int
        number of groups in exposure

    start : ndarray
        lowest channel in fit, 1-D int

    end_st : ndarray
        stack array of endpoints, 2-D int

    end_heads : ndarray
        number of endpoints for each pixel, 1-D int

    pixel_done : ndarray
        whether each pixel's calculations are completed, 1-D bool

    all_pix : ndarray
        all pixels in image, 1-D

    inv_var : ndarray
        values of 1/variance for good pixels, 1-D float

    num_seg : ndarray
        numbers of segments for good pixels, 1-D int

    slope : ndarray
       weighted slope for current iteration's pixels for data section, 1-D
       float

    intercept : ndarray
       y-intercepts from fit for data section, 1-D float

    variance : float, ndarray
       variance of residuals for fit for data section, 1-D

    sig_intercept : ndarray
       sigma of y-intercepts from fit for data section, 1-D float

    sig_slope : ndarray
       sigma of slopes from fit for data section (for a single segment), 1-D
       float

    opt_res : OptRes object
        all fitting quantities, used to compute final results
        and to populate optional output product

    save_opt : bool
       save optional fitting results

    mask_2d_init : ndarray
        copy of intial mask_2d, 2-D bool

    ramp_mask_sum : ndarray
        number of channels to fit for each pixel, 1-D int

    Returns
    -------
    f_max_seg : int
        actual maximum number of segments within a ramp, updated here based on
        fitting ramps in the current data section; later used when truncating
        arrays before output.

    num_seg : ndarray
        numbers of segments for good pixels, 1-D int
    """

    # Dataset has NGROUPS=2, so special fitting is done for all pixels.
    # All segments are at the end of the array.
    #    - set start to -1 to designate all fitting done
    #    - remove current end from end stack
    #    - set number of ends to 0
    #    - add slopes and variances to running sums
    #    - set pixel_done to True to designate all fitting done
    if ngroups == 2:
        start[all_pix] = -1
        end_st[end_heads[all_pix] - 1, all_pix] = 0
        end_heads[all_pix] = 0
        pixel_done[all_pix] = True

        g_pix = all_pix[variance[all_pix] > 0.]
        if len(g_pix) > 0:
            inv_var[g_pix] += 1.0 / variance[g_pix]

            opt_res.append_arr(num_seg, g_pix, intercept, slope, sig_intercept,
                               sig_slope, inv_var, save_opt)

            num_seg[g_pix] = 1

            return 1, num_seg

    # Dataset has NGROUPS=1 ; so special fitting is done for all pixels
    # and all intervals are at the end of the array.
    #    - set start to -1 to designate all fitting done
    #    - remove current end from end stack
    #    - set number of ends to 0
    #    - add slopes and variances to running sums
    #    - set pixel_done to True to designate all fitting done
    start[all_pix] = -1
    end_st[end_heads[all_pix] - 1, all_pix] = 0
    end_heads[all_pix] = 0
    pixel_done[all_pix] = True

    wh_check = np.where(mask_2d_init[0, :] & (ramp_mask_sum == 1))
    if len(wh_check[0]) > 0:
        g_pix = wh_check[0]

        # Ignore all pixels having no good groups (so the single group is bad)
        if len(g_pix) > 0:
            inv_var[g_pix] += 1.0 / variance[g_pix]

            # Append results to arrays
            opt_res.append_arr(num_seg, g_pix, intercept, slope, sig_intercept,
                               sig_slope, inv_var, save_opt)

            num_seg[g_pix] = 1

    return 1, num_seg


def fit_lines(data, mask_2d, rn_sect, gain_sect, ngroups, weighting, gdq_sect_r, ramp_data):
    """
    Do linear least squares fit to data cube in this integration for a single
    segment for all pixels.  In addition to applying the mask due to identified
    cosmic rays, the data is also masked to exclude intervals that are too short
    to fit well. The first channel to fit in a segment is either the first group
    in the ramp, or a group in which a cosmic ray has been flagged.

    Parameters
    ----------
    data : ndarray
       array of values for current data section, 3-D float

    mask_2d : ndarray
       delineates which channels to fit for each pixel, 2-D bool

    rn_sect : ndarray
        read noise values for all pixels in data section

    gain_sect : ndarray
        gain values for all pixels in data section

    ngroups : int
        number of groups per integration

    weighting : str
        'optimal' specifies that optimal weighting should be used; currently
        the only weighting supported.

    gdq_sect_r : ndarray
        The section data presented as a 2-D array with dimnsions (ngroups, npix)

    ramp_data : RampData
        The ramp data needed for processing, specifically flag values.

    Returns
    -------
    Note - all of these pertain to a single segment (hence '_s')

    slope_s : ndarray
       1-D weighted slope for current iteration's pixels for data section

    intercept_s : ndarray
       1-D y-intercepts from fit for data section

    variance_s : ndarray
       1-D variance of residuals for fit for data section

    sig_intercept_s : ndarray
       1-D sigma of y-intercepts from fit for data section

    sig_slope_s : ndarray
       1-D sigma of slopes from fit for data section (for a single segment)

    """
    c_mask_2d = mask_2d.copy()

    # num of reads/pixel unmasked
    nreads_1d = c_mask_2d.astype(np.int16).sum(axis=0)
    npix = c_mask_2d.shape[1]

    slope_s = np.zeros(npix, dtype=np.float32)
    variance_s = np.zeros(npix, dtype=np.float32)
    intercept_s = np.zeros(npix, dtype=np.float32)
    sig_intercept_s = np.zeros(npix, dtype=np.float32)
    sig_slope_s = np.zeros(npix, dtype=np.float32)

    # Calculate slopes etc. for datasets having either 1 or 2 groups per
    #   integration, and return
    if ngroups == 1:  # process all pixels in 1 group/integration dataset
        slope_s, intercept_s, variance_s, sig_intercept_s, sig_slope_s = \
            fit_1_group(slope_s, intercept_s, variance_s, sig_intercept_s,
                        sig_slope_s, npix, data, c_mask_2d)

        return slope_s, intercept_s, variance_s, sig_intercept_s, sig_slope_s

    if ngroups == 2:  # process all pixels in 2 group/integration dataset
        rn_sect_1d = rn_sect.reshape(npix)
        slope_s, intercept_s, variance_s, sig_intercept_s, sig_slope_s = fit_2_group(
            slope_s, intercept_s, variance_s, sig_intercept_s, sig_slope_s, npix,
            data, c_mask_2d, rn_sect_1d, gdq_sect_r, ramp_data)

        return slope_s, intercept_s, variance_s, sig_intercept_s, sig_slope_s

    # reshape data_masked
    data_masked = data * np.reshape(c_mask_2d, data.shape)
    data_masked = np.reshape(data_masked, (data_masked.shape[0], npix))

    # For datasets having >2 groups/integration, for any semiramp in which the
    #   0th group is good and the 1st group is bad, determine whether or not to
    #   use the 0th group.
    wh_pix_1r = np.where(c_mask_2d[0, :] & (np.logical_not(c_mask_2d[1, :])))

    if len(wh_pix_1r[0]) > 0:
        slope_s, intercept_s, variance_s, sig_intercept_s, sig_slope_s = \
            fit_single_read(slope_s, intercept_s, variance_s, sig_intercept_s,
                            sig_slope_s, npix, data, wh_pix_1r)

    del wh_pix_1r

    # For datasets having >2 groups/integrations, for any semiramp in which only
    #   the 0th and 1st group are good, set slope, etc
    wh_pix_2r = np.where(c_mask_2d.sum(axis=0) == 2)  # ramps with 2 good groups

    slope_s, intercept_s, variance_s, sig_slope_s, sig_intercept_s = \
        fit_double_read(c_mask_2d, wh_pix_2r, data_masked, slope_s, intercept_s,
                        variance_s, sig_slope_s, sig_intercept_s, rn_sect)

    del wh_pix_2r

    # Select ramps having >2 good groups
    wh_pix_to_use = np.where(c_mask_2d.sum(axis=0) > 2)

    good_pix = wh_pix_to_use[0]  # Ramps with >2 good groups
    data_masked = data_masked[:, good_pix]

    del wh_pix_to_use

    xvalues = np.arange(data_masked.shape[0])[:, np.newaxis] * c_mask_2d
    xvalues = xvalues[:, good_pix]  # set to those pixels to be used

    c_mask_2d = c_mask_2d[:, good_pix]
    nreads_1d = nreads_1d[good_pix]

    if weighting.lower() == 'optimal':  # fit using optimal weighting
        # get sums from optimal weighting
        sumx, sumxx, sumxy, sumy, nreads_wtd, xvalues = calc_opt_sums(
            rn_sect, gain_sect, data_masked, c_mask_2d, xvalues, good_pix)

        slope, intercept, sig_slope, sig_intercept = \
            calc_opt_fit(nreads_wtd, sumxx, sumx, sumxy, sumy)

        variance = sig_slope**2.  # variance due to fit values

    elif weighting.lower() == 'unweighted':  # fit using unweighted weighting
        # get sums from unweighted weighting
        sumx, sumxx, sumxy, sumy = calc_unwtd_sums(data_masked, xvalues)

        slope, intercept, sig_slope, sig_intercept, line_fit =\
            calc_unwtd_fit(xvalues, nreads_1d, sumxx, sumx, sumxy, sumy)

        denominator = nreads_1d * sumxx - sumx**2

        # In case this branch is ever used again, disable, and then re-enable
        #   harmless arithmetic warrnings
        warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
        warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)
        variance = nreads_1d / denominator
        warnings.resetwarnings()

        denominator = 0

    else:  # unsupported weighting type specified
        log.error('FATAL ERROR: unsupported weighting type specified.')

    slope_s[good_pix] = slope
    variance_s[good_pix] = variance
    intercept_s[good_pix] = intercept
    sig_intercept_s[good_pix] = sig_intercept
    sig_slope_s[good_pix] = sig_slope

    return slope_s, intercept_s, variance_s, sig_intercept_s, sig_slope_s


def fit_single_read(slope_s, intercept_s, variance_s, sig_intercept_s,
                    sig_slope_s, npix, data, wh_pix_1r):
    """
    For datasets having >2 groups/integrations, for any semiramp in which the
    0th group is good and the 1st group is either SAT or CR, set slope, etc.

    Parameters
    ----------
    slope_s : ndarray
        1-D weighted slope for current iteration's pixels for data section

    intercept_s : ndarray
        1-D y-intercepts from fit for data section

    variance_s : ndarray
        1-D variance of residuals for fit for data section

    sig_intercept_s : ndarray
        1-D sigma of y-intercepts from fit for data section

    sig_slope_s : ndarray
        1-D sigma of slopes from fit for data section

    npix : int
        number of pixels in 2D array

    data : float
        array of values for current data section

    wh_pix_1r : tuple
        locations of pixels whose only good group is the 0th group

    Returns
    -------
    slope_s : ndarray
        1-D weighted slope for current iteration's pixels for data section

    intercept_s : ndarray
        1-D y-intercepts from fit for data section

    variance_s : ndarray
        1-D variance of residuals for fit for data section

    sig_slope_s : ndarray
        1-D sigma of slopes from fit for data section

    sig_intercept_s : ndarray
        1-D sigma of y-intercepts from fit for data section
    """
    data0_slice = data[0, :, :].reshape(npix)
    slope_s[wh_pix_1r] = data0_slice[wh_pix_1r]

    # The following arrays will have values correctly calculated later; for
    #   now they are just place-holders
    variance_s[wh_pix_1r] = utils.LARGE_VARIANCE
    sig_slope_s[wh_pix_1r] = 0.
    intercept_s[wh_pix_1r] = 0.
    sig_intercept_s[wh_pix_1r] = 0.

    return slope_s, intercept_s, variance_s, sig_slope_s, sig_intercept_s


def fit_double_read(mask_2d, wh_pix_2r, data_masked, slope_s, intercept_s,
                    variance_s, sig_slope_s, sig_intercept_s, rn_sect):
    """
    Process all semi-ramps having exactly 2 good groups. May need to optimize
    later to remove loop over pixels.

    Parameters
    ----------
    mask_2d : ndarray
        2-D bool delineates which channels to fit for each pixel

    wh_pix_2r : tuple
        locations of pixels whose only good groups are the 0th and the 1st

    data_masked : ndarray
        2-D masked values for all pixels in data section

    slope_s : ndarray
        1-D weighted slope for current iteration's pixels for data section

    intercept_s : ndarray
        1-D y-intercepts from fit for data section

    variance_s : ndarray
        1-D variance of residuals for fit for data section

    sig_slope_s : ndarray
        1-D sigma of slopes from fit for data section

    sig_intercept_s : ndarray
        1-D sigma of y-intercepts from fit for data section

    rn_sect : ndarray
        2-D read noise values for all pixels in data section

    Returns
    -------
    slope_s : ndarray
        1-D weighted slope for current iteration's pixels for data section

    intercept_s : ndarray
        1-D y-intercepts from fit for data section

    variance_s : ndarray
        1-D variance of residuals for fit for data section

    sig_slope_s : ndarray
        1-D sigma of slopes from fit for data section

    sig_intercept_s : ndarray
        1-D sigma of y-intercepts from fit for data section
    """
    rn_sect_flattened = rn_sect.flatten()

    for ff in range(len(wh_pix_2r[0])):  # loop over the pixels
        pixel_ff = wh_pix_2r[0][ff]  # pixel index (1d)

        rn = rn_sect_flattened[pixel_ff]  # read noise for this pixel

        read_nums = np.where(mask_2d[:, pixel_ff])
        second_read = read_nums[0][1]
        data_ramp = data_masked[:, pixel_ff] * mask_2d[:, pixel_ff]
        data_semi = data_ramp[mask_2d[:, pixel_ff]]  # picks only the 2
        diff_data = data_semi[1] - data_semi[0]

        slope_s[pixel_ff] = diff_data
        intercept_s[pixel_ff] = \
            data_semi[1] * (1. - second_read) + data_semi[0] * second_read  # by geometry
        variance_s[pixel_ff] = 2.0 * rn * rn
        sig_slope_s[pixel_ff] = np.sqrt(2) * rn
        sig_intercept_s[pixel_ff] = np.sqrt(2) * rn

    return slope_s, intercept_s, variance_s, sig_slope_s, sig_intercept_s


def calc_unwtd_fit(xvalues, nreads_1d, sumxx, sumx, sumxy, sumy):
    """
    Do linear least squares fit to data cube in this integration, using
    unweighted fits to the segments. Currently not supported.

    Parameters
    ----------
    xvalues : ndarray
        1-D int indices of valid pixel values for all groups

    nreads_1d : ndarray
        1-D int number of reads in an integration

    sumxx : float
        sum of squares of xvalues

    sumx : float
        sum of xvalues

    sumxy : float
        sum of product of xvalues and data

    sumy : float
        sum of data

    Returns
    -------
    slope : ndarray
       1-D weighted slope for current iteration's pixels for data section

    intercept : ndarray
       1-D y-intercepts from fit for data section

    sig_slope : ndarray
       1-D sigma of slopes from fit for data section

    sig_intercept : ndarray
       1-D sigma of y-intercepts from fit for data section

    line_fit : ndarray
       1-D values of fit using slope and intercept
    """

    denominator = nreads_1d * sumxx - sumx**2

    # In case this branch is ever used again, suppress, and then re-enable
    #   harmless arithmetic warnings
    warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
    warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)
    slope = (nreads_1d * sumxy - sumx * sumy) / denominator
    intercept = (sumxx * sumy - sumx * sumxy) / denominator
    sig_intercept = (sumxx / denominator)**0.5
    sig_slope = (nreads_1d / denominator)**0.5
    warnings.resetwarnings()

    line_fit = (slope * xvalues) + intercept

    return slope, intercept, sig_slope, sig_intercept, line_fit


def calc_opt_fit(nreads_wtd, sumxx, sumx, sumxy, sumy):
    """
    Do linear least squares fit to data cube in this integration for a single
    semi-ramp for all pixels, using optimally weighted fits to the semi_ramps.
    The weighting uses the formulation by Fixsen (Fixsen et al, PASP, 112, 1350).
    Note - these weights, sigmas, and variances pertain only to the fitting, and
    the variances are *NOT* the variances of the slope due to noise.

    Parameters
    ----------
    nreads_wtd : ndarray
        sum of product of data and optimal weight, 1-D float

    sumxx : ndarray
        sum of squares of xvalues, 1-D float

    sumx : ndarray
        sum of xvalues, 1-D float

    sumxy : ndarray
        sum of product of xvalues and data, 1-D float

    sumy : ndarray
        sum of data, 1-D float

    Returns
    -------
    slope : ndarray
       weighted slope for current iteration's pixels for data section, 1-D
       float

    intercept : ndarray
       y-intercepts from fit for data section, 1-D float

    sig_slope : ndarray
       sigma of slopes from fit for data section, 1-D float

    sig_intercept : ndarray
       sigma of y-intercepts from fit for data section, 1-D float
    """
    denominator = nreads_wtd * sumxx - sumx**2

    # Suppress, and then re-enable harmless arithmetic warnings
    warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
    warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)

    slope = (nreads_wtd * sumxy - sumx * sumy) / denominator
    intercept = (sumxx * sumy - sumx * sumxy) / denominator
    sig_intercept = (sumxx / denominator)**0.5
    sig_slope = (nreads_wtd / denominator)**0.5  # STD of the slope's fit

    warnings.resetwarnings()

    return slope, intercept, sig_slope, sig_intercept


def fit_1_group(slope_s, intercept_s, variance_s, sig_intercept_s,
                sig_slope_s, npix, data, mask_2d):
    """
    This function sets the fitting arrays for datasets having only 1 group
    per integration.

    Parameters
    ----------
    slope_s : ndarray
        weighted slope for current iteration's pixels for data section, 1-D float

    intercept_s : ndarray
        y-intercepts from fit for data section, 1-D float

    variance_s : ndarray
        variance of residuals for fit for data section, 1-D float

    sig_intercept_s : ndarray
        sigma of y-intercepts from fit for data section, 1-D float

    sig_slope_s : ndarray
        sigma of slopes from fit for data section, 1-D float

    npix : int
        number of pixels in 2d array

    data : float
        array of values for current data section

    mask_2d : ndarray
        delineates which channels to fit for each pixel, 2-D bool

    Returns
    -------
    slope_s : ndarray
        weighted slope for current iteration's pixels for data section, 1-D float

    intercept_s : ndarray
        y-intercepts from fit for data section, 1-D float

    variance_s : ndarray
        variance of residuals for fit for data section, 1-D float

    sig_intercept_s : ndarray
        sigma of y-intercepts from fit for data section, 1-D float

    sig_slope_s : ndarray
        sigma of slopes from fit for data section, 1-D float
    """
    # For pixels not saturated, recalculate the slope as the value of the SCI
    #   data in that group, which will later be divided by the group exposure
    #   time to give the count rate. Recalculate other fit quantities to be
    #   benign.
    slope_s = data[0, :, :].reshape(npix)

    # The following arrays will have values correctly calculated later; for
    #    now they are just place-holders
    variance_s = np.zeros(npix, dtype=np.float32) + utils.LARGE_VARIANCE
    sig_slope_s = slope_s * 0.
    intercept_s = slope_s * 0.
    sig_intercept_s = slope_s * 0.

    # For saturated pixels, overwrite slope with benign values.
    wh_sat0 = np.where(np.logical_not(mask_2d[0, :]))

    if len(wh_sat0[0]) > 0:
        sat_pix = wh_sat0[0]
        slope_s[sat_pix] = 0.

    return slope_s, intercept_s, variance_s, sig_intercept_s, sig_slope_s


def check_both_groups_good(gdq):
    """
    Special case checker for 2 group ramps.  This checks to see if both groups
    in a two group ramp are good.

    Parameter
    ---------
    gdq : ndarray
        The group DQ, 2-D uint8 with dimensions (2, npix), where npix is the
        number of groups, i.e., npx = nrows * ncols.

    Return
    ------
    both : ndarray
        Boolean array for pixels with 2 good groups.
    """
    # Get each group for every pixel.
    g0 = gdq[0, :]
    g1 = gdq[1, :]

    # Mark the pixels with good groups in the zeroth group.
    group_0_good = np.zeros(g0.shape, dtype=bool)
    group_0_good[g0 == 0] = True

    # Mark the pixels with good groups in the first group.
    group_1_good = np.zeros(g1.shape, dtype=bool)
    group_1_good[g1 == 0] = True

    # Mark the pixels with good groups in the both groups.
    both = group_0_good & group_1_good

    return both


def check_good_0_bad_1(gdq):
    """
    Special case checker for 2 group ramps.  This checks to see if group 0 is
    good, but group 1 is bad, making it effectively a one group ramp.

    Parameter
    ---------
    gdq : ndarray
        The group DQ, 2-D uint8 with dimensions (2, npix), where npix is the
        number of groups, i.e., npx = nrows * ncols.

    Return
    ------
    both : ndarray
        Boolean array for pixels with good 0 group and bad 1 group
    """
    # Get each group for every pixel.
    g0 = gdq[0, :]
    g1 = gdq[1, :]

    # Mark the pixels with good groups in the zeroth group.
    group_0_good = np.zeros(g0.shape, dtype=bool)
    group_0_good[g0 == 0] = True

    # Mark the pixels with bad groups in the first group.
    group_1_good = np.zeros(g1.shape, dtype=bool)
    group_1_good[g1 != 0] = True

    # Mark the pixels with good group 0 and bad group 1.
    both = group_0_good & group_1_good

    return both


def check_bad_0_good_1(gdq, sat):
    """
    Special case checker for 2 group ramps.  This checks to see if group 0 is
    bad, but group 1 is good, making it effectively a one group ramp.

    Parameter
    ---------
    gdq : ndarray
        The group DQ, 2-D uint8 with dimensions (2, npix), where npix is the
        number of groups, i.e., npx = nrows * ncols.

    sat : uint8
        The group DQ saturation flag.

    Return
    ------
    both : ndarray
        Boolean array for pixels with bad 0 group and good 1 group
    """
    # Get each group for every pixel.
    g0 = gdq[0, :]
    g1 = gdq[1, :]

    # Mark the pixels with bad groups in the zeroeth group.
    group_0_bad = np.zeros(g0.shape, dtype=bool)
    group_0_bad[g0 != 0] = True

    # Mark pixels flagged as saturated in zeroeth group, which means group 1
    # cannot be a good group.
    group_0_sat = np.zeros(g0.shape, dtype=np.uint8)
    group_0_sat = np.bitwise_and(g0, sat)
    group_0_sat.dtype = bool

    # Mark pixels flagged in the zeroeth group, but not flagged as saturated.
    group_0_bad_nsat = group_0_bad ^ group_0_sat

    # Mark pixels with good first group.
    group_1_good = np.zeros(g1.shape, dtype=bool)
    group_1_good[g1 == 0] = True

    # Mark the pixels with non-saturated bad zeroeth group and good first group.
    both = group_0_bad_nsat & group_1_good

    return both


def fit_2_group(slope_s, intercept_s, variance_s, sig_intercept_s, sig_slope_s,
                npix, data, mask_2d, rn_sect_1d, gdq_sect_r, ramp_data):
    """
    This function sets the fitting arrays for datasets having only 2 groups
    per integration.

    Parameters
    ----------
    slope_s : ndarray
        weighted slope for current iteration's pixels for data section, 1-D
        float

    intercept_s : ndarray
        y-intercepts from fit for data section, 1-D float

    variance_s : ndarray
        variance of residuals for fit for data section, 1-D float

    sig_intercept_s : ndarray
        sigma of y-intercepts from fit for data section, 1-D float

    sig_slope_s : ndarray
        sigma of slopes from fit for data section, 1-D float

    npix : int
        number of pixels in 2d array

    data : ndarray
        array of values for current data section, 3-D float

    mask_2d : ndarray
        delineates which channels to fit for each pixel, 2-D bool

    rn_sect_1d : ndarray
        read noise values for all pixels in data section, 1-D float

    gdq_sect_r : ndarray
        The section data presented as a 2-D array with dimnsions (ngroups, npix)

    ramp_data : RampData
        The ramp data needed for processing, specifically flag values.

    Returns
    -------
    slope_s : ndarray
        weighted slope for current iteration's pixels for data section, 1-D float

    intercept_s : ndarray
        y-intercepts from fit for data section, 1-D float

    variance_s : ndarray
        variance of residuals for fit for data section, 1-D float

    sig_intercept_s : ndarray
        sigma of y-intercepts from fit for data section, 1-D float

    sig_slope_s : ndarray
        sigma of slopes from fit for data section, 1-D float
    """
    # Shape data as (ngroups, npix)
    data_r = data.reshape((2, npix))

    # Special case 1.  Both groups in ramp are good.
    both_groups_good = check_both_groups_good(gdq_sect_r)
    wh_sat_no = np.where(both_groups_good)
    if len(wh_sat_no[0]) > 0:
        data0 = data_r[0, :]
        data1 = data_r[1, :]
        slope_s[wh_sat_no] = data1[wh_sat_no] - data0[wh_sat_no]
        sig_slope_s[wh_sat_no] = np.sqrt(2) * rn_sect_1d[wh_sat_no]
        intercept_s[wh_sat_no] = data0[wh_sat_no] - data1[wh_sat_no]
        sig_intercept_s[wh_sat_no] = np.sqrt(2) * rn_sect_1d[wh_sat_no]
        variance_s[wh_sat_no] = np.sqrt(2) * rn_sect_1d[wh_sat_no]
    del wh_sat_no

    # For one group segments only the slope is computed from the data.  The
    # variance is set to something non-zero, so the data is not thrown out
    # later.  The other values remain zero.

    # Special case 3.  Good 0th group, bad 1st group.
    good_0_bad_1 = check_good_0_bad_1(gdq_sect_r)
    one_group_locs = np.where(good_0_bad_1)
    if len(one_group_locs[0]) > 0:
        data0 = data_r[0, :]
        slope_s[one_group_locs] = data0[one_group_locs]
        variance_s[one_group_locs] = 1.
    del one_group_locs

    # Special case 4.  Bad 0th group, good 1st group.
    bad_0_good_1 = check_bad_0_good_1(gdq_sect_r, ramp_data.flags_saturated)
    one_group_locs = np.where(bad_0_good_1)
    if len(one_group_locs[0]) > 0:
        data1 = data_r[1, :]
        slope_s[one_group_locs] = data1[one_group_locs]
        variance_s[one_group_locs] = 1.
    del one_group_locs

    return slope_s, intercept_s, variance_s, sig_intercept_s, sig_slope_s


def calc_num_seg(gdq, n_int, jump_det, do_not_use):
    """
    Calculate the maximum number of segments that will be fit within an
    integration, calculated over all pixels and all integrations.  This value
    is based on the locations of cosmic ray-affected pixels in all of the ramps,
    and will be used to allocate arrays used for the optional output product.

    Parameters
    ----------
    gdq : ndarray
        cube of GROUPDQ array for a data, 3-D flag

    n_int : int (unused)
        total number of integrations in data set

    jump_det: uint32
        Jump detection flag

    do_not_use: uint32
        Do not use flag

    Return
    -------
    max_num_seg : int
        The maximum number of segements within an integration
    max_cr : int
        The maximum number of cosmic rays within an integration
    """
    max_cr = 0  # max number of CRS for all integrations

    # For all 2d pixels, get max number of CRs or DO_NOT_USE flags along their
    # ramps, to use as a surrogate for the number of segments along the ramps
    # Note that we only care about flags that are NOT in the first or last groups,
    # because exclusion of a first or last group won't result in an additional segment.
    check_flag = jump_det | do_not_use
    max_cr = np.count_nonzero(np.bitwise_and(gdq[:, 1:-1], check_flag), axis=1).max()

    # Do not want to return a value > the number of groups, which can occur if
    #  this is a MIRI dataset in which the first or last group was flagged as
    #  DO_NOT_USE and also flagged as a jump.
    max_num_seg = int(max_cr) + 1  # n CRS implies n+1 segments
    if max_num_seg > gdq.shape[1]:
        max_num_seg = gdq.shape[1]

    return max_num_seg, max_cr


def calc_unwtd_sums(data_masked, xvalues):
    """
    Calculate the sums needed to determine the slope and intercept (and sigma
    of each) using an unweighted fit. Unweighted fitting currently not
    supported.

    Parameters
    ----------
    data_masked : ndarray
        masked values for all pixels in data section, 2-D float

    xvalues : ndarray
        indices of valid pixel values for all groups, 1-D int

    Return:
    -------
    sumx : float
        sum of xvalues

    sumxx : float
        sum of squares of xvalues

    sumxy : float
        sum of product of xvalues and data

    sumy : float
        sum of data
    """
    sumx = xvalues.sum(axis=0)
    sumxx = (xvalues**2).sum(axis=0)
    sumy = (np.reshape(data_masked.sum(axis=0), sumx.shape))
    sumxy = (xvalues * np.reshape(data_masked, xvalues.shape)).sum(axis=0)

    return sumx, sumxx, sumxy, sumy


def calc_opt_sums(rn_sect, gain_sect, data_masked, mask_2d, xvalues, good_pix):
    """
    Calculate the sums needed to determine the slope and intercept (and sigma of
    each) using the optimal weights.  For each good pixel's segment, from the
    initial and final indices and the corresponding number of counts, calculate
    the SNR. From the SNR, calculate the weighting exponent using the formulation
    by Fixsen (Fixsen et al, PASP, 112, 1350). Using this exponent and the gain
    and the readnoise, the weights are calculated from which the sums are
    calculated.

    Parameters
    ----------
    rn_sect : ndarray
        read noise values for all pixels in data section, 2-D float

    gain_sect : ndarray
        gain values for all pixels in data section, 2-D float

    data_masked : ndarray
        masked values for all pixels in data section, 2-D float

    mask_2d : ndarray
        delineates which channels to fit for each pixel, 2-D bool

    xvalues : ndarray
        indices of valid pixel values for all groups, 2-D int

    good_pix : ndarray
        indices of pixels having valid data for all groups, 1-D int

    Return:
    -------
    sumx : float
        sum of xvalues

    sumxx : float
        sum of squares of xvalues

    sumxy : float
        sum of product of xvalues and data

    sumy : float
        sum of data

    nreads_wtd : ndarray
        sum of optimal weights, 1-D float

    xvalues : ndarray
        rolled up indices of valid pixel values for all groups, 2-D int
    """
    c_mask_2d = mask_2d.copy()  # copy the mask to prevent propagation
    rn_sect = np.float32(rn_sect)

    # Return 'empty' sums if there is no more data to fit
    if data_masked.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]),\
            np.array([]), np.array([])

    # get initial group for each good pixel for this semiramp
    fnz = np.argmax(c_mask_2d, axis=0)

    # For those pixels that are all False, set to sentinel value of -1
    fnz[c_mask_2d.sum(axis=0) == 0] = -1

    mask_2d_sum = c_mask_2d.sum(axis=0)   # number of valid groups/pixel

    # get final valid group for each pixel for this semiramp
    ind_lastnz = fnz + mask_2d_sum - 1

    # get SCI value of initial good group for semiramp
    data_zero = data_masked[fnz, range(data_masked.shape[1])]

    # get SCI value of final good group for semiramp
    data_final = data_masked[(ind_lastnz), range(data_masked.shape[1])]
    data_diff = data_final - data_zero  # correctly does *NOT* have nans

    ind_lastnz = 0

    # Use the readnoise and gain for good pixels only
    rn_sect_rav = rn_sect.flatten()[good_pix]
    rn_2_r = rn_sect_rav * rn_sect_rav

    gain_sect_r = gain_sect.flatten()[good_pix]

    # Calculate the sigma for nonzero gain values
    sigma_ir = data_final.copy() * 0.0
    numer_ir = data_final.copy() * 0.0

    # Calculate the SNR for pixels from the readnoise, the gain, and the
    # difference between the last and first reads for pixels where this results
    # in a positive SNR. Otherwise set the SNR to 0.
    sqrt_arg = rn_2_r + data_diff * gain_sect_r
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value.*", RuntimeWarning)
        wh_pos = np.where((sqrt_arg >= 0.) & (gain_sect_r != 0.))
    numer_ir[wh_pos] = \
        np.sqrt(rn_2_r[wh_pos] + data_diff[wh_pos] * gain_sect_r[wh_pos])
    sigma_ir[wh_pos] = numer_ir[wh_pos] / gain_sect_r[wh_pos]
    snr = data_diff * 0.
    snr[wh_pos] = data_diff[wh_pos] / sigma_ir[wh_pos]
    snr[np.isnan(snr)] = 0.0
    snr[snr < 0.] = 0.0

    del wh_pos

    gain_sect_r = 0
    numer_ir = 0
    data_diff = 0
    sigma_ir = 0

    power_wt_r = calc_power(snr)  # Get the interpolated power for this SNR
    # Make array of number of good groups, and exponents for each pixel
    num_nz = (data_masked != 0.).sum(0)  # number of nonzero groups per pixel
    nrd_data_a = num_nz.copy()
    num_nz = 0

    nrd_prime = (nrd_data_a - 1) / 2.
    nrd_data_a = 0

    # Calculate inverse read noise^2 for use in weights
    # Suppress, then re-enable, harmless arithmetic warning
    warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)
    invrdns2_r = 1. / rn_2_r
    warnings.resetwarnings()

    rn_sect = 0
    fnz = 0

    # Set optimal weights for each group of each pixel;
    #    for all pixels at once, loop over the groups
    wt_h = np.zeros(data_masked.shape, dtype=np.float32)

    for jj_rd in range(data_masked.shape[0]):
        wt_h[jj_rd, :] = \
            abs((abs(jj_rd - nrd_prime) / nrd_prime) ** power_wt_r) * invrdns2_r

    wt_h[np.isnan(wt_h)] = 0.
    wt_h[np.isinf(wt_h)] = 0.

    # For all pixels, 'roll' up the leading zeros such that the 0th group of
    #  each pixel is the lowest nonzero group for that pixel
    wh_m2d_f = np.logical_not(c_mask_2d[0, :])  # ramps with initial group False
    while wh_m2d_f.sum() > 0:
        data_masked[:, wh_m2d_f] = np.roll(data_masked[:, wh_m2d_f], -1, axis=0)
        c_mask_2d[:, wh_m2d_f] = np.roll(c_mask_2d[:, wh_m2d_f], -1, axis=0)
        xvalues[:, wh_m2d_f] = np.roll(xvalues[:, wh_m2d_f], -1, axis=0)
        wh_m2d_f = np.logical_not(c_mask_2d[0, :])

    # Create weighted sums for Poisson noise and read noise
    nreads_wtd = (wt_h * c_mask_2d).sum(axis=0)  # using optimal weights

    sumx = (xvalues * wt_h).sum(axis=0)
    sumxx = (xvalues**2 * wt_h).sum(axis=0)

    c_data_masked = data_masked.copy()
    c_data_masked[np.isnan(c_data_masked)] = 0.
    sumy = (np.reshape((c_data_masked * wt_h).sum(axis=0), sumx.shape))
    sumxy = (xvalues * wt_h * np.reshape(c_data_masked, xvalues.shape)).sum(axis=0)

    return sumx, sumxx, sumxy, sumy, nreads_wtd, xvalues
