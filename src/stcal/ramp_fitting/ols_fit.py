#! /usr/bin/env python

import logging
import multiprocessing
import sys
import time
from multiprocessing import cpu_count

import numpy as np

from stcal.multiprocessing import compute_num_cores

from . import ramp_fit_class
from .slope_fitter import ols_slope_fitter  # c extension

BUFSIZE = 1024 * 300000  # 300Mb cache size for data section

log = logging.getLogger(__name__)


def ols_ramp_fit_multi(ramp_data, save_opt, readnoise_2d, gain_2d, weighting, max_cores):
    """
    Setup the inputs to ols_ramp_fit with and without multiprocessing. The
    inputs will be sliced into the number of cores that are being used for
    multiprocessing. Because the data models cannot be pickled, only numpy
    arrays are passed and returned as parameters to ols_ramp_fit.

    Parameters
    ----------
    ramp_data : RampData
        Input data necessary for computing ramp fitting.

    save_opt : bool
       calculate optional fitting results

    readnoise_2d : ndarray
        readnoise for all pixels

    gain_2d : ndarray
        gain for all pixels

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
    nrows = ramp_data.data.shape[2]
    num_available_cores = cpu_count()
    number_slices = compute_num_cores(max_cores, nrows, num_available_cores)
    log.info(f"Number of multiprocessing slices: {number_slices}")

    # For MIRI datasets having >1 group, if all pixels in the final group are
    #   flagged as DO_NOT_USE, resize the input model arrays to exclude the
    #   final group.  Similarly, if leading groups 1 though N have all pixels
    #   flagged as DO_NOT_USE, those groups will be ignored by ramp fitting, and
    #   the input model arrays will be resized appropriately. If all pixels in
    #   all groups are flagged, return None for the models.
    if ramp_data.instrument_name == "MIRI" and ramp_data.data.shape[1] > 1:
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
            ramp_data, save_opt, readnoise_2d, gain_2d, weighting
        )
        if image_info is None or integ_info is None:
            return None, None, None

        return image_info, integ_info, opt_info

    # Call ramp fitting for multi-processor (multiple data slices) case
    image_info, integ_info, opt_info = ols_ramp_fit_multiprocessing(
        ramp_data, save_opt, readnoise_2d, gain_2d, weighting, number_slices
    )

    return image_info, integ_info, opt_info


def ols_ramp_fit_multiprocessing(ramp_data, save_opt, readnoise_2d, gain_2d, weighting, number_slices):
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
    log.info("Number of processors used for multiprocessing: %s", number_slices)
    slices, rows_per_slice = compute_slices_for_starmap(
        ramp_data, save_opt, readnoise_2d, gain_2d, weighting, number_slices
    )

    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(processes=number_slices)
    pool_results = pool.starmap(ols_ramp_fit_single, slices)
    pool.close()
    pool.join()

    # Reassemble results
    image_info, integ_info, opt_info = assemble_pool_results(
        ramp_data, save_opt, pool_results, rows_per_slice
    )

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
    for result in pool_results:
        image_slice, integ_slice, opt_slice = result
        if image_slice is None or integ_slice is None:
            return None, None, None

    image_info, integ_info, opt_info = create_output_info(ramp_data, pool_results, save_opt)

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
    (slope, sigslope, var_poisson, var_rnoise, yint, sigyint, pedestal, weights, crmag) = opt_info
    (oslope, osigslope, ovar_poisson, ovar_rnoise, oyint, osigyint, opedestal, oweights, ocrmag) = opt_slice

    srow, erow = row_start, row_start + nrows

    # The optional results product is of variable size in its second dimension.
    # The number of segments/cosmic rays determine the final products size.
    # Because each slice is computed independently, the number of segments may
    # differ from segment to segment.  The final output product is created
    # using the max size for this dimension.  To ensure correct assignment is
    # done during this step, the second dimension, as well as the row
    # dimension, must be specified.
    slope[:, : oslope.shape[1], srow:erow, :] = oslope
    sigslope[:, : osigslope.shape[1], srow:erow, :] = osigslope
    var_poisson[:, : ovar_poisson.shape[1], srow:erow, :] = ovar_poisson
    var_rnoise[:, : ovar_rnoise.shape[1], srow:erow, :] = ovar_rnoise
    yint[:, : oyint.shape[1], srow:erow, :] = oyint
    sigyint[:, : osigyint.shape[1], srow:erow, :] = osigyint
    weights[:, : oweights.shape[1], srow:erow, :] = oweights
    crmag[:, : ocrmag.shape[1], srow:erow, :] = ocrmag

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

        opt_info = (
            oslope,
            osigslope,
            ovar_poisson,
            ovar_rnoise,
            oyint,
            osigyint,
            opedestal,
            oweights,
            ocrmag,
        )
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


def compute_slices_for_starmap(ramp_data, save_opt, readnoise_2d, gain_2d, weighting, number_slices):
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
        rnoise_slice = readnoise_2d[start_row : start_row + rslices[k], :].copy()
        gain_slice = gain_2d[start_row : start_row + rslices[k], :].copy()
        slices.insert(k, (ramp_slice, save_opt, rnoise_slice, gain_slice, weighting))
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
    data = ramp_data.data[:, :, start_row : start_row + nrows, :].copy()
    groupdq = ramp_data.groupdq[:, :, start_row : start_row + nrows, :].copy()
    pixeldq = ramp_data.pixeldq[start_row : start_row + nrows, :].copy()
    average_dark_current = ramp_data.average_dark_current[start_row : start_row + nrows, :].copy()

    ramp_data_slice.set_arrays(data, groupdq, pixeldq, average_dark_current)

    if ramp_data.zeroframe is not None:
        ramp_data_slice.zeroframe = ramp_data.zeroframe[:, start_row : start_row + nrows, :].copy()

    # Carry over meta data.
    ramp_data_slice.set_meta(
        name=ramp_data.instrument_name,
        frame_time=ramp_data.frame_time,
        group_time=ramp_data.group_time,
        groupgap=ramp_data.groupgap,
        nframes=ramp_data.nframes,
        drop_frames1=ramp_data.drop_frames1,
    )

    # Carry over DQ flags.
    ramp_data_slice.flags_do_not_use = ramp_data.flags_do_not_use
    ramp_data_slice.flags_jump_det = ramp_data.flags_jump_det
    ramp_data_slice.flags_saturated = ramp_data.flags_saturated
    ramp_data_slice.flags_no_gain_val = ramp_data.flags_no_gain_val
    ramp_data_slice.flags_unreliable_slope = ramp_data.flags_unreliable_slope
    ramp_data_slice.flags_chargeloss = ramp_data.flags_chargeloss

    # For possible CHARGELOSS flagging.
    if ramp_data.orig_gdq is not None:
        ogdq = ramp_data.orig_gdq[:, :, start_row : start_row + nrows, :].copy()
        ramp_data_slice.orig_gdq = ogdq
    else:
        ramp_data_slice.orig_gdq = None

    # Slice info
    ramp_data_slice.start_row = start_row
    ramp_data_slice.num_rows = nrows

    return ramp_data_slice


def ols_ramp_fit_single(ramp_data, save_opt, readnoise_2d, gain_2d, weighting):
    """
    Fit a ramp using ordinary least squares. Calculate the count rate for each
    pixel in all data cube sections and all integrations, equal to the weighted
    slope for all sections (intervals between cosmic rays) of the pixel's ramp
    divided by the effective integration time.

    Parameters
    ----------
    ramp_data : RampData
        Input data necessary for computing ramp fitting.

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
    c_start = time.time()

    ramp_data, gain_2d, readnoise_2d, bswap = endianness_handler(ramp_data, gain_2d, readnoise_2d)

    if ramp_data.drop_frames1 is None:
        ramp_data.drop_frames1 = 0

    log.debug("Running ols_slope_fitter")
    image_info, integ_info, opt_info = ols_slope_fitter(ramp_data, gain_2d, readnoise_2d, weighting, save_opt)

    c_end = time.time()

    # Read noise is used after STCAL ramp fitting for the CHARGELOSS
    # processing, so make sure it works right for there.  In other words
    # if they got byteswapped for the C extension, they need to be
    # byteswapped back to properly work in python once returned from
    # ramp fitting.
    rn_bswap, gain_bswap = bswap
    if rn_bswap:
        readnoise_2d.view(readnoise_2d.dtype.newbyteorder("S")).byteswap(inplace=True)
    if gain_bswap:
        gain_2d.view(gain_2d.dtype.newbyteorder("S")).byteswap(inplace=True)

    c_diff = c_end - c_start
    log.info(f"Ramp Fitting C Time: {c_diff}")

    return image_info, integ_info, opt_info


def handle_array_endianness(arr, sys_order):
    """
    Determines if the array byte order is the same as the system byte order.  If
    it is not, then byteswap the array.

    Parameters
    ----------
    arr : ndarray
        The array whose endianness to check against the system endianness.

    sys_order : str
        The system order ("<" is little endian, while ">" is big endian).

    Return
    ------
    arr : ndarray
        The ndarray in the correct byte order
    """
    arr_order = arr.dtype.byteorder
    bswap = False
    if (arr_order == ">" and sys_order == "<") or (arr_order == "<" and sys_order == ">"):
        arr.view(arr.dtype.newbyteorder("S")).byteswap(inplace=True)
        bswap = True

    return arr, bswap


def endianness_handler(ramp_data, gain_2d, readnoise_2d):
    """
    Check all arrays for endianness against the system endianness,
    so when used by the C extension, the endianness is correct.  Numpy
    ndarrays can be in any byte order and is handled transparently to the
    user.  The arrays in the C extension are expected to be in byte order
    on the system which the ramp fitting is being run.

    Parameters
    ----------
    ramp_data : RampData
        Carries ndarrays needing checked and possibly byte swapped.

    gain_2d : ndarray
        An ndarray needing checked and possibly byte swapped.

    readnoise_2d : ndarray
        An ndarray needing checked and possibly byte swapped.

    Return
    ------
    ramp_data : RampData
        Carries ndarrays checked and possibly byte swapped.

    gain_2d : ndarray
        An ndarray checked and possibly byte swapped.

    readnoise_2d : ndarray
        An ndarray checked and possibly byte swapped.
    """
    sys_order = "<" if sys.byteorder == "little" else ">"

    # If the gain and/or readnoise arrays are byteswapped before going
    # into the C extension, then that needs to be noted and byteswapped
    # when returned from the C extension.
    gain_2d, gain_bswap = handle_array_endianness(gain_2d, sys_order)
    readnoise_2d, rn_bswap = handle_array_endianness(readnoise_2d, sys_order)

    ramp_data.data, _ = handle_array_endianness(ramp_data.data, sys_order)
    ramp_data.average_dark_current, _ = handle_array_endianness(ramp_data.average_dark_current, sys_order)
    ramp_data.groupdq, _ = handle_array_endianness(ramp_data.groupdq, sys_order)
    ramp_data.pixeldq, _ = handle_array_endianness(ramp_data.pixeldq, sys_order)

    return ramp_data, gain_2d, readnoise_2d, (rn_bswap, gain_bswap)


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
        True if usable data available for further processing.
    """
    data = ramp_data.data
    groupdq = ramp_data.groupdq
    orig_gdq = ramp_data.orig_gdq

    n_int, ngroups, nrows, ncols = data.shape

    num_bad_slices = 0  # number of initial groups that are all DO_NOT_USE

    while np.all(np.bitwise_and(groupdq[:, 0, :, :], ramp_data.flags_do_not_use)):
        num_bad_slices += 1
        ngroups -= 1

        # Check if there are remaining groups before accessing data
        if ngroups < 1:  # no usable data
            log.error("1. All groups have all pixels flagged as DO_NOT_USE,")
            log.error("  so will not process this dataset.")
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
        if orig_gdq is not None:
            orig_gdq = orig_gdq[:, num_bad_slices:, :, :]

    log.info("Number of leading groups that are flagged as DO_NOT_USE: %s", num_bad_slices)

    # If all groups were flagged, the final group would have been picked up
    #   in the while loop above, ngroups would have been set to 0, and Nones
    #   would have been returned.  If execution has gotten here, there must
    #   be at least 1 remaining group that is not all flagged.
    if np.all(np.bitwise_and(groupdq[:, -1, :, :], ramp_data.flags_do_not_use)):
        ngroups -= 1

        # Check if there are remaining groups before accessing data
        if ngroups < 1:  # no usable data
            log.error("2. All groups have all pixels flagged as DO_NOT_USE,")
            log.error("  so will not process this dataset.")
            return False

        data = data[:, :-1, :, :]
        groupdq = groupdq[:, :-1, :, :]
        if orig_gdq is not None:
            orig_gdq = orig_gdq[:, :-1, :, :]

        log.info("MIRI dataset has all pixels in the final group flagged as DO_NOT_USE.")

    # Next block is to satisfy github issue 1681:
    # "MIRI FirstFrame and LastFrame minimum number of groups"
    if ngroups < 2:
        log.warning("MIRI datasets require at least 2 groups/integration")
        log.warning("(NGROUPS), so will not process this dataset.")
        return False

    ramp_data.data = data
    ramp_data.groupdq = groupdq
    if orig_gdq is not None:
        ramp_data.orig_gdq = orig_gdq

    return True
