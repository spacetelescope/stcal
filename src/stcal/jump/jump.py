#  jump.py - detect cosmic ray jumps and their side effects like
#            snowballs and showers.

import logging
import multiprocessing
import time
import warnings
from scipy import signal

import numpy as np
import cv2 as cv
import astropy.stats as stats

from astropy.convolution import Ring2DKernel
from astropy.convolution import convolve

from .twopoint_difference_class import TwoPointParams
from . import twopoint_difference as twopt

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def detect_jumps_data(jump_data):
    """
    Detect jumps and their side effects, such as showers and snowballs.

    It loads and sets the various input data and parameters needed by each of
    the individual detection methods and then calls the detection methods in
    turn.

    Note that the detection methods are currently set up on the assumption
    that the input science data array will be in units of
    electrons, hence this routine scales those input arrays by the detector
    gain. The methods assume that the read noise values will be in units
    of DN.

    The gain is applied to the science data array using the
    appropriate instrument- and detector-dependent values for each pixel of an
    image.  Also, a 2-dimensional read noise array with appropriate values for
    each pixel is passed to the detection methods.

    Parameters
    ----------
    jump_data : JumpData
        Class containing parameters and methods to detect jumps.

    Returns
    -------
    gdq : int, 4D array
        updated group dq array

    pdq : int, 2D array
        updated pixel dq array

    total_primary_crs : int
        the number of primary cosmic rays found

    number_extended_events : int
        the number of showers or XXX found

    stddev : float
        standard deviation computed during sigma clipping
    """
    sat, jump, dnu = jump_data.fl_sat, jump_data.fl_jump, jump_data.fl_dnu
    number_extended_events = 0

    pdq = setup_pdq(jump_data)

    # Apply gain to the SCI and readnoise arrays so they're in units
    # of electrons
    data = jump_data.data * jump_data.gain_2d
    gdq = jump_data.gdq
    readnoise_2d = jump_data.rnoise_2d * jump_data.gain_2d

    # also apply to the after_jump thresholds
    # XXX Maybe move this computation
    jump_data.after_jump_flag_e1 = jump_data.after_jump_flag_dn1 * np.nanmedian(jump_data.gain_2d)
    jump_data.after_jump_flag_e2 = jump_data.after_jump_flag_dn2 * np.nanmedian(jump_data.gain_2d)

    # Apply the 2-point difference method as a first pass
    log.info("Executing two-point difference method")
    start = time.time()

    # figure out how many slices to make based on 'max_cores'
    max_available = multiprocessing.cpu_count()
    n_rows = data.shape[2]
    n_slices = calc_num_slices(n_rows, jump_data.max_cores, max_available)

    twopt_params = TwoPointParams(jump_data, False)
    if n_slices == 1:
        twopt_params.minimum_groups = 3  # XXX Should this be hard coded as 3?
        gdq, row_below_dq, row_above_dq, total_primary_crs, stddev = twopt.find_crs(
                    data, gdq, readnoise_2d, twopt_params)
    else:
        gdq, total_primary_crs, stddev = twopoint_diff_multi(
            jump_data, twopt_params, data, gdq, readnoise_2d, n_slices)

    # remove redundant bits in pixels that have jump flagged but were
    # already flagged as do_not_use or saturated.
    gdq[gdq & (jump | dnu) == (jump | dnu)] ^= jump
    gdq[gdq & (jump | sat) == (jump | sat)] ^= jump

    #  This is the flag that controls the flagging of snowballs.
    if jump_data.expand_large_events:
        gdq, total_snowballs = flag_large_events(gdq, jump, sat, jump_data)
        log.info("Total snowballs = %i", total_snowballs)
        number_extended_events = total_snowballs  # XXX overwritten

    if jump_data.find_showers:
        gdq, num_showers = find_faint_extended(data, gdq, pdq, readnoise_2d, jump_data)
        log.info("Total showers= %i", num_showers)
        number_extended_events = num_showers  # XXX overwritten

    elapsed = time.time() - start
    log.info("Total elapsed time = %g sec", elapsed)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*in divide.*", RuntimeWarning)
        # Back out the applied gain to the SCI and readnoise arrays so they're
        #    back in units of DN
        data /= jump_data.gain_2d
        readnoise_2d /= jump_data.gain_2d

    # Return the updated data quality arrays
    return gdq, pdq, total_primary_crs, number_extended_events, stddev


def twopoint_diff_multi(jump_data, twopt_params, data, gdq, readnoise_2d, n_slices):
    """
    Split data for jump detection multiprocessing.
    
    Parameters
    ----------
    jump_data : JumpData
        Class containing parameters and methods to detect jumps.

    twopt_params : TwoPointParams
        Class containing parameters and methods for two point differences.

    data : ndarray
        The science data, 4D array float.

    gdq : ndarray 
        The group DQ, 4D array uint8.

    readnoise_2d : ndarray
        The read noise reference, 2D array float.

    n_slices : int
        The number of data slices for multiprocessing.

    Returns
    -------
    gdq : ndarray
        the group DQ array, 4D uint8

    total_primary_crs : int
        total number of primary cosmic rays computed

    stddev : float
        standard deviation computed during sigma clipping
    """
    slices, yinc = slice_data(twopt_params, data, gdq, readnoise_2d, n_slices)

    log.info("Creating %d processes for jump detection ", n_slices)
    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(processes=n_slices)
    ######### JUST FOR DEBUGGING #########################
    # pool = ctx.Pool(processes=1)
    # Starts each slice in its own process. Starmap allows more than one
    # parameter to be passed.
    real_result = pool.starmap(twopt.find_crs, slices)
    pool.close()
    pool.join()

    return reassemble_sliced_data(real_result, jump_data, gdq, yinc)


def reassemble_sliced_data(real_result, jump_data, gdq, yinc):
    """
    Reassemble the data from each process for multiprocessing.

    Parameters
    ----------
    real_result : tuple
        The tuple return values from twopt.find_crs
        (gdq, row_below_gdq, row_above_gdq, num_primary_crs, dummy/stddev)

    jump_data : JumpData
        Class containing parameters and methods to detect jumps.

    gdq : ndarray
        The group DQ, 4D array uint8.

    yinc : int
        The number of rows in each slice (rows are the y-axis, so this
        says how many rows to increment to get to the next slice.

    Returns
    -------
    gdq : ndarray
        The group DQ, 4D array uint8.

    total_primary_crs : int
        Total number of primary cosmic rays detected.

    stddev : float
        standard deviation computed during sigma clipping

    """
    nints, ngroups, nrows, ncols = gdq.shape
    row_above_gdq = np.zeros((nints, ngroups, ncols), dtype=np.uint8)
    previous_row_above_gdq = np.zeros((nints, ngroups, ncols), dtype=np.uint8)
    row_below_gdq = np.zeros((nints, ngroups, ncols), dtype=np.uint8)

    # Reconstruct gdq, the row_above_gdq, and the row_below_gdq from the
    # slice result
    total_primary_crs = 0
    if jump_data.only_use_ints:
        stddev = np.zeros((ngroups - 1, nrows, ncols), dtype=np.float32)
    else:
        stddev = np.zeros((nrows, ncols), dtype=np.float32)

    # Reassemble the data
    for k, resultslice in enumerate(real_result):
        if len(real_result) == k + 1:  # last result
            gdq[:, :, k * yinc: nrows, :] = resultslice[0]
            if jump_data.only_use_ints:
                stddev[:, k * yinc: nrows, :] = resultslice[4]
            else:
                stddev[k * yinc: nrows, :] = resultslice[4]
        else:
            gdq[:, :, k * yinc: (k + 1) * yinc, :] = resultslice[0]
            if jump_data.only_use_ints:
                stddev[:, k * yinc: (k + 1) * yinc, :] = resultslice[4]
            else:
                stddev[k * yinc : (k + 1) * yinc, :] = resultslice[4]
        row_below_gdq[:, :, :] = resultslice[1]
        row_above_gdq[:, :, :] = resultslice[2]
        total_primary_crs += resultslice[3]
        if k != 0:
            # For all but the first slice, flag any CR neighbors in the top
            # row of the previous slice and flag any neighbors in the
            # bottom row of this slice saved from the top of the previous
            # slice
            gdq[:, :, k * yinc - 1, :] |= row_below_gdq[:, :, :]
            gdq[:, :, k * yinc, :] |= previous_row_above_gdq[:, :, :]

        # save the neighbors to be flagged that will be in the next slice
        previous_row_above_gdq = row_above_gdq.copy()

    return gdq, total_primary_crs, stddev



def slice_data(twopt_params, data, gdq, readnoise_2d, n_slices):
    """
    Create a slice of data for each process for multiprocessing.

    Parameters
    ----------
    twopt_params : TwoPointParams
        Class containing parameters and methods for two point differences.

    data : ndarray
        The science data, 4D array float.

    gdq : ndarray 
        The group DQ, 4D array uint8.

    readnoise_2d : ndarray
        The read noise reference, 2D array float.

    n_slices : int
        The number of data slices for multiprocessing.

    Returns
    -------
    slices : array
        The array of data slices to be used in multiprocessing

    yinc : int
        The number of rows in each slice (rows are the y-axis, so this
        says how many rows to increment to get to the next slice.
    """
    nrows = data.shape[2]
    yinc = nrows // n_slices
    slices = []
    # Slice up data, gdq, readnoise_2d into slices
    # Each element of slices is a tuple of
    # (data, gdq, readnoise_2d, rejection_thresh, three_grp_thresh,
    #  four_grp_thresh, nframes)
    twopt_params.copy_arrs = False  # we don't need to copy arrays again in find_crs
    for i in range(n_slices - 1):
        slices.insert(
            i,
            (
                data[:, :, i * yinc: (i + 1) * yinc, :],
                gdq[:, :, i * yinc: (i + 1) * yinc, :],
                readnoise_2d[i * yinc: (i + 1) * yinc, :],
                twopt_params,
            ),
        )

    # last slice get the rest
    slices.insert(
        n_slices - 1,
        (
            data[:, :, (n_slices - 1) * yinc: nrows, :],
            gdq[:, :, (n_slices - 1) * yinc: nrows, :],
            readnoise_2d[(n_slices - 1) * yinc: nrows, :],
            twopt_params,
        ),
    )
    return slices, yinc


def setup_pdq(jump_data):
    """
    Prepare the pixel DQ array for procesing, removing invalid data.

    Parameters
    ----------
    jump_data : JumpData
        Class containing parameters and methods to detect jumps.

    Returns
    -------
    pdq : ndarray
        The pixel DQ array (2D)
    """
    pdq = jump_data.pdq
    bad_gain = (jump_data.gain_2d <= 0.0) | np.isnan(jump_data.gain_2d)
    pdq[bad_gain] |= (jump_data.fl_ngv | jump_data.fl_dnu)

    return pdq


def flag_large_events(gdq, jump_flag, sat_flag, jump_data):
    """
    Control the creation of expanded regions that are flagged as jumps.

    These events are called snowballs for the NIR. While they are most commonly
    circular, there are elliptical ones. This routine does not handle the
    detection of MIRI showers.

    Parameters
    ----------
    gdq : int, 4D array
        Group dq array

    jump_flag : int
        DQ flag for jump detection.

    sat_flag: int
        DQ flag for saturation

    Returns
    -------
    total Snowballs
    """
    log.info("Flagging Snowballs")

    n_showers_grp = []
    total_snowballs = 0
    nints, ngrps, nrows, ncols = gdq.shape
    persist_jumps = np.zeros(shape=(nints, nrows, ncols), dtype=np.uint8)
    for integration in range(nints):
        for group in range(1, ngrps):
            current_gdq = gdq[integration, group, :, :]
            current_sat = np.bitwise_and(current_gdq, sat_flag)

            prev_gdq = gdq[integration, group - 1, :, :]
            prev_sat = np.bitwise_and(prev_gdq, sat_flag)

            not_prev_sat = np.logical_not(prev_sat)
            new_sat = current_sat * not_prev_sat
            if group < ngrps - 1:
                next_gdq = gdq[integration, group + 1, :, :]
                next_sat = np.bitwise_and(next_gdq, sat_flag)
                not_current_sat = np.logical_not(current_sat)
                next_new_sat = next_sat * not_current_sat

            next_sat_ellipses = find_ellipses(next_new_sat, sat_flag, jump_data.min_sat_area)
            sat_ellipses = find_ellipses(new_sat, sat_flag, jump_data.min_sat_area)

            # find the ellipse parameters for jump regions
            jump_ellipses = find_ellipses(
                gdq[integration, group, :, :], jump_flag, jump_data.min_jump_area)
            
            if jump_data.sat_required_snowball:
                gdq, snowballs, persist_jumps = make_snowballs(
                    gdq, integration, group, jump_ellipses, sat_ellipses,
                    next_sat_ellipses, jump_data, persist_jumps,
                )
            else:
                snowballs = jump_ellipses
            n_showers_grp.append(len(snowballs))
            total_snowballs += len(snowballs)
            gdq, num_events = extend_ellipses(
                gdq, integration, group, snowballs, jump_data,
                expansion=jump_data.expand_factor, num_grps_masked=0,
            )

    #  Test to see if the flagging of the saturated cores will be
    #  extended into the subsequent integrations. Persist_jumps contains
    #  all the pixels that were saturated in the cores of snowballs.
    if jump_data.mask_persist_grps_next_int:
        for intg in range(1, nints):
            if jump_data.persist_grps_flagged >= 1:
                last_grp_flagged = min(jump_data.persist_grps_flagged, ngrps)
                gdq[intg, 1:last_grp_flagged, :, :] = np.bitwise_or(
                        gdq[intg, 1:last_grp_flagged, :, :],
                        np.repeat(persist_jumps[intg - 1, np.newaxis, :, :],
                        last_grp_flagged - 1, axis=0))
    return gdq, total_snowballs


def extend_saturation(cube, grp, sat_ellipses, jump_data, persist_jumps):
    """
    Extend the saturated ellipses that are larger than the min_sat_radius.
    
    Parameters
    ----------
    cube : ndarray
        Group DQ cube for an integration.

    grp : int
        The current group.

    sat_ellipses : cv.ellipse
        The saturated ellipse.

    jump_data : JumpData

    persist_jumps : ndarray
        3D (nints, nrows, ncols) uint8

    Returns
    -------
    outcube : ndarray
        Group DQ cube for an integration.

    persist_jumps : ndarray
        3D (nints, nrows, ncols) uint8
    """
    ngroups, nrows, ncols = cube.shape
    satcolor = 22  # (0, 0, 22) is a dark blue in RGB
    for ellipse in sat_ellipses:
        ceny = ellipse[0][0]
        cenx = ellipse[0][1]
        cen = (round(ceny), round(cenx))
        minor_axis = min(ellipse[1][1], ellipse[1][0])

        if minor_axis > jump_data.min_sat_radius_extend:
            axis1 = ellipse[1][0] + jump_data.sat_expand
            axis2 = ellipse[1][1] + jump_data.sat_expand
            axis1 = min(axis1, jump_data.max_extended_radius)
            axis2 = min(axis2, jump_data.max_extended_radius)

            alpha = ellipse[2]

            indx, sat_ellipse = ellipse_subim(
                ceny, cenx, axis1, axis2, alpha, satcolor, (nrows, ncols))
            (iy1, iy2, ix1, ix2) = indx

            # Create another non-extended ellipse that is used to
            # create the persist_jumps for this integration. This
            # will be used to mask groups in subsequent integrations.

            is_sat = sat_ellipse == satcolor
            for i in range(grp, cube.shape[0]):
                cube[i][iy1:iy2, ix1:ix2][is_sat] = jump_data.fl_sat

            ax1, ax2 = (ellipse[1][0], ellipse[1][1])
            indx, persist_ellipse = ellipse_subim(
                ceny, cenx, ax1, ax2, alpha, satcolor, (nrows, ncols))
            (iy1, iy2, ix1, ix2) = indx

            persist_mask = persist_ellipse == satcolor
            persist_jumps[iy1:iy2, ix1:ix2][persist_mask] = jump_data.fl_jump

    return cube, persist_jumps


def ellipse_subim(ceny, cenx, axis1, axis2, alpha, value, shape):
    """Draw a filled ellipse in a small array at a given (returned) location
    Parameters
    ----------
    ceny : float
        Center of the ellipse in y (second axis of an image)
    cenx : float
        Center of the ellipse in x (first axis of an image)
    axis1 : float
        One (full) axis of the ellipse
    axis2 : float
        The other (full) axis of the ellipse
    alpha : float
        Angle (in degrees) between axis1 and x
    value : unsigned 8-bit integer
        Value to fill the image with
    shape : (int, int)
        The shape of the full 2D array into which the returned
        subimage should be placed.
    Returns
    -------
    indx : (int, int, int, int)
        Indices (iy1, iy2, ix1, ix2) such that
        fullimage[iy1:iy2, ix1:ix2] = subimage (see below)
    subimage : 2D 8-bit unsigned int array
        Small image containing the ellipse, goes into fullimage
        as described above.
    """
    yc, xc = round(ceny), round(cenx)

    # How big of a subarray do we need for the subimage?

    dn_over_2 = max(round(axis1/2), round(axis2/2)) + 2

    # Note that the convention between which index is x and which
    # is y is a little confusing here.  To cv.ellipse, the first
    # coordinate corresponds to the second Python index.  That is
    # why x and y are a bit mixed up below.

    ix1 = max(yc - dn_over_2, 0)
    ix2 = min(yc + dn_over_2 + 1, shape[1])
    iy1 = max(xc - dn_over_2, 0)
    iy2 = min(xc + dn_over_2 + 1, shape[0])

    image = np.zeros(shape=(iy2 - iy1, ix2 - ix1, 3), dtype=np.uint8)
    image = cv.ellipse(
        image,
        (yc - ix1, xc - iy1),
        (round(axis1 / 2), round(axis2 / 2)),
        alpha,
        0,
        360,
        (0, 0, value),
        -1,
    )

    # The last ("blue") part contains the filled ellipse that we want.
    subimage = image[:, :, 2]
    return (iy1, iy2, ix1, ix2), subimage



def extend_ellipses(
    gdq_cube, intg, grp, ellipses, jump_data,
    expansion=1.9, expand_by_ratio=True, num_grps_masked=1,
):
    """
    Extend the ellipses.

    Parameters
    ----------
    gdq_cube : ndarray
        Group DQ cube for an integration.  Modified in-place.

    intg : int
        The current integration.

    grp : int
        The current group.

    ellipses : cv.ellipse
        Ellipses for events.

    jump_data : JumpData
        Class containing parameters and methods to detect jumps.

    expansion : float
        The factor that increases the size of the snowball or enclosed ellipse.

    expand_by_ratio : bool
        Should the ellipse expansion be used?

    num_grps_masked : int
        The number of groups flagged.

    Returns
    -------
    gdq_cube : ndarray
        Computed 3-D group DQ array, modified in-place

    num_ellipses : int
        The number of ellipses passed in as a parameter.
    """
    # For a given DQ plane it will use the list of ellipses to create
    #  expanded ellipses of pixels with
    # the jump flag set.
    _, ngroups, nrows, ncols = gdq_cube.shape
    num_ellipses = len(ellipses)
    for ellipse in ellipses:
        ceny = ellipse[0][0]
        cenx = ellipse[0][1]
        axes = compute_axes(expand_by_ratio, ellipse, expansion, jump_data)

        alpha = ellipse[2]

        # Get the expanded ellipse in a subimage, along with the
        # indices that place this subimage within the full array.
        axis1 = axes[0]*2
        axis2 = axes[1]*2
        indx, jump_ellipse = ellipse_subim(
            ceny, cenx, axis1, axis2, alpha, jump_data.fl_jump, (nrows, ncols))
        (iy1, iy2, ix1, ix2) = indx
        
        # Propagate forward by num_grps_masked groups.

        for flg_grp in range(grp, min(grp + num_grps_masked + 1, ngroups)):

            # Only propagate the snowball forward to unsaturated pixels.

            sat_pix = gdq_cube[intg, flg_grp, iy1:iy2, ix1:ix2] & jump_data.fl_sat
            jump_ellipse[sat_pix == jump_data.fl_sat] = 0
            gdq_cube[intg, flg_grp, iy1:iy2, ix1:ix2] |= jump_ellipse

    return gdq_cube, num_ellipses

def find_ellipses(dqplane, bitmask, min_area):
    """
    Find ellipses based on DQ masks in bitmask.

    Parameters
    ----------
    dqplane : ndarray
        2D plane of an integration and group

    bitmask : uint8
        bitmask of DQ flags

    min_area : float
        The minimum area of saturated pixels at the center of a snowball. Only
        contours with area above the minimum will create snowballs.

    Returns 
    -------
    list of computed ellipses
    """
    # Using an input DQ plane this routine will find the groups of pixels with
    # at least the minimum
    # area and return a list of the minimum enclosing ellipse parameters.
    pixels = np.bitwise_and(dqplane, bitmask)
    contours, hierarchy = cv.findContours(pixels, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bigcontours = [con for con in contours if cv.contourArea(con) > min_area]

    # minAreaRect is used because fitEllipse requires 5 points and it is
    # possible to have a contour
    # with just 4 points.
    return [cv.minAreaRect(con) for con in bigcontours]


def make_snowballs(
    gdq, integration, group, jump_ellipses, sat_ellipses,
    next_sat_ellipses, jump_data, persist_jumps
):
    """
    Find snowballs.

    Parameter
    ---------
    gdq : ndarray
        The 4-D group DQ array.

    integration : int
        The current integration being used.

    group : int
        The current group being used.

    jump_ellipses : cv.ellipses
        Ellipses computed based on jump detection.

    sat_ellipses : cv.ellipses
        Ellipses computed based on saturation.

    next_sat_ellipses : cv.ellipses
        Ellipses computed based on saturation in the next group.

    jump_data : JumpData
        Class containing parameters and methods to detect jumps.

    persist_jumps : ndarray
        Zero array to be filled in.

    Returns
    -------
    gdq : ndarray
        The 4-D group DQ array.
        
    snowballs : list
        List of snowballs found.

    persist_jumps : ndarray
        Filled in array.
    """
    nints, ngroups, nrows, ncols = gdq.shape
    low_threshold = jump_data.edge_size
    high_threshold = max(0, nrows - jump_data.edge_size)

    # This routine will create a list of snowballs (ellipses) that have the
    # center of the saturation circle within the enclosing jump rectangle.
    snowballs = []
    for jump in jump_ellipses:
        if near_edge(jump, low_threshold, high_threshold):
            # if the jump ellipse is near the edge, do not require saturation in the
            # center of the jump ellipse
            snowballs.append(jump)
        else:
            for sat in sat_ellipses:
                if ((point_inside_ellipse(sat[0], jump) and jump not in snowballs)):
                    snowballs.append(jump)
            if group < ngroups - 1:
                # Is there saturation inside the jump in the next group?
                for next_sat in next_sat_ellipses:
                    if ((point_inside_ellipse(next_sat[0], jump)) and jump not in snowballs):
                        snowballs.append(jump)

    # extend the saturated ellipses that are larger than the min_sat_radius
    gdq[integration, :, :, :], persist_jumps[integration, :, :] = extend_saturation(
        gdq[integration, :, :, :],
        group,
        sat_ellipses,
        jump_data,
        persist_jumps[integration, :, :],
    )

    return gdq, snowballs, persist_jumps


def point_inside_ellipse(point, ellipse):
    """
    Detect if a point is inside an ellipse.

    Parameters
    ----------
    point : tuple
        Point of interest.

    ellipse : cv2.ellipse
        Ellipse for testing.

    Returns
    -------
    Boolean decision if point is in ellipse
    """
    delta_center = np.sqrt((point[0] - ellipse[0][0]) ** 2 + (point[1] - ellipse[0][1]) ** 2)
    major_axis = max(ellipse[1][0], ellipse[1][1])

    return delta_center < major_axis


def near_edge(jump, low_threshold, high_threshold):
    """
    Test whether the center of a jump is close to the edge of the detector.

    Jumps that are within the threshold will not require a saturated core
    since this may be off the detector

    Parameters
    ----------
    jump : cv2.ellipse
        Ellipse to check if close to detector edge.

    low_threshold :  int
        Low threshold distance from the edge of the detector where saturated cores are not
        required for snowball detection.

    high_threshold : 
        High threshold distance from the edge of the detector where saturated cores are not
        required for snowball detection.

    Returns
    -------
    Boolean : True if ellipse is close to the detector's edge.
    """
    return (
        jump[0][0] < low_threshold
        or jump[0][1] < low_threshold
        or jump[0][0] > high_threshold
        or jump[0][1] > high_threshold
    )


def find_faint_extended(
        indata, ingdq, pdq, readnoise_2d, jump_data, min_diffs_for_shower=10):
    """
    Flag groups based on showers detected.

    Parameters
    ----------
      indata : float, 4D array
          Science array.

      gdq : int, 2D array
          Group dq array.

      readnoise_2d : float, 2D array
          Readnoise for all pixels.

    Returns
    -------
    gdq : int, 4D array
        updated group dq array.

    number_ellipse : int
        Total number of showers detected.
    """
    log.info("Flagging Showers")
    refpix_flag = jump_data.fl_ref

    gdq = ingdq.copy()
    data = indata.copy()
    nints, ngrps, nrows, ncols = data.shape

    num_grps_donotuse = count_dnu_groups(gdq, jump_data)

    total_diffs = nints * (ngrps - 1) - num_grps_donotuse
    if total_diffs < min_diffs_for_shower:
        log.warning("Not enough differences for shower detections")
        return ingdq, 0

    data = nan_invalid_data(data, gdq, jump_data)

    refy, refx = np.where(pdq == refpix_flag)
    gdq[:, :, refy, refx] = jump_data.fl_dnu
    first_diffs = np.diff(data, axis=1)
    del data

    all_ellipses = []

    warnings.filterwarnings("ignore")

    read_noise_2 = readnoise_2d**2
    if nints >= jump_data.minimum_sigclip_groups:
        mean, median, stddev = stats.sigma_clipped_stats(first_diffs, sigma=5, axis=0)
    else:
        median_diffs = np.nanmedian(first_diffs, axis=(0, 1))
        sigma = np.sqrt(np.abs(median_diffs) + read_noise_2 / jump_data.nframes)

    for intg in range(nints):
        if nints < jump_data.minimum_sigclip_groups:
            # The difference from the median difference for each group
            ratio = diff_meddiff_int(intg, median_diffs, sigma, first_diffs)

        #  The convolution kernel creation
        ring_2D_kernel = Ring2DKernel(
                jump_data.extend_inner_radius, jump_data.extend_outer_radius)
        first_good_group = find_first_good_group(gdq[intg, :, :, :], jump_data.fl_dnu)
        for grp in range(first_good_group + 1, ngrps):
            if nints >= jump_data.minimum_sigclip_groups:
                ratio = diff_meddiff_grp(intg, grp, median, stddev, first_diffs)

            bigcontours = get_bigcontours(
                    ratio, intg, grp, gdq, pdq, jump_data, ring_2D_kernel)

            # get the minimum enclosing rectangle which is the same as the
            # minimum enclosing ellipse
            ellipses = [cv.minAreaRect(con) for con in bigcontours]
            image = np.zeros(shape=(nrows, ncols, 3), dtype=np.uint8)
            expand_by_ratio, expansion = True, 1.0
            image = process_ellipses(ellipses, image, expand_by_ratio, expansion, jump_data)

            if len(ellipses) > 0:
                # add all the showers for this integration to the list
                all_ellipses.append([intg, grp, ellipses])
                # Reset the warnings filter to its original state

    warnings.resetwarnings()
    total_showers = 0

    if all_ellipses:
        #  Now we actually do the flagging of the pixels inside showers.
        # This is deferred until all showers are detected. because the showers
        # can flag future groups and would confuse the detection algorithm if
        # we worked on groups that already had some flagged showers.
        for showers in all_ellipses:
            intg, grp, ellipses = showers[:3]
            total_showers += len(ellipses)
            gdq, num = extend_ellipses(
                gdq,
                intg,
                grp,
                ellipses,
                jump_data,
                expansion=jump_data.extend_ellipse_expand_ratio,
                expand_by_ratio=True,
                num_grps_masked=jump_data.grps_masked_after_shower,
            )

    gdq = max_flux_showers(jump_data, nints, indata, ingdq, gdq)

    return gdq, total_showers


def max_flux_showers(jump_data, nints, indata, ingdq, gdq):
    """
    Ensure that flagging showers didn't change final fluxes by more than allowed.

    Parameters
    ----------
    jump_data : JumpData
        Class containing parameters and methods to detect jumps.

    nints : int
        The number of integrations

    indata : ndarray
        The input data 4D float.

    ingdq : ndarray
        The input group DQ 4D uint8.

    gdq : ndarray
        The computed group DQ 4D uint8.

    Returns
    -------
    gdq : ndarray
        The computed group DQ 4D uint8.
    """
    # Ensure that flagging showers didn't change final fluxes by more than the allowed amount
    for intg in range(nints):
        # Consider DO_NOT_USE, SATURATION, and JUMP_DET flags
        invalid_flags = jump_data.fl_dnu | jump_data.fl_sat| jump_data.fl_jump

        # Approximate pre-shower rates
        tempdata = indata[intg, :, :, :].copy()
        # Ignore any groups flagged in the original gdq array
        tempdata[ingdq[intg, :, :, :] & invalid_flags != 0] = np.nan
        # Compute group differences
        diff = np.diff(tempdata, axis=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
            image1 = np.nanmean(diff, axis=0)
        del tempdata

        # Approximate post-shower rates
        tempdata = indata[intg, :, :, :].copy()
        # Ignore any groups flagged in the shower gdq array
        tempdata[gdq[intg, :, :, :] & invalid_flags != 0] = np.nan
        # Compute group differences
        diff = np.diff(tempdata, axis=0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="All-NaN")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
            image2 = np.nanmean(diff, axis=0)
        del tempdata

        # Revert the group flags to the pre-shower flags for any pixels whose rates
        # became NaN or changed by more than the amount reasonable for a real CR shower
        # Note that max_shower_amplitude should now be in DN/group not DN/s
        diff = np.abs(image1 - image2)
        indx = np.where((np.isfinite(diff) == False) | (diff > jump_data.max_shower_amplitude))
        gdq[intg, :, indx[0], indx[1]] = ingdq[intg, :, indx[0], indx[1]]

    return gdq


def count_dnu_groups(gdq, jump_data):
    """
    Count the number of groups are flagged as DO_NOT_USE.

    Parameters
    ----------
    gdq : ndarray
        The group DQ 4D uint8.
        
    jump_data : JumpData
        Class containing parameters and methods to detect jumps.

    Returns
    -------
    num_grps_donotuse : int
        The number of groups flagged as DO_NOT_USE.
    """
    nints, ngrps = gdq.shape[:2]
    num_grps_donotuse = 0
    for integ in range(nints):
        for grp in range(ngrps):
            if np.all(np.bitwise_and(gdq[integ, grp, :, :], jump_data.fl_dnu)):
                num_grps_donotuse += 1
    return num_grps_donotuse


def process_ellipses(ellipses, image, expand_by_ratio, expansion, jump_data):
    """
    Draw ellipses onto an image.

    Parameters
    ----------
    ellipses : list
        List of ellipses

    image : ndarray
        The image on which to draw the ellipses.

    expand_by_ratio : bool
        Should the ellipses be expanded?

    expansion : float
        The ellipse expansion factor

    jump_data : JumpData
        Class containing parameters and methods to detect jumps.

    Returns
    -------
    image : ndarray
        The image with ellipses drawn on it.
    """
    for ellipse in ellipses:
        ceny, cenx = ellipse[0][0], ellipse[0][1]
        cen = (round(ellipse[0][0]), round(ellipse[0][1]))
        axes = compute_axes(expand_by_ratio, ellipse, expansion, jump_data)
        alpha = ellipse[2]
        color = (0, 0, jump_data.fl_jump)
        image = cv.ellipse(image, cen, axes, alpha, 0, 360, color, -1)

    return image


def compute_axes(expand_by_ratio, ellipse, expansion, jump_data):
    """
    Expand the ellipse by the expansion factor.

    The number of pixels added to both axes is the number of pixels added
    to the minor axis. This prevents very large flagged ellipses with high
    axis ratio ellipses. The major and minor axis are not always the same
    index.  Therefore, we have to test to find which is actually the minor axis.

    Parameters
    ----------
    expand_by_ratio : bool
        Should the axes be expanded?

    ellipse : cv2.ellipse
        Ellipse to expand.

    expansion : float
        The factor that increases the size of the snowball or enclosed ellipse.

    jump_data : JumpData
        Class containing parameters and methods to detect jumps.

    Returns
    -------
    axes : tuple
        Expanded and rounded ellipse axes.
    """
    if expand_by_ratio:
        if ellipse[1][1] < ellipse[1][0]:
            axis1 = ellipse[1][0] + (expansion - 1.0) * ellipse[1][1]
            axis2 = ellipse[1][1] * expansion
        else:
            axis1 = ellipse[1][0] * expansion
            axis2 = ellipse[1][1] + (expansion - 1.0) * ellipse[1][0]
    else:
        axis1 = ellipse[1][0] + expansion
        axis2 = ellipse[1][1] + expansion
    axis1 = min(axis1, jump_data.max_extended_radius)
    axis2 = min(axis2, jump_data.max_extended_radius)

    return (round(axis1 / 2), round(axis2 / 2))


def get_bigcontours(ratio, intg, grp, gdq, pdq, jump_data, ring_2D_kernel):
    """Perform convolution to find contours larger than a minimum area.

    Parameters
    ----------
    ratio : ndarray

    intg : int
        Current integration

    grp : int
        Current group

    gdq : ndarray
        Group DQ array 4D uint8

    pdq : ndarray
        Pixel DQ array 2D uint32

    jump_data : JumpData
        Class containing parameters and methods to detect jumps.

    ring_2D_kernel : astropy.convolution.Ring2DKernel
        2D Ring filter kernel

    Returns
    -------
    bigcontours : list 
        list of OpenCV countours
    """
    masked_ratio = ratio[grp - 1].copy()
    jump_flag = jump_data.fl_jump
    sat_flag = jump_data.fl_sat
    dnu_flag = jump_data.fl_dnu

    #  mask pixels that are already flagged as jump, sat, or dnu
    combined_pixel_mask = np.bitwise_or(gdq[intg, grp, :, :], pdq[:, :])

    jump_sat_or_dnu = np.bitwise_and(combined_pixel_mask, jump_flag|sat_flag|dnu_flag) != 0
    masked_ratio[jump_sat_or_dnu] = np.nan
    
    kernel = ring_2D_kernel.array
    
    # Equivalent to but faster than
    # masked_smoothed_ratio = convolve(masked_ratio, ring_2D_kernel, preserve_nan=True)
    
    masked_smoothed_ratio = convolve_fast(masked_ratio, kernel)

    extended_emission = (masked_smoothed_ratio > jump_data.extend_snr_threshold).astype(np.uint8)

    #  find the contours of the extended emission
    contours, hierarchy = cv.findContours(
            extended_emission, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    #  get the contours that are above the minimum size
    bigcontours = [con for con in contours if cv.contourArea(con) > jump_data.extend_min_area]
    return bigcontours 



def convolve_fast(inarray, kernel, copy=False):
    """Convolve an array with a kernel, interpolating over NaNs.
    Faster version of astropy.convolution.convolve(preserve_nan=True)
    Parameters
    ----------
    inarray : 2D array of floats
        Array for convolution
    kernel : 2D array of floats
        Convolution kernel.  Both dimensions must be odd.
    copy : bool
        Make a copy of inarray to avoid modifying NaN values.  Default False.
    Returns
    -------
    convolved_array : 2D array of floats
        Convolution of inarray and kernel, interpolating over NaNs.
    """

    # We will mask nan pixels by setting them to zero.  We
    # will convolve by our kernel, then divide by the weight
    # given by the valid pixels convolved with the kernel in
    # order to normalize.  Finally, we will reset the
    # initially nan pixels to nan.
    #
    # This function is equivalent to
    # convolved_array = astropy.convolution.convolve(inarray, kernel, preserve_nan=True)
    # but runs in about half the time.

    if copy:
        array = inarray.copy()
    else:
        array = inarray

    good = np.isfinite(array)
    array[~good] = 0

    convolved_array = signal.oaconvolve(array, kernel, mode='same')

    # Embed the flag in a larger array to reproduce the behavior at
    # the edge with a fill value of zero.

    padded_good_arr = np.ones((good.shape[0] + kernel.shape[0] - 1,
                               good.shape[1] + kernel.shape[1] - 1))
    n = kernel.shape[0]//2
    padded_good_arr[n:-n, n:-n] = good
    norm = signal.oaconvolve(padded_good_arr, kernel, mode='valid')

    # Avoid dividing by a tiny number due to roundoff error.

    good &= norm > 1e-3*np.mean(kernel)
    convolved_array /= norm

    # Replace NaNs

    convolved_array[~good] = np.nan

    return convolved_array


def diff_meddiff_int(intg, median_diffs, sigma, first_diffs_masked):
    """
    Compute the SNR ratio of each difference.

    Parameters
    ----------
    intg : int
        Current intregration

    median_diffs : ndarray
        Median of differences in integration

    sigma : ndarray
        Weighting.

    first_diffs_masked : ndarray
        Masked first differences.

    Returns
    -------
    ratio : ndarray
        SNR ratio
    """

    e_jump = first_diffs_masked[intg] - median_diffs[np.newaxis, :, :]

    # SNR ratio of each diff.
    ratio = np.abs(e_jump) / sigma[np.newaxis, :, :]

    return ratio


def diff_meddiff_grp(intg, grp, median, stddev, first_diffs_masked):
    """
    Find the median difference group.

    Parameters
    ----------
    intg : int
        Current intregration

    grp : int
        Current group

    median : float
        Median computed during sigma clipping.

    stddev : float
        Standard deviation computed during sigma clipping.

    first_diffs_masked : ndarray
        Masked first differences.

    Returns
    -------
    ratio : ndarray
        SNR ratio
    """
    median_diffs = median[grp - 1]
    sigma = stddev[grp - 1]

    # The difference from the median difference for each group
    e_jump = first_diffs_masked[intg] - median_diffs[np.newaxis, :, :]

    # SNR ratio of each diff.
    ratio = np.abs(e_jump) / sigma[np.newaxis, :, :]

    return ratio


def nan_invalid_data(data, gdq, jump_data):
    """
    Mark flagged data as invalid by setting the science data to NaN.

    Parameters
    ----------
    data : ndarray
        Science data 4D float

    gdq : ndarray
        Group DQ 4D uint8

    jump_data : JumpData
        Class containing parameters and methods to detect jumps.

    Returns
    -------
    data : ndarray
        NaN'd cience data 4D float
    """
    jump_dnu_flag = jump_data.fl_jump + jump_data.fl_dnu
    sat_dnu_flag = jump_data.fl_sat + jump_data.fl_dnu
    data[gdq == jump_dnu_flag] = np.nan
    data[gdq == sat_dnu_flag] = np.nan
    data[gdq == jump_data.fl_sat] = np.nan
    data[gdq == jump_data.fl_jump] = np.nan
    data[gdq == jump_data.fl_dnu] = np.nan

    return data


def find_first_good_group(int_gdq, do_not_use):
    """
    Find first good group.

    Parameters
    ----------
    int_gdq : ndarray
        Group DQ for an integration 3D uint8.

    do_not_use : int
        The DO_NOT_USE flag.

    Returns
    -------
    first_good_group : ndarray
        The first good group of the pixel integration.
    """
    ngrps = int_gdq.shape[0]
    skip_grp = True
    first_good_group = 0
    for grp in range(ngrps):
        mask = np.bitwise_and(int_gdq[grp], do_not_use)
        skip_grp = np.all(mask)
        if not skip_grp:
            first_good_group = grp
            break

    return first_good_group


def calc_num_slices(n_rows, max_cores, max_available):
    """
    Compute the number of data slices needed for multiprocessesing.

    Parameters
    ----------
    n_rows : int
        The number of rows of the science data.

    max_cores : str
        The number of processes requested.

    max_available ; int
        The maximum number of CPU cores available.

    Returns
    -------
    The number of slices to slice the data into.
    """
    n_slices = 1
    if max_cores.isnumeric():
        n_slices = int(max_cores)
    elif max_cores.lower() == "none" or max_cores.lower() == "one":
        n_slices = 1
    elif max_cores == "quarter":
        n_slices = max_available // 4 or 1
    elif max_cores == "half":
        n_slices = max_available // 2 or 1
    elif max_cores == "all":
        n_slices = max_available

    # Make sure we don't have more slices than rows or available cores.
    return min([n_rows, n_slices, max_available])
