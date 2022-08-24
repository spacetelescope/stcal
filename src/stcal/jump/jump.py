import time
import logging
import numpy as np
import cv2 as cv
from astropy.io import fits

from . import twopoint_difference as twopt
from . import constants

import multiprocessing

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def detect_jumps(frames_per_group, data, gdq, pdq, err,
                 gain_2d, readnoise_2d, rejection_thresh,
                 three_grp_thresh, four_grp_thresh, max_cores, max_jump_to_flag_neighbors,
                 min_jump_to_flag_neighbors, flag_4_neighbors, dqflags,
                 after_jump_flag_dn1=0.0,
                 after_jump_flag_n1=0,
                 after_jump_flag_dn2=0.0,
                 after_jump_flag_n2=0,
                 min_sat_area=1,
                 min_jump_area=6,
                 max_offset=4,
                 expand_factor=1.9,
                 use_ellipses=False,
                 sat_required_snowball=True):
    """
    This is the high-level controlling routine for the jump detection process.
    It loads and sets the various input data and parameters needed by each of
    the individual detection methods and then calls the detection methods in
    turn.

    Note that the detection methods are currently set up on the assumption
    that the input science and error data arrays will be in units of
    electrons, hence this routine scales those input arrays by the detector
    gain. The methods assume that the read noise values will be in units
    of DN.

    The gain is applied to the science data and error arrays using the
    appropriate instrument- and detector-dependent values for each pixel of an
    image.  Also, a 2-dimensional read noise array with appropriate values for
    each pixel is passed to the detection methods.

    Parameters
    ----------
    frames_per_group : int
        number of frames per group

    data : float, 4D array
        science array

    gdq : int, 4D array
        group dq array

    pdq : int, 2D array
        pixelg dq array

    err : float, 4D array
        error array

    gain_2d : float, 2D array
        gain for all pixels

    readnoise_2d : float, 2D array
        readnoise for all pixels

    rejection_thresh : float
        The 'normal' cosmic ray sigma rejection threshold for ramps with more
        than 4 groups

    three_grp_thresh : float
        cosmic ray sigma rejection threshold for ramps having 3 groups

    four_grp_thresh : float
        cosmic ray sigma rejection threshold for ramps having 4 groups

    max_cores: str
        Maximum number of cores to use for multiprocessing. Available choices
        are 'none' (which will create one process), 'quarter', 'half', 'all'
        (of availble cpu cores).

    max_jump_to_flag_neighbors : float
        value in units of sigma that sets the upper limit for flagging of
        neighbors. Any jump above this cutoff will not have its neighbors
        flagged.

    min_jump_to_flag_neighbors : float
        value in units of sigma that sets the lower limit for flagging of
        neighbors (marginal detections). Any primary jump below this value will
        not have its neighbors flagged.

    flag_4_neighbors): bool
        if set to True (default is True), it will cause the four perpendicular
        neighbors of all detected jumps to also be flagged as a jump.

    dqflags: dict
        A dictionary with at least the following keywords:
        DO_NOT_USE, SATURATED, JUMP_DET, NO_GAIN_VALUE, GOOD

    after_jump_flag_dn1 : float
        Jumps with amplitudes above the specified DN value will have subsequent
        groups flagged with the number determined by the after_jump_flag_n1

    after_jump_flag_n1 : int
        Gives the number of groups to flag after jumps with DN values above that
        given by after_jump_flag_dn1

    after_jump_flag_dn2 : float
        Jumps with amplitudes above the specified DN value will have subsequent
        groups flagged with the number determined by the after_jump_flag_n2

    after_jump_flag_n2 : int
        Gives the number of groups to flag after jumps with DN values above that
        given by after_jump_flag_dn2

    Returns
    -------
    gdq : int, 4D array
        updated group dq array

    pdq : int, 2D array
        updated pixel dq array
    """
    constants.update_dqflags(dqflags)  # populate dq flags
    sat_flag = dqflags["SATURATED"]
    jump_flag = dqflags["JUMP_DET"]
    # Flag the pixeldq where the gain is <=0 or NaN so they will be ignored
    wh_g = np.where(gain_2d <= 0.)
    if len(wh_g[0] > 0):
        pdq[wh_g] = np.bitwise_or(pdq[wh_g], dqflags["NO_GAIN_VALUE"])
        pdq[wh_g] = np.bitwise_or(pdq[wh_g], dqflags["DO_NOT_USE"])

    wh_g = np.where(np.isnan(gain_2d))
    if len(wh_g[0] > 0):
        pdq[wh_g] = np.bitwise_or(pdq[wh_g], dqflags["NO_GAIN_VALUE"])
        pdq[wh_g] = np.bitwise_or(pdq[wh_g], dqflags["DO_NOT_USE"])

    # Apply gain to the SCI, ERR, and readnoise arrays so they're in units
    # of electrons
    data *= gain_2d
    err *= gain_2d
    readnoise_2d *= gain_2d
    # also apply to the after_jump thresholds
    after_jump_flag_e1 = after_jump_flag_dn1 * gain_2d
    after_jump_flag_e2 = after_jump_flag_dn2 * gain_2d

    # Apply the 2-point difference method as a first pass
    log.info('Executing two-point difference method')
    start = time.time()

    # Set parameters of input data shape
    n_rows = data.shape[-2]
    n_cols = data.shape[-1]
    n_groups = data.shape[1]
    n_ints = data.shape[0]

    row_above_gdq = np.zeros((n_ints, n_groups, n_cols), dtype=np.uint8)
    previous_row_above_gdq = np.zeros((n_ints, n_groups, n_cols),
                                      dtype=np.uint8)
    row_below_gdq = np.zeros((n_ints, n_groups, n_cols), dtype=np.uint8)

    # figure out how many slices to make based on 'max_cores'

    max_available = multiprocessing.cpu_count()
    if max_cores.lower() == 'none':
        n_slices = 1
    elif max_cores == 'quarter':
        n_slices = max_available // 4 or 1
    elif max_cores == 'half':
        n_slices = max_available // 2 or 1
    elif max_cores == 'all':
        n_slices = max_available

    if n_slices == 1:
        gdq, row_below_dq, row_above_dq = \
            twopt.find_crs(data, gdq, readnoise_2d, rejection_thresh,
                           three_grp_thresh, four_grp_thresh, frames_per_group,
                           flag_4_neighbors, max_jump_to_flag_neighbors,
                           min_jump_to_flag_neighbors, dqflags,
                           after_jump_flag_e1=after_jump_flag_e1,
                           after_jump_flag_n1=after_jump_flag_n1,
                           after_jump_flag_e2=after_jump_flag_e2,
                           after_jump_flag_n2=after_jump_flag_n2)

        elapsed = time.time() - start
    else:
        yinc = int(n_rows / n_slices)
        slices = []
        # Slice up data, gdq, readnoise_2d into slices
        # Each element of slices is a tuple of
        # (data, gdq, readnoise_2d, rejection_thresh, three_grp_thresh,
        #  four_grp_thresh, nframes)

        # must copy arrays here, find_crs will make copies but if slices
        # are being passed in for multiprocessing then the original gdq will be
        # modified unless copied beforehand
        gdq = gdq.copy()
        data = data.copy()
        copy_arrs = False   # we dont need to copy arrays again in find_crs

        for i in range(n_slices - 1):
            slices.insert(i, (data[:, :, i * yinc:(i + 1) * yinc, :],
                              gdq[:, :, i * yinc:(i + 1) * yinc, :],
                              readnoise_2d[i * yinc:(i + 1) * yinc, :],
                              rejection_thresh, three_grp_thresh, four_grp_thresh,
                              frames_per_group, flag_4_neighbors,
                              max_jump_to_flag_neighbors,
                              min_jump_to_flag_neighbors, dqflags,
                              after_jump_flag_e1, after_jump_flag_n1,
                              after_jump_flag_e2, after_jump_flag_n2,
                              copy_arrs))

        # last slice get the rest
        slices.insert(n_slices - 1, (data[:, :, (n_slices - 1) * yinc:n_rows, :],
                                     gdq[:, :, (n_slices - 1) * yinc:n_rows, :],
                                     readnoise_2d[(n_slices - 1) * yinc:n_rows, :],
                                     rejection_thresh, three_grp_thresh,
                                     four_grp_thresh, frames_per_group,
                                     flag_4_neighbors, max_jump_to_flag_neighbors,
                                     min_jump_to_flag_neighbors, dqflags,
                                     after_jump_flag_e1, after_jump_flag_n1,
                                     after_jump_flag_e2, after_jump_flag_n2,
                                     copy_arrs))
        log.info("Creating %d processes for jump detection " % n_slices)
        pool = multiprocessing.Pool(processes=n_slices)
        # Starts each slice in its own process. Starmap allows more than one
        # parameter to be passed.
        real_result = pool.starmap(twopt.find_crs, slices)
        pool.close()
        pool.join()
        k = 0

        # Reconstruct gdq, the row_above_gdq, and the row_below_gdq from the
        # slice result
        for resultslice in real_result:

            if len(real_result) == k + 1:  # last result
                gdq[:, :, k * yinc:n_rows, :] = resultslice[0]
            else:
                gdq[:, :, k * yinc:(k + 1) * yinc, :] = resultslice[0]
            row_below_gdq[:, :, :] = resultslice[1]
            row_above_gdq[:, :, :] = resultslice[2]
            if k != 0:
                # For all but the first slice, flag any CR neighbors in the top
                # row of the previous slice and flag any neighbors in the
                # bottom row of this slice saved from the top of the previous
                # slice
                gdq[:, :, k * yinc - 1, :] = \
                    np.bitwise_or(gdq[:, :, k * yinc - 1, :],
                                  row_below_gdq[:, :, :])
                gdq[:, :, k * yinc, :] = \
                    np.bitwise_or(gdq[:, :, k * yinc, :],
                                  previous_row_above_gdq[:, :, :])

            # save the neighbors to be flagged that will be in the next slice
            previous_row_above_gdq = row_above_gdq.copy()
            k += 1
        elapsed = time.time() - start

        flag_large_events(data, gdq, jump_flag, sat_flag, min_sat_area=min_sat_area,
                          min_jump_area=min_jump_area, max_offset=max_offset,
                          expand_factor=expand_factor, use_ellipses=use_ellipses,
                          sat_required_snowball=sat_required_snowball)

    elapsed = time.time() - start
    log.info('Total elapsed time = %g sec' % elapsed)

    # Back out the applied gain to the SCI, ERR, and readnoise arrays so they're
    #    back in units of DN
    data /= gain_2d
    err /= gain_2d
    readnoise_2d /= gain_2d

    # Return the updated data quality arrays
    return gdq, pdq


def flag_large_events(gdq, jump_flag, sat_flag, min_sat_area=1,
                          min_jump_area=6, max_offset=4,
                          expand_factor=1.9, use_ellipses=False,
                          sat_required_snowball=True):
    for integration in range(gdq.shape[0]):
        for group in range(gdq.shape[1]):
            if use_ellipses:
                jump_ellipses = find_ellipses(gdq[integration, group, :, :], jump_flag, min_jump_area)
                gdq[integration, group, :, :] = extend_ellipses(gdq[integration, group, :, :],
                                                                jump_ellipses, sat_flag, jump_flag,
                                                                expansion=expand_factor)
            else:
                sat_circles = find_circles(gdq[integration, group, :, :], sat_flag, min_sat_area)
                jump_circles = find_circles(gdq[integration, group, :, :], jump_flag, min_jump_area)
                if sat_required_snowball:
                    snowballs = make_snowballs(jump_circles, sat_circles, max_offset)
                else:
                    snowballs = jump_circles
                gdq[integration, group, :, :] = extend_snowballs(gdq[integration, group, :, :],
                                                                 snowballs, expansion=expand_factor)


def extend_snowballs(plane, snowballs, sat_flag, jump_flag, expansion=1.5):
    for snowball in snowballs:
        jump_radius = snowball[1]
        jump_center = snowball[0]
        xcen = jump_center[0]
        ycen = jump_center[1]
        extend_radius = jump_radius * expansion
        xmin = int(xcen - extend_radius)
        xmax = int(xcen + extend_radius)
        ymin = int(ycen - extend_radius)
        ymax = int(ycen + extend_radius)
        for x in range(max(0, xmin), min(plane.shape[1], xmax)):
            for y in range(max(0, ymin), min(plane.shape[1], ymax)):
                if np.sqrt((y - ycen) ** 2 + (x - xcen) ** 2) < extend_radius:
                    if not np.bitwise_and(plane[y, x], sat_flag):
                        plane[y, x] = np.bitwise_or(plane[y, x], jump_flag)
    return plane


def extend_ellipses(plane, ellipses, sat_flag, jump_flag, expansion=1.1):
    image = np.zeros(shape=(plane.shape[0], plane.shape[1], 3), dtype=np.uint8)
    print(len(ellipses))
    for ellipse in ellipses:
        ceny = ellipse[0][0]
        cenx = ellipse[0][1]
        majaxis = ellipse[1][0] * expansion
        minaxis = ellipse[1][1] * expansion
        alpha = ellipse[2]
        image = cv.ellipse(image, (round(ceny), round(cenx)), (round(majaxis/ 2),
                           round(minaxis/ 2)), alpha,
                           0, 360, (0, 0, 4), -1)
        jump_ellipse = image[:, :, 2]
#        fits.writeto("jump_ellipse.fits", jump_ellipse, overwrite=True)
        pixels = np.where(jump_ellipse == jump_flag)
        min_y = np.min(pixels[0])
        min_x = np.min(pixels[1])
        max_y = np.max(pixels[0])
        max_x = np.max(pixels[1])
        for x in range(min_x, max_x+1):
            for y in range(min_y, max_y+1):
                if not np.bitwise_and(plane[y, x], sat_flag):
                    plane[y, x] = np.bitwise_or(jump_ellipse[y, x], plane[y, x])
    return plane


def find_circles(dqplane, bitmask, min_area):
    pixels = np.bitwise_and(dqplane, bitmask)
    contours, hierarchy = cv.findContours(pixels, cv.RETR_EXTERNAL, 2)
    bigcontours = [con for con in contours if cv.contourArea(con) > min_area]
    circles = [cv.minEnclosingCircle(con) for con in bigcontours]
    return circles


def find_ellipses(dqplane, bitmask, min_area):
    pixels = np.bitwise_and(dqplane, bitmask)
    contours, hierarchy = cv.findContours(pixels, cv.RETR_EXTERNAL, 2)
    bigcontours = [con for con in contours if cv.contourArea(con) > min_area]
    ellipses = [cv.fitEllipse(con) for con in bigcontours]
    return ellipses


def make_snowballs(jump_circles, sat_circles, max_offset):
    snowballs = []
    for jump in jump_circles:
        for sat in sat_circles:
            distance = np.sqrt((jump[0][0] - sat[0][0]) ** 2 + (jump[0][1] - sat[0][1]) ** 2)
            if distance < max_offset:
                if jump not in snowballs:
                    snowballs.append(jump)
    return snowballs
