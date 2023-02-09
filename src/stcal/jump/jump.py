import time
import logging
import warnings
from astropy.io import fits
from astropy.convolution import Ring2DKernel
from astropy.convolution import convolve
import numpy as np
from . import twopoint_difference as twopt
from . import constants

import multiprocessing

try:
    import cv2 as cv

    OPENCV_INSTALLED = True
except ImportError:
    OPENCV_INSTALLED = False
    warnings.warn('Could not import `opencv-python`; '
                  'certain snowball detection and usage of ellipses will be inoperable')

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
                 min_jump_area=5,
                 expand_factor=2.0,
                 use_ellipses=False,
                 sat_required_snowball=True,
                 expand_large_events=True,
                 edge_size = 25):
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

    min_sat_area : float
        The minimum area of saturated pixels at the center of a snowball. Only contours
        with area above the minimum will create snowballs.

    min_jump_area : float
        The minimum contour area to trigger the creation of enclosing ellipses or circles.

    expand_factor : float
        The factor that is used to increase the size of the enclosing circle/ellipse jump
        flagged pixels.

    use_ellipses : bool
        If true the minimum enclosing ellipse will be created for jump regions that meet the area
        requirement. This is best for MIRI showers which are only rarely circular. For the NIR detectors
        this should set to False to force circles to be used.

    sat_required_snowball : bool
        If true there must be a saturation circle within the radius of the jump circle to trigger
        the creation of a snowball. All true snowballs appear to have at least one saturated pixel.

    expand_large_events : bool
        When True this triggers the flagging of snowballs and showers for NIR and MIRI detectors. If
        set to False nether type of extended flagging occurrs.

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

        #  This is the flag that controls the flagging of either snowballs or showers.
        if expand_large_events:
            flag_large_events(gdq, jump_flag, sat_flag, min_sat_area=min_sat_area,
                              min_jump_area=min_jump_area,
                              expand_factor=expand_factor, use_ellipses=use_ellipses,
                              sat_required_snowball=sat_required_snowball)

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
#        fits.writeto("input_jump_cube.fits", gdq, overwrite=True)
        #  This is the flag that controls the flagging of either snowballs or showers.
        if expand_large_events:
            flag_large_events(gdq, jump_flag, sat_flag, min_sat_area=min_sat_area,
                              min_jump_area=min_jump_area,
                              expand_factor=expand_factor, use_ellipses=use_ellipses,
                              sat_required_snowball=sat_required_snowball,
                              edge_size=edge_size)
        fits.writeto("input_jump_cube_after_le.fits", gdq, overwrite=True)
        gdq = find_faint_extended(data, gdq, readnoise_2d, frames_per_group, snr_threshold=1.2,
                            min_shower_area=90, inner=1,
                            outer=2.6, sat_flag=sat_flag, jump_flag=jump_flag, ellipse_expand=1.1)
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
                      min_jump_area=6,
                      expand_factor=2.0, use_ellipses=False,
                      sat_required_snowball=True, min_sat_radius_extend=2.5, sat_expand=2,
                      edge_size=25):
    """
    This routine controls the creation of expanded regions that are flagged as jumps. These are called
    snowballs for the NIR and are almost always circular with a saturated core. For MIRI they are better
    fit with ellipses.

    :param gdq: The group DQ cube for all integrations
    :param jump_flag: The bit value that represents jump
    :param sat_flag:  The bit value that represents saturation
    :param min_sat_area:  The minimum area of saturated pixels within the jump circle to trigger the
                        creation of a snowball.
    :param min_jump_area:  The minimum area of a contour to cause the creation of the minimum
                        enclosing ellipse or circle
    :param expand_factor: The factor that increases the size of the snowball or enclosing ellipse.
    :param use_ellipses:  Use ellipses rather than circles (better for MIRI)
    :param sat_required_snowball: Require that there is a saturated pixel within the radius of the jump
                        circle to trigger the formation of a snowball.
    :return: none
    """

    log.info('TEST Flagging large events (snowballs, showers).')

    n_showers_grp = []
    n_showers_grp_ellipse = []
    fits.writeto("input_jump_cube.fits", gdq, overwrite=True)
    for integration in range(gdq.shape[0]):
        for group in range(1, gdq.shape[1]):
            if (group//10) * 10 == group:
                print("Grp", group)
            else:
                print("Grp", group, end=" ")
            if use_ellipses:
                new_flagged_pixels = 1.0*gdq[integration, group, :, :] - 1.0*gdq[integration, group - 1, :, :]
                new_flagged_pixels[new_flagged_pixels < 0] = 0
                fits.writeto('new_flagged_pixels.fits', new_flagged_pixels, overwrite=True)
                jump_ellipses = find_ellipses(new_flagged_pixels.astype('uint8'), jump_flag, min_jump_area)
                n_showers_grp_ellipse.append(len(jump_ellipses))
                gdq[integration, group, :, :], num_events = \
                    extend_ellipses(gdq, integration, group, jump_ellipses, sat_flag, jump_flag,
                                    expansion=expand_factor)
            else:
                current_gdq = 1.0 * gdq[integration, group, :, :]
                prev_gdq = 1.0 * gdq[integration, group - 1, :, :]
                diff_gdq = 1.0 * current_gdq - prev_gdq
                diff_gdq[diff_gdq != sat_flag] = 0
                new_sat = diff_gdq.astype('uint8')
#                fits.writeto("diff_gdq.fits", diff_gdq, overwrite=True)
#                fits.writeto('current_gdq.fits', current_gdq, overwrite = True)
#                fits.writeto('prev_gdq.fits', prev_gdq, overwrite=True)
#                new_flagged_pixels = gdq[integration, group, :, :] - gdq[integration, group - 1, :, :]
                fits.writeto("new_sat.fits", new_sat, overwrite=True)
                # find the ellipse parameters for newly saturated pixels
                sat_ellipses = find_ellipses(new_sat, sat_flag, min_sat_area)
                # expand the larger saturated cores to deal with the charge migration from the
                # saturated cores.
#                gdq[integration, , :, :] = extend_saturation(gdq[integration, :, :, :],
                gdq[integration, :, :, :] = extend_saturation(gdq[integration, :, :, :],
                                                              group, sat_ellipses, sat_flag, jump_flag,
                                                              min_sat_radius_extend, expansion=sat_expand)
#               fits.writeto("after_extend_large_events.fits", gdq, overwrite=True)
                #  recalculate the newly flagged pixels after the expansion of saturation
                current_gdq = 1.0 * gdq[integration, group, :, :]
                prev_gdq = 1.0 * gdq[integration, group - 1, :, :]
                diff_gdq = 1.0 * current_gdq - prev_gdq
                diff_gdq[diff_gdq < 0] = 0
                new_sat = diff_gdq.astype('uint8')
#                fits.writeto("diff_gdq2.fits", diff_gdq, overwrite=True)
#                fits.writeto('current_gdq2.fits', current_gdq, overwrite=True)
#                fits.writeto('prev_gdq2.fits', prev_gdq, overwrite=True)
                # find all the newly saturated pixel
                sat_pixels = np.bitwise_and(diff_gdq.astype('uint8'), sat_flag)
                saty, satx = np.where(sat_pixels == sat_flag)
                only_jump = diff_gdq.copy()
                fits.writeto("onlyjump.fits", only_jump, overwrite=True)
                # reset the saturated pixel to be jump to allow the jump circles to have the
                # central saturated region set to "jump" instead of "saturation".
                only_jump[saty, satx] = jump_flag
                fits.writeto("onlyjump2.fits", only_jump, overwrite=True)
 #               only_jump_cube[integration, group, :, :] = only_jump
                jump_ellipses = find_ellipses(only_jump.astype('uint8'), jump_flag, min_jump_area)
                if sat_required_snowball:
                    low_threshold = edge_size
                    high_threshold = gdq.shape[2] - edge_size
                    snowballs = make_snowballs(jump_ellipses, sat_ellipses, low_threshold, high_threshold)
                else:
                    snowballs = jump_ellipses
                n_showers_grp.append(len(snowballs))
                gdq, num_events = extend_ellipses(gdq, integration, group, snowballs, sat_flag,
                                                  jump_flag, expansion=expand_factor)
#                fits.writeto("final_gdq.fits", gdq[integration, group,:, :], overwrite=True)
        fits.writeto("last_gdq_inside.fits", gdq, overwrite=True)
        if use_ellipses:
            if np.all(np.array(n_showers_grp_ellipse) == 0):
                log.info(f'No showers found in integration {integration}.')
            else:
                log.info(f' In integration {integration}, number of' +
                         f'showers in each group = {n_showers_grp_ellipse}')
        else:
            if np.all(np.array(n_showers_grp) == 0):
                log.info(f'No snowballs found in integration {integration}.')
            else:
                log.info(f' In integration {integration}, number of snowballs ' +
                         f'in each group = {n_showers_grp}')


def extend_snowballs(plane, snowballs, sat_flag, jump_flag, expansion=1.5):
    # For a given DQ plane it will use the list of snowballs to create expanded circles of pixels with
    # the jump flag set.
    image = np.zeros(shape=(plane.shape[0], plane.shape[1], 3), dtype=np.uint8)
    num_circles = len(snowballs)
    sat_pix = np.bitwise_and(plane, sat_flag)
    for snowball in snowballs:
        jump_radius = snowball[1]
        jump_center = snowball[0]
        cenx = jump_center[1]
        ceny = jump_center[0]
        extend_radius = round(jump_radius * expansion)
        image = cv.circle(image, (round(ceny), round(cenx)), extend_radius, (0, 0, 4), -1)
        jump_circle = image[:, :, 2]
        saty, satx = np.where(sat_pix == sat_flag)
        jump_circle[saty, satx] = 0
        plane = np.bitwise_or(plane, jump_circle)

    return plane, num_circles


def extend_saturation(cube, grp, sat_ellipses, sat_flag, jump_flag,
                      min_sat_radius_extend, expansion=2):
    image = np.zeros(shape=(cube.shape[1], cube.shape[2], 3), dtype=np.uint8)
    jump_pix = np.bitwise_and(cube[grp, :, :], jump_flag)
    outcube = cube.copy()
    for ellipse in sat_ellipses:
        ceny = ellipse[0][0]
        cenx = ellipse[0][1]
        minor_axis = min(ellipse[1][1], ellipse[1][0])
#        print("Grp", grp, " radius ", minor_axis, "count", count, "center", ellipse[0])
        if minor_axis > min_sat_radius_extend:
            axis1 = ellipse[1][0] + expansion
            axis2 = ellipse[1][1] + expansion
            alpha = ellipse[2]
            image = cv.ellipse(image, (round(ceny), round(cenx)), (round(axis1 + 0.5),
                               round(axis2 + 0.5)), alpha, 0, 360, (0, 0, 22), -1)
            sat_ellipse = image[:, :, 2]
            saty, satx = np.where(sat_ellipse == 22)
            outcube[grp:, saty, satx] = sat_flag
    return outcube


def extend_ellipses(gdq_cube, intg, grp, ellipses, sat_flag, jump_flag, expansion=1.9, expand_by_ratio=True,
                    num_grps_masked=1):
    # For a given DQ plane it will use the list of ellipses to create expanded ellipses of pixels with
    # the jump flag set.
    plane = gdq_cube[intg, grp, :, :]
    max_grp = gdq_cube.shape[1]
    image = np.zeros(shape=(plane.shape[0], plane.shape[1], 3), dtype=np.uint8)
    num_ellipses = len(ellipses)
    sat_pix = np.bitwise_and(plane, sat_flag)
    for ellipse in ellipses:
        ceny = ellipse[0][0]
        cenx = ellipse[0][1]
        # Expand the ellipse by the expansion factor. The number of pixels added to both axes is
        # the number of pixels added to the minor axis. This prevents very large flagged ellipses
        # with high axis ratio ellipses. The major and minor axis are not always the same index.
        # Therefore, we have to test to find which is actually the minor axis.
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
        alpha = ellipse[2]
        image = cv.ellipse(image, (round(ceny), round(cenx)), (round(axis1 / 2),
                           round(axis2 / 2)), alpha, 0, 360, (0, 0, jump_flag), -1)
        jump_ellipse = image[:, :, 2]
        last_grp = min(grp + num_grps_masked, gdq_cube.shape[1])
        for flg_grp in range(grp, last_grp):
            sat_pix = np.bitwise_and(gdq_cube[intg, flg_grp, :, :], sat_flag)
            saty, satx = np.where(sat_pix == sat_flag)
            jump_ellipse[saty, satx] = 0
            gdq_cube[intg, flg_grp, :, :] = np.bitwise_or(gdq_cube[intg, flg_grp, :, :], jump_ellipse)
    return gdq_cube, num_ellipses


def find_circles(dqplane, bitmask, min_area):
    # Using an input DQ plane this routine will find the groups of pixels with at least the minimum
    # area and return a list of the minimum enclosing circle parameters.
    pixels = np.bitwise_and(dqplane, bitmask)
    contours, hierarchy = cv.findContours(pixels, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bigcontours = [con for con in contours if cv.contourArea(con) >= min_area]
    circles = [cv.minEnclosingCircle(con) for con in bigcontours]
    return circles


def find_ellipses(dqplane, bitmask, min_area):
    # Using an input DQ plane this routine will find the groups of pixels with at least the minimum
    # area and return a list of the minimum enclosing ellipse parameters.
    pixels = np.bitwise_and(dqplane, bitmask)
    contours, hierarchy = cv.findContours(pixels, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bigcontours = [con for con in contours if cv.contourArea(con) > min_area]
    # minAreaRect is used becuase fitEllipse requires 5 points and it is possible to have a contour
    # with just 4 points.
    ellipses = [cv.minAreaRect(con) for con in bigcontours]
    return ellipses


def make_snowballs(jump_ellipses, sat_ellipses, low_threshold, high_threshold):
    # Ths routine will create a list of snowballs (ellipses) that have the center of the saturation circle
    # within the enclosing jump rectangle.
    snowballs = []
    for jump in jump_ellipses:
        sat_found = False
        if near_edge(jump, low_threshold, high_threshold):
            snowballs.append(jump)
        else:
            for sat in sat_ellipses:
                # center of saturation is within the enclosing jump rectangle
                if point_inside_ellipse(sat[0], jump):
                    if jump not in snowballs:
                        snowballs.append(jump)
#                        print("sat inside found", sat, jump)
                        sat_found = True
#        if not sat_found:
#            print("no saturation within jump rectangle ", jump)
    return snowballs


def old_point_inside_ellipse(point, ellipse):
    box = cv.boxPoints(ellipse)
    ceny = ellipse[0][0]
    cenx = ellipse[0][1]
    axis1 = ellipse[1][0]
    axis2 = ellipse[1][1]
    theta = np.deg2rad(ellipse[2])
    pointx = point[0]
    pointy = point[1]
    radius = ((np.cos(theta) * (pointx - cenx) + np.sin(theta) * (pointy - ceny))**2)/axis2**2 + \
             ((np.sin(theta) * (pointx - cenx) + np.cos(theta) * (pointy - ceny))**2)/axis1**2
    if radius < 1:
        return True
    else:
        return False


def point_inside_ellipse(point, ellipse):
    delta_center = np.sqrt((point[0]-ellipse[0][0])**2 + (point[1]-ellipse[0][1])**2)
    minor_axis = min(ellipse[1][0], ellipse[1][1])
    if delta_center < minor_axis:
        return True
    else:
        return False
def point_inside_rectangle(point, ellipse):
    box = cv.boxPoints(ellipse)
    area1 = triangle_area(point, box[0], box[1])
    area2 = triangle_area(point, box[1], box[2])
    area3 = triangle_area(point, box[2], box[3])
    area4 = triangle_area(point, box[3], box[0])
    rectangle_area = ellipse[1][0] * ellipse[1][1]
    triangle_area_sum = area1 + area2 + area3 + area4
    if triangle_area_sum > rectangle_area:
        return False
    else:
        return True

#Area = abs( (Bx * Ay - Ax * By) +
#            (Cx * By - Bx * Cy) +
#            (Ax * Cy - Cx * Ay) ) / 2
def triangle_area(point, vert1, vert2):
    area = np.abs((vert1[1] * point[0] - point[1] * vert1[0]) +
                  (vert2[1] * vert1[0] - vert1[1] * vert2[0]) +
                  (point[1] * vert2[0] - vert2[1] * point[0])) / 2
    return area

def near_edge(jump, low_threshold, high_threshold):
    if jump[0][0] < low_threshold or jump[0][1] < low_threshold\
        or jump[0][0] > high_threshold or jump[0][1] > high_threshold:
        return True
    else:
        return False

def find_faint_extended(data, gdq, read_noise_2d, nframes, snr_threshold=1.3, min_shower_area=40, inner=1,
                            outer=2, sat_flag=2, jump_flag=4, ellipse_expand = 1.1, num_grps_masked=10):
    print("input to find_faint_extended")
    print(np.nanmedian(read_noise_2d), nframes, snr_threshold, min_shower_area, inner, outer, sat_flag, jump_flag, ellipse_expand)
    read_noise_2 = read_noise_2d**2
    data[gdq == sat_flag] = np.nan
    data[gdq == 1] = np.nan
    data[gdq == jump_flag] = np.nan
    all_ellipses = []
    for intg in range(data.shape[0]):
        diff = np.diff(data[intg], axis=0)
        median_diffs = np.nanmedian(diff, axis=0)
        # calculate sigma for each pixel
        sigma = np.sqrt(np.abs(median_diffs) + read_noise_2 / nframes)
        e_jump = diff - median_diffs[np.newaxis, :, :]
        ratio = np.abs(e_jump) / sigma[np.newaxis, :, :]
#        fits.writeto("sigma.fits", sigma, overwrite=True)
        fits.writeto("input_data.fits", data[intg], overwrite=True)
        fits.writeto("diffs.fits", diff, overwrite=True)
        fits.writeto("ratio.fits", ratio, overwrite=True)
        fits.writeto("median_diffs.fits", median_diffs, overwrite=True)
        ring_2D_kernel = Ring2DKernel(inner, outer)
        for grp in range(1, ratio.shape[0] + 1):
            ellipses = []
            masked_ratio = ratio[grp-1].copy()
            jumpy, jumpx = np.where(gdq[intg, grp, :, :] == jump_flag)
            masked_ratio[jumpy, jumpx] = np.nan
            saty, satx = np.where(gdq[intg, grp, :, :] == sat_flag)
            masked_ratio[saty, satx] = np.nan
            masked_smoothed_ratio = convolve(masked_ratio, ring_2D_kernel)
#            smoothed_data = convolve(ratio[grp-1], ring_2D_kernel)
            extended_emission = np.zeros(shape=(ratio.shape[1], ratio.shape[2]), dtype=np.uint8)
            exty, extx = np.where(masked_smoothed_ratio > snr_threshold)
            extended_emission[exty, extx] = 1
#            extended_emission[masked_smoothed_ratio > snr_threshold] = 1
            pixels = np.bitwise_and(extended_emission, 1)
            contours, hierarchy = cv.findContours(pixels, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            bigcontours = [con for con in contours if cv.contourArea(con) > min_shower_area]
            ellipses = [cv.minAreaRect(con) for con in bigcontours]
            if grp == 1 and intg == 0:
                fits.writeto("median_diffs.fits", median_diffs, overwrite=True)
                fits.writeto("simga.fits", sigma, overwrite=True)
                fits.writeto("ratio_grp.fits", ratio[grp-1], overwrite=True)
                fits.writeto("starting_gdq.fits",gdq[intg, grp], overwrite=True)
                fits.writeto("extended_emission.fits", 1.0*extended_emission, overwrite=True)
                fits.writeto("masked_ratio.fits", masked_ratio, overwrite=True)
                fits.writeto("masked_smoothed_ratio.fits", masked_smoothed_ratio, overwrite=True)
                fits.writeto("pixels.fits", pixels, overwrite=True)
                print('grp', grp, snr_threshold, 'ellipses', ellipses)
                print('min_shower_area', min_shower_area)
                for con in contours:
                    if cv.contourArea(con) > 20:
                        M = cv.moments(con)
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
#                        print("area", cx, cy, cv.contourArea(con))
            for con in bigcontours:
                M = cv.moments(con)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
#                print('grp', grp, cx, cy, cv.contourArea(con))
            if len(ellipses) > 0:
                    all_ellipses.append([intg, grp, ellipses])
            test = 7
#            fits.writeto("before_ext_gdq.fits",gdq, overwrite=True)
#            gdq[intg, grp, :, :], num = extend_ellipses(gdq[intg, grp, :, :], ellipses, sat_flag, jump_flag,
#                                                        expansion=ellipse_expand, expand_by_ratio=True)
#            if grp == 1 and intg == 0:
#                fits.writeto("after_ext_gdq.fits", gdq, overwrite=True)
#    saty, satx = np.where(sat_pix == sat_flag)
#    jump_ellipse[saty, satx] = 0
    if all_ellipses:
        for showers in all_ellipses:
            intg = showers[0]
            grp = showers[1]
            ellipses = showers[2]
            gdq, num = extend_ellipses(gdq, intg, grp, ellipses, sat_flag, jump_flag,
                                                    expansion=ellipse_expand, expand_by_ratio=True,
                                                    num_grps_masked=num_grps_masked)
    return gdq
