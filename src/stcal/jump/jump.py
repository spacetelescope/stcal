import logging
import multiprocessing
import time
from astropy.io import fits
import numpy as np
import cv2 as cv

from astropy.convolution import Ring2DKernel
from astropy.convolution import convolve

from . import constants
from . import twopoint_difference as twopt

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def detect_jumps(frames_per_group, data, gdq, pdq, err,
                 gain_2d, readnoise_2d, rejection_thresh,
                 three_grp_thresh, four_grp_thresh, max_cores,
                 max_jump_to_flag_neighbors,
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
                 expand_large_events=False,
                 sat_expand=2, min_sat_radius_extend=2.5, find_showers=False,
                 edge_size=25, extend_snr_threshold=1.2, extend_min_area=90,
                 extend_inner_radius=1, extend_outer_radius=2.6,
                 extend_ellipse_expand_ratio=1.2, grps_masked_after_shower=5,
                 max_extended_radius=200, minimum_groups=3,
                 minimum_sigclip_groups=100, only_use_ints=True):

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

    flag_4_neighbors: bool
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
        The minimum area of saturated pixels at the center of a snowball. Only
        contours with area above the minimum will create snowballs.

    min_jump_area : float
        The minimum contour area to trigger the creation of enclosing ellipses
        or circles.

    expand_factor : float
        The factor that is used to increase the size of the enclosing
        circle/ellipse jump flagged pixels.

    use_ellipses : deprecated

    sat_required_snowball : bool
        If true there must be a saturation circle within the radius of the jump
        circle to trigger the creation of a snowball. All true snowballs appear
        to have at least one saturated pixel.

    edge_size : int
        The distance from the edge of the detector where saturated cores are not
        required for snowball detection

    expand_large_events : bool
        When True this triggers the flagging of snowballs for NIR detectors.

    sat_expand : int
        The number of pixels to expand the saturated core of detected snowballs

    find_showers : boolean
        Turns on the flagging of the faint extended emission of MIRI showers

    extend_snr_threshold : float
        The SNR minimum for the detection of faint extended showers in MIRI

    extend_min_area : float
        The required minimum area of extended emission after convolution for the
        detection of showers in MIRI

    extend_inner_radius : float
        The inner radius of the Ring2DKernal that is used for the detection of
        extended emission in showers

    extend_outer_radius : float
        The outer radius of the Ring2DKernal that is used for the detection of
        extended emission in showers

    extend_ellipse_expand_ratio : float
        Multiplicative factor to expand the radius of the ellipse fit to the
        detected extended emission in MIRI showers

    grps_masked_after_shower : int
        Number of groups after detected extended emission to flag as a jump for
        MIRI showers

    max_extended_radius : int
        The maximum radius for any extension of saturation or jump

    min_sat_radius_extend : float
        The minimum radius of the saturated core of a snowball for the core to
        be extended

    Returns
    -------
    gdq : int, 4D array
        updated group dq array

    pdq : int, 2D array
        updated pixel dq array
    """
    print("only use ints detect jumps", only_use_ints)
    constants.update_dqflags(dqflags)  # populate dq flags
    sat_flag = dqflags["SATURATED"]
    jump_flag = dqflags["JUMP_DET"]
    number_extended_events = 0
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
        gdq, row_below_dq, row_above_dq, total_primary_crs = \
            twopt.find_crs(data, gdq, readnoise_2d, rejection_thresh,
                           three_grp_thresh, four_grp_thresh, frames_per_group,
                           flag_4_neighbors, max_jump_to_flag_neighbors,
                           min_jump_to_flag_neighbors, dqflags,
                           after_jump_flag_e1=after_jump_flag_e1,
                           after_jump_flag_n1=after_jump_flag_n1,
                           after_jump_flag_e2=after_jump_flag_e2,
                           after_jump_flag_n2=after_jump_flag_n2, copy_arrs=False,
                           minimum_groups=3, minimum_sigclip_groups=minimum_sigclip_groups,
                           only_use_ints=only_use_ints)
        #  This is the flag that controls the flagging of either snowballs.
        if expand_large_events:
            flag_large_events(gdq, jump_flag, sat_flag, min_sat_area=min_sat_area,
                              min_jump_area=min_jump_area,
                              expand_factor=expand_factor,
                              sat_required_snowball=sat_required_snowball,
                              min_sat_radius_extend=min_sat_radius_extend,
                              edge_size=edge_size, sat_expand=sat_expand,
                              max_extended_radius=max_extended_radius)
        if find_showers:
            gdq, num_showers = find_faint_extended(data, gdq, readnoise_2d,
                                                   frames_per_group,
                                                   snr_threshold=extend_snr_threshold,
                                                   min_shower_area=extend_min_area,
                                                   inner=extend_inner_radius,
                                                   outer=extend_outer_radius,
                                                   sat_flag=sat_flag, jump_flag=jump_flag,
                                                   ellipse_expand=extend_ellipse_expand_ratio,
                                                   num_grps_masked=grps_masked_after_shower,
                                                   max_extended_radius=max_extended_radius)
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
        copy_arrs = False  # we don't need to copy arrays again in find_crs

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
                              copy_arrs, minimum_groups, minimum_sigclip_groups,
                              only_use_ints))

        # last slice get the rest
        slices.insert(n_slices - 1, (data[:, :, (n_slices - 1) *
                                     yinc:n_rows, :],
                                     gdq[:, :, (n_slices - 1) *
                                     yinc:n_rows, :],
                                     readnoise_2d[(n_slices - 1) *
                                     yinc:n_rows, :],
                                     rejection_thresh, three_grp_thresh,
                                     four_grp_thresh, frames_per_group,
                                     flag_4_neighbors,
                                     max_jump_to_flag_neighbors,
                                     min_jump_to_flag_neighbors, dqflags,
                                     after_jump_flag_e1, after_jump_flag_n1,
                                     after_jump_flag_e2, after_jump_flag_n2,
                                     copy_arrs, minimum_groups, minimum_sigclip_groups,
                                     only_use_ints))
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
        total_primary_crs = 0
        for resultslice in real_result:
            if len(real_result) == k + 1:  # last result
                gdq[:, :, k * yinc:n_rows, :] = resultslice[0]
            else:
                gdq[:, :, k * yinc:(k + 1) * yinc, :] = resultslice[0]
            row_below_gdq[:, :, :] = resultslice[1]
            row_above_gdq[:, :, :] = resultslice[2]
            total_primary_crs += resultslice[3]
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
        print("total primary CRs, multiple threads", total_primary_crs)
        #  This is the flag that controls the flagging of either
        #  snowballs or showers.
        if expand_large_events:
            total_snowballs = flag_large_events(gdq, jump_flag, sat_flag,
                              min_sat_area=min_sat_area,
                              min_jump_area=min_jump_area,
                              expand_factor=expand_factor,
                              sat_required_snowball=sat_required_snowball,
                              min_sat_radius_extend=min_sat_radius_extend,
                              edge_size=edge_size, sat_expand=sat_expand,
                              max_extended_radius=max_extended_radius)
            log.info('Total snowballs = %i' % total_snowballs)
            number_extended_events = total_snowballs
        if find_showers:
            gdq, num_showers = \
                find_faint_extended(data, gdq, readnoise_2d,
                                    frames_per_group,
                                    snr_threshold=extend_snr_threshold,
                                    min_shower_area=extend_min_area,
                                    inner=extend_inner_radius,
                                    outer=extend_outer_radius,
                                    sat_flag=sat_flag,
                                    jump_flag=jump_flag,
                                    ellipse_expand=extend_ellipse_expand_ratio,
                                    num_grps_masked=grps_masked_after_shower,
                                    max_extended_radius=max_extended_radius)
            log.info('Total showers= %i' % num_showers)
            number_extended_events = num_showers
    elapsed = time.time() - start
    log.info('Total elapsed time = %g sec' % elapsed)

    # Back out the applied gain to the SCI, ERR, and readnoise arrays so they're
    #    back in units of DN
    data /= gain_2d
    err /= gain_2d
    readnoise_2d /= gain_2d

    # Return the updated data quality arrays
#    return gdq, pdq
    return gdq, pdq, total_primary_crs, number_extended_events


def flag_large_events(gdq, jump_flag, sat_flag, min_sat_area=1,
                      min_jump_area=6,
                      expand_factor=2.0,
                      sat_required_snowball=True, min_sat_radius_extend=2.5,
                      sat_expand=2, edge_size=25, max_extended_radius=200):
    """
    This routine controls the creation of expanded regions that are flagged as
    jumps.

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
    min_sat_area : int
        The minimum area of saturated pixels within the jump circle to trigger
        the creation of a snowball.
    min_jump_area : int
        The minimum area of a contour to cause the creation of the
        minimum enclosing ellipse or circle.
    expand_factor : float
        The factor that increases the size of the snowball or enclosing ellipse.
    sat_required_snowball : bool
        Require that there is a saturated pixel within the radius of the jump
        circle to trigger the formation of a snowball.
    min_sat_radius_extend : float
        The smallest radius to trigger extension of the saturated core
    sat_expand : int
        The number of pixels to extend the saturated core by
    edge_size : int
        The distance from the edge of the detector where saturation is not
        required for a snowball to be created
    max_extended_radius : int
        The largest radius that a snowball or shower can be extended

    Returns
    -------
    Nothing, gdq array is modified.

    """

    log.info('Flagging large Snowballs')

    n_showers_grp = []
    total_snowballs = 0
    for integration in range(gdq.shape[0]):
        for group in range(1, gdq.shape[1]):
            current_gdq = 1.0 * gdq[integration, group, :, :]
            prev_gdq = 1.0 * gdq[integration, group - 1, :, :]
            diff_gdq = 1.0 * current_gdq - prev_gdq
            diff_gdq[diff_gdq != sat_flag] = 0
            new_sat = diff_gdq.astype('uint8')
            # find the ellipse parameters for newly saturated pixels
            sat_ellipses = find_ellipses(new_sat, sat_flag, min_sat_area)

            # find the ellipse parameters for jump regions
            jump_ellipses = find_ellipses(gdq[integration, group, :, :],
                                          jump_flag, min_jump_area)
            if sat_required_snowball:
                low_threshold = edge_size
                high_threshold = max(0, gdq.shape[2] - edge_size)

                gdq, snowballs = make_snowballs(gdq, integration, group,
                                                jump_ellipses, sat_ellipses,
                                                low_threshold, high_threshold,
                                                min_sat_radius_extend,
                                                sat_expand, sat_flag,
                                                max_extended_radius)
            else:
                snowballs = jump_ellipses
            n_showers_grp.append(len(snowballs))
            total_snowballs += len(snowballs)
            gdq, num_events = extend_ellipses(gdq, integration, group,
                                              snowballs,
                                              sat_flag, jump_flag,
                                              expansion=expand_factor,
                                              max_extended_radius=max_extended_radius)
        if np.all(np.array(n_showers_grp) == 0):
            log.info(f'No snowballs found in integration {integration}.')
        else:
            log.info(f' In integration {integration}, number of snowballs ' +
                     f'in each group = {n_showers_grp}')
    return total_snowballs

def extend_saturation(cube, grp, sat_ellipses, sat_flag,
                      min_sat_radius_extend, expansion=2,
                      max_extended_radius=200):
    image = np.zeros(shape=(cube.shape[1], cube.shape[2], 3), dtype=np.uint8)
    outcube = cube.copy()
    for ellipse in sat_ellipses:
        ceny = ellipse[0][0]
        cenx = ellipse[0][1]
        minor_axis = min(ellipse[1][1], ellipse[1][0])
        if minor_axis > min_sat_radius_extend:
            axis1 = ellipse[1][0] + expansion
            axis2 = ellipse[1][1] + expansion
            alpha = ellipse[2]
            axis1 = min(axis1, max_extended_radius)
            axis2 = min(axis2, max_extended_radius)
            image = cv.ellipse(image, (round(ceny), round(cenx)),
                               (round(axis1/2),
                               round(axis2/2)), alpha, 0, 360, (0, 0, 22), -1)
            sat_ellipse = image[:, :, 2]
            saty, satx = np.where(sat_ellipse == 22)
            outcube[grp:, saty, satx] = sat_flag
    return outcube


def extend_ellipses(gdq_cube, intg, grp, ellipses, sat_flag, jump_flag,
                    expansion=1.9, expand_by_ratio=True,
                    num_grps_masked=1, max_extended_radius=200):
    # For a given DQ plane it will use the list of ellipses to create
    #  expanded ellipses of pixels with
    # the jump flag set.
    plane = gdq_cube[intg, grp, :, :]
    image = np.zeros(shape=(plane.shape[0], plane.shape[1], 3), dtype=np.uint8)
    num_ellipses = len(ellipses)
    for ellipse in ellipses:
        ceny = ellipse[0][0]
        cenx = ellipse[0][1]
        # Expand the ellipse by the expansion factor. The number of pixels
        # added to both axes is
        # the number of pixels added to the minor axis. This prevents very
        # large flagged ellipses
        # with high axis ratio ellipses. The major and minor axis are not
        # always the same index.
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
        axis1 = min(axis1, max_extended_radius)
        axis2 = min(axis2, max_extended_radius)
        alpha = ellipse[2]
        image = cv.ellipse(image, (round(ceny), round(cenx)), (round(axis1 / 2),
                           round(axis2 / 2)), alpha, 0, 360,
                           (0, 0, jump_flag), -1)
        jump_ellipse = image[:, :, 2]
        last_grp = min(grp + num_grps_masked, gdq_cube.shape[1])
        #  This loop will flag the number of groups
        for flg_grp in range(grp, last_grp):
            sat_pix = np.bitwise_and(gdq_cube[intg, flg_grp, :, :], sat_flag)
            saty, satx = np.where(sat_pix == sat_flag)
            jump_ellipse[saty, satx] = 0
            gdq_cube[intg, flg_grp, :, :] = \
                np.bitwise_or(gdq_cube[intg, flg_grp, :, :], jump_ellipse)
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
    # Using an input DQ plane this routine will find the groups of pixels with
    # at least the minimum
    # area and return a list of the minimum enclosing ellipse parameters.
    pixels = np.bitwise_and(dqplane, bitmask)
    contours, hierarchy = cv.findContours(pixels, cv.RETR_EXTERNAL,
                                          cv.CHAIN_APPROX_SIMPLE)
    bigcontours = [con for con in contours if cv.contourArea(con) > min_area]
    # minAreaRect is used becuase fitEllipse requires 5 points and it is
    # possible to have a contour
    # with just 4 points.
    ellipses = [cv.minAreaRect(con) for con in bigcontours]
    return ellipses


def make_snowballs(gdq, integration, group, jump_ellipses, sat_ellipses,
                   low_threshold, high_threshold,
                   min_sat_radius, expansion, sat_flag, max_extended_radius):
    # Ths routine will create a list of snowballs (ellipses) that have the
    # center
    # of the saturation circle within the enclosing jump rectangle.
    snowballs = []
    for jump in jump_ellipses:
        if near_edge(jump, low_threshold, high_threshold):
            snowballs.append(jump)
        else:
            for sat in sat_ellipses:
                # center of saturation is within the enclosing jump rectangle
                if point_inside_ellipse(sat[0], jump):
                    # center of jump should be saturated
                    jump_center = jump[0]
                    if gdq[integration, group, round(jump_center[1]),
                           round(jump_center[0])] == sat_flag:
                        if jump not in snowballs:
                            snowballs.append(jump)
                            gdq[integration, :, :, :] = \
                                extend_saturation(gdq[integration, :, :, :],
                                                  group, [sat], sat_flag,
                                                  min_sat_radius,
                                                  expansion=expansion,
                                                  max_extended_radius=max_extended_radius)
    return gdq, snowballs


def point_inside_ellipse(point, ellipse):
    delta_center = np.sqrt((point[0]-ellipse[0][0])**2 +
                           (point[1]-ellipse[0][1])**2)
    minor_axis = min(ellipse[1][0], ellipse[1][1])
    if delta_center < minor_axis:
        return True
    else:
        return False


def near_edge(jump, low_threshold, high_threshold):
    #  This routing tests whether the center of a jump is close to the edge of
    # the detector. Jumps that are within the threshold will not requre a
    # saturated core since this may be off the detector
    if jump[0][0] < low_threshold or jump[0][1] < low_threshold\
            or jump[0][0] > high_threshold or jump[0][1] > high_threshold:
        return True
    else:
        return False


def find_faint_extended(indata, gdq, readnoise_2d, nframes, snr_threshold=1.3,
                        min_shower_area=40, inner=1, outer=2, sat_flag=2,
                        jump_flag=4, ellipse_expand=1.1, num_grps_masked=25,
                        max_extended_radius=200):
    """
    Parameters
    ----------
      indata : float, 4D array
          Science array.
      gdq : int, 2D array
          Group dq array.
      readnoise_2d : float, 2D array
          Readnoise for all pixels.
      nframes : int
          The number frames that are averaged in the group.
      snr_threshold : float
          The signal-to-noise ratio threshold for detection of extended
          emission.
      min_shower_area : int
          The minimum area for a group of pixels to be flagged as a shower.
      inner: int
          The inner radius of the ring_2D_kernal used for the convolution.
      outer : int
          The outer radius of the ring_2D_kernal used for the convolution.
      sat_flag : int
          The integer value of the saturation flag.
      jump_flag : int
          The integer value of the jump flag
      ellipse_expand: float
          The relative increase in the size of the fitted ellipse to be
          applied to the shower.
    num_grps_masked: int
        The number of groups after the detected shower to be flagged as jump.
    max_extended_radius: int
        The upper limit for the extension of saturation and jump
    Returns
    -------
    gdq : int, 4D array
      updated group dq array.
    number_ellipse : int
    Total number of showers detected.

    """
    print("find showers on sigmaclip")
    read_noise_2 = readnoise_2d**2
    data = indata.copy()
    data[gdq == sat_flag] = np.nan
    data[gdq == 1] = np.nan
    data[gdq == jump_flag] = np.nan
    all_ellipses = []
    for intg in range(data.shape[0]):
        diff = np.diff(data[intg], axis=0)
        median_diffs = np.nanmedian(diff, axis=0)
        # calculate sigma for each pixel
        sigma = np.sqrt(np.abs(median_diffs) + read_noise_2 / nframes)

        # The difference from the median difference for each group
        e_jump = diff - median_diffs[np.newaxis, :, :]

        ratio = np.abs(e_jump) / sigma[np.newaxis, :, :]  # SNR ratio of
        # each diff.

        #  The convolution kernal creation
        ring_2D_kernel = Ring2DKernel(inner, outer)
        for grp in range(1, ratio.shape[0] + 1):
            masked_ratio = ratio[grp-1].copy()
            jumpy, jumpx = np.where(gdq[intg, grp, :, :] == jump_flag)
            #  mask pix. that are already flagged as jump
            masked_ratio[jumpy, jumpx] = np.nan

            saty, satx = np.where(gdq[intg, grp, :, :] == sat_flag)

            #  mask pix. that are already flagged as sat.
            masked_ratio[saty, satx] = np.nan

            masked_smoothed_ratio = convolve(masked_ratio, ring_2D_kernel)
            extended_emission = np.zeros(shape=(ratio.shape[1],
                                                ratio.shape[2]), dtype=np.uint8)
            exty, extx = np.where(masked_smoothed_ratio > snr_threshold)
            extended_emission[exty, extx] = 1
            if grp == 179 and intg == 0:
                fits.writeto("masked_ratio.fits",masked_ratio, overwrite=True)
                fits.writeto("masked_smoothed_ratio.fits", masked_smoothed_ratio, overwrite=True)
                fits.writeto("extended_emission.fits", extended_emission, overwrite=True)
            #  find the contours of the extended emission
            contours, hierarchy = cv.findContours(extended_emission,
                                                  cv.RETR_EXTERNAL,
                                                  cv.CHAIN_APPROX_SIMPLE)
            #  get the countours that are above the minimum size
            bigcontours = [con for con in contours if cv.contourArea(con) >
                           min_shower_area]
            #  get the minimum enclosing rectangle which is the same as the
            # minimum enclosing ellipse
            ellipses = [cv.minAreaRect(con) for con in bigcontours]
            if grp==179 and intg == 0:

                expand_by_ratio = True
                expansion = 1.0
                plane = gdq[intg, grp, :, :]
                image = np.zeros(shape=(plane.shape[0], plane.shape[1], 3), dtype=np.uint8)
                image2 = np.zeros_like(image)
                cv.drawContours(image2, bigcontours, -1, (0, 0, jump_flag), -1)
                big_contour_image = image2[:, :, 2]
                fits.writeto("big_contour.fits", big_contour_image, overwrite=True)
                fits.writeto("image2.fits", image2, overwrite=True)
                num_ellipses = len(ellipses)
                for ellipse in ellipses:
                    ceny = ellipse[0][0]
                    cenx = ellipse[0][1]
                    # Expand the ellipse by the expansion factor. The number of pixels
                    # added to both axes is
                    # the number of pixels added to the minor axis. This prevents very
                    # large flagged ellipses
                    # with high axis ratio ellipses. The major and minor axis are not
                    # always the same index.
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
                    axis1 = min(axis1, max_extended_radius)
                    axis2 = min(axis2, max_extended_radius)
                    alpha = ellipse[2]
                    image = cv.ellipse(image, (round(ceny), round(cenx)), (round(axis1 / 2),
                                                                           round(axis2 / 2)), alpha, 0, 360,
                                       (0, 0, jump_flag), -1)
                    jump_ellipse = image[:, :, 2]
                fits.writeto("jump_ellipse.fits", jump_ellipse, overwrite=True)
            if len(ellipses) > 0:
                # add all the showers for this integration to the list
                all_ellipses.append([intg, grp, ellipses])
    if all_ellipses:
        #  Now we actually do the flagging of the pixels inside showers.
        # This is deferred until all showers are detected. because the showers
        # can flag future groups and would confuse the detection algorthim if
        # we worked on groups that already had some flagged showers.
        for showers in all_ellipses:
            intg = showers[0]
            grp = showers[1]
            ellipses = showers[2]
            gdq, num = extend_ellipses(gdq, intg, grp, ellipses, sat_flag,
                                       jump_flag, expansion=ellipse_expand,
                                       expand_by_ratio=True,
                                       num_grps_masked=num_grps_masked,
                                       max_extended_radius=max_extended_radius)
    if np.all(all_ellipses == 0):
        log.info('No showers found in exposure.')
    else:
        num_showers = len(all_ellipses)
        log.info(f' Number of showers flagged = {num_showers}')
    return gdq, len(all_ellipses)
