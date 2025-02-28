import logging
import warnings

import numpy as np
import warnings
from astropy import stats
from astropy.utils.exceptions import AstropyUserWarning

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def find_crs(dataa, group_dq, read_noise, twopt_p):
    """
    Detect jump due to cosmic rays using the two point difference method.

    Parameters
    ----------
    dataa: float, 4D array (num_ints, num_groups, num_rows,  num_cols)
        input ramp data
    group_dq : int, 4D array
        group DQ flags
    read_noise : float, 2D array
        The read noise of each pixel
    twopt_p : TwoPointParams
        Class containing two point difference parameters.

    Returns
    -------
    gdq : int, 4D array
        group DQ array with reset flags
    row_below_gdq : int, 3D array (num_ints, num_groups, num_cols)
        pixels below current row also to be flagged as a CR
    row_above_gdq : int, 3D array (num_ints, num_groups, num_cols)
        pixels above current row also to be flagged as a CR
    """
    dat, gdq, read_noise_2 = set_up_data(dataa, group_dq, read_noise, twopt_p)

    # Get data characteristics
    nints, ngroups, nrows, ncols = dataa.shape
    ndiffs = (ngroups - 1) * nints

    # Create arrays for output
    row_above_gdq = np.zeros((nints, ngroups, ncols), dtype=np.uint8)
    row_below_gdq = np.zeros((nints, ngroups, ncols), dtype=np.uint8)

    ngroups_ans = groups_all_set_dnu(nints, ngroups, gdq, twopt_p)
    min_usable_groups = ngroups_ans[0]
    total_groups = ngroups_ans[1]
    min_usable_diffs = ngroups_ans[2]
    sig_clip_grps_fails = ngroups_ans[3]
    total_noise_min_grps_fails = ngroups_ans[4]
    total_sigclip_groups = ngroups_ans[5]

    # Determine whether there are enough usable groups for the two sigma clip options
    if (check_group_counts(nints, total_sigclip_groups, twopt_p)):
        sig_clip_grps_fails = True

    if min_usable_groups < twopt_p.minimum_groups:
        total_noise_min_grps_fails = True

    if total_noise_min_grps_fails and sig_clip_grps_fails:
        log.info("Jump Step was skipped because exposure has less than the minimum number of usable groups")
        dummy = np.zeros((ngroups - 1, nrows, ncols), dtype=np.float32)
        return gdq, row_below_gdq, row_above_gdq, -99, dummy

    gdq, first_diffs, median_diffs, sigma, stddev = run_jump_detection(
        dat, gdq, ndiffs, read_noise_2, nints, ngroups, total_groups, min_usable_diffs, twopt_p)

    num_primary_crs = np.sum(gdq & twopt_p.fl_jump == twopt_p.fl_jump)

    gdq, row_below_gdq, row_above_gdq = jump_detection_post_processing(
        gdq, nints, ngroups, first_diffs, median_diffs, sigma,
        row_below_gdq, row_above_gdq, twopt_p)
            
    if stddev is not None:
        return gdq, row_below_gdq, row_above_gdq, num_primary_crs, stddev

    if twopt_p.only_use_ints:
        dummy = np.zeros((dataa.shape[1] - 1, dataa.shape[2], dataa.shape[3]), dtype=np.float32)
    else:
        dummy = np.zeros((dataa.shape[2], dataa.shape[3]), dtype=np.float32)

    return gdq, row_below_gdq, row_above_gdq, num_primary_crs, dummy

    
def jump_detection_post_processing(
    gdq, nints, ngroups, first_diffs, median_diffs, sigma, row_below_gdq, row_above_gdq, twopt_p
):
    """
    Post processing for jump detection.

    gdq : ndarray
        Group DQ.
    nints : int
        Number of integration for exposure.
    ngroups : int
        Number of groups in an integration
    first_diffs : ndarray
        The first differences of the groups.
    median_diffs : ndarray
        The median of the first differences.
    sigma : ndarray
        The sigma for each pixel.
    row_below_gdq : ndarray
        Pixels below current row also to be flagged as a CR.
    row_above_gdq : ndarray
        Pixels above current row also to be flagged as a CR.
    twopt_p : TwoPointParams 
        Class containing two point difference parameters.

    Returns
    -------
    gdq : ndarray
        Updated group DQ.
    row_below_gdq : ndarray
        Pixels below current row also to be flagged as a CR.
    row_above_gdq : ndarray
        Pixels above current row also to be flagged as a CR.
    """
    # Flag the four neighbors using bitwise or, shifting the reference
    # boolean flag on pixel right, then left, then up, then down.
    # Flag neighbors above the threshold for which neither saturation 
    # nor donotuse is set.
    if twopt_p.flag_4_neighbors:
        gdq, row_below_gdq, row_above_gdq = flag_four_neighbors(
            gdq, nints, ngroups, first_diffs, median_diffs, sigma,
            row_below_gdq, row_above_gdq, twopt_p)
                
    # Flag n groups after jumps above the specified thresholds to
    # account for the transient seen after ramp jumps.  Again, use
    # boolean arrays; the propagation happens in a separate function.
    if twopt_p.after_jump_flag_n1 > 0 or twopt_p.after_jump_flag_n2 > 0:
        gdq = transient_jumps(gdq, nints, first_diffs, median_diffs, twopt_p)

    return gdq, row_below_gdq, row_above_gdq


def run_jump_detection(
    dat, gdq, ndiffs, read_noise_2, nints, ngroups, total_groups, min_usable_diffs, twopt_p
):
    """
    Detect jumps.

    Parameters
    ----------
    dat : ndarray
        Science data.
    gdq : ndarray
        Group DQ array.
    ndiffs : int
        The number of differences.
    read_noise_2 : ndarray
        The square of the read noise reference array.
    nints : int
        The number of integrations for exposure.
    ngroups : int
        The number of groups in an integration.
    total_groups : int
        Total usable groups to check.
    min_usable_diffs : int
        The minimum number of usable differences needed.
    twopt_p : TwoPointParams 
        Class containing two point difference parameters.

    Returns
    -------
    gdq : ndarray
        Update group DQ array.
    first_diffs : ndarray
        The first differences of the groups.
    median_diffs : ndarray
        The median of the first differences.
    sigma : ndarray
        The sigma for each pixel.
    """
    # set 'saturated' or 'do not use' pixels to nan in data
    dat[gdq & (twopt_p.fl_dnu | twopt_p.fl_sat) != 0] = np.nan
    
    # calculate the differences between adjacent groups (first diffs)
    # Bad data will be NaN; np.nanmedian will be used later.
    first_diffs = np.diff(dat, axis=1)
    first_diffs_finite = np.isfinite(first_diffs)
    
    # calc. the median of first_diffs for each pixel along the group axis
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*All-NaN slice encountered.*", RuntimeWarning)
        median_diffs = np.nanmedian(first_diffs, axis=(0, 1))

    # calculate sigma for each pixel
    sigma = np.sqrt(np.abs(median_diffs) + read_noise_2 / twopt_p.nframes)

    # reset sigma so pxels with 0 readnoise are not flagged as jumps
    sigma[sigma == 0.] = np.nan

    # Test to see if there are enough groups to use sigma clipping
    stddev = None
    if (check_sigma_clip_groups(nints, total_groups, twopt_p)):
        gdq, stddev = det_jump_sigma_clipping(
            gdq, nints, ngroups, total_groups, first_diffs_finite, first_diffs, twopt_p)
    else:  # There are not enough groups for sigma clipping
        if min_usable_diffs >= twopt_p.min_diffs_single_pass:
            gdq = look_for_more_than_one_jump(
                gdq, nints, first_diffs, median_diffs, sigma, first_diffs_finite, twopt_p) 
        else:  # low number of diffs requires iterative flagging
            gdq = iterative_jump(gdq, ndiffs, first_diffs, read_noise_2, twopt_p)

    return gdq, first_diffs, median_diffs, sigma, stddev


def iterative_jump(gdq, ndiffs, first_diffs, read_noise_2, twopt_p):
    """
    Detect jumps iteratively.

    Parameters
    ----------
    gdq : ndarray
        The group DQ array.
    ndiffs : int
        The number of differences.
    first_diffs : ndarray
        The first differences of the groups.
    read_noise_2 : ndarray
        The square of the read noise reference array.
    twopt_p : TwoPointParams 
        Class containing two point difference parameters.

    Returns
    -------
    gdq : ndarray
        Updated group DQ array.
    """
    # Compute the median of first_diffs for each pixel along the group axis.
    # Do not overwrite first_diffs, median_diffs, sigma.
    first_diffs_abs = np.abs(first_diffs)

    cr_pix, ratio = get_cr_locs(
        first_diffs_abs, read_noise_2, ndiffs, twopt_p) 

    # Iterate over all groups and integrations: flag and clip the
    # first CR found for each pixel (if any), then recompute medians
    # and sigmas and search all of the pixels that had a CR for
    # additional CRs. Repeat until no more CRs are found.

    for i in range(ndiffs): # Can't have more than ndiffs CRs per pixel!

        warnings.filterwarnings("ignore", ".*All-NaN slice encountered.*", RuntimeWarning)
        # Newly flagged jump locations
        new_cr = (ratio == np.nanmax(ratio, axis=(0, 1))) & cr_pix[:, np.newaxis]        
        warnings.resetwarnings()

        # No new jumps: we are done.
        if np.sum(new_cr) == 0:
            break

        # Add these jumps to the gdq array, and mask those differences.
        gdq[:, 1:][new_cr] |= twopt_p.fl_jump
        first_diffs_abs[new_cr] = np.nan

        # Look for more jumps! We only need to check pixels that had a
        # CR flagged in this iteration.
        cr_pix, ratio = get_cr_locs(first_diffs_abs, read_noise_2, ndiffs,
                                    twopt_p, index=np.any(new_cr, axis=(0, 1)))
        
    return gdq


def get_cr_locs(first_diffs_abs, read_noise_2, ndiffs, twopt_p, index=None):
    """
    Compute the pairs of rows and columns with cosmic rays.

    Parameters
    ----------
    first_diffs_abs : ndarray
        The absolute value of first differences of the groups.
    read_noise_2 : ndarray
        The square of the read noise reference array.
    ndiffs : int
        The number of differences.
    twopt_p : TwoPointParams 
        Class containing two point difference parameters.
    index : ndarray, bool or None
        Boolean index of pixels that require checking.  If None,
        check all pixels. Default None.

    Returns
    -------
    cr_pixel : ndarray, bool
        Boolean index of pixels with at least one jump
    ratio : ndarray
        Used for threshold comparison
    """

    nints, ndiffs_int, nrows, ncols = first_diffs_abs.shape
    median_diffs_iter = np.zeros((nrows, ncols), np.float32)

    # If index is supplied, we use zero for median_diffs_iter except
    # for pixels marked by index in order to save computation time.
    
    if index is not None:
        firstdiffs_reshaped = first_diffs_abs[:, :, index].reshape(nints, ndiffs_int, -1)
        median_diffs_iter[index] = calc_med_first_diffs(firstdiffs_reshaped)
    else:
        median_diffs_iter = calc_med_first_diffs(first_diffs_abs)
        
    # calculate sigma for each pixel
    sigma_iter = np.sqrt(np.abs(median_diffs_iter) + read_noise_2 / twopt_p.nframes)
    # reset sigma so pixels with 0 readnoise are not flagged as jumps
    sigma_iter[sigma_iter == 0.0] = np.nan

    # compute 'ratio' for each group. this is the value that will be
    # compared to 'threshold' to classify jumps. subtract the median
    # of first_diffs from first_diffs, take the abs. value and divide
    # by sigma.  If index is supplied, use zero for pixels not in
    # index (so that no CR will be found in that pixel).

    e_jump = np.zeros(first_diffs_abs.shape, dtype=np.float32)
    e_jump[:, :, index] = first_diffs_abs[:, :, index] - median_diffs_iter[index]
    ratio = np.abs(e_jump) / sigma_iter[np.newaxis, :, :]

    # create a 2d array containing the value of the largest 'ratio'
    # for each pixel and each integration.
    warnings.filterwarnings("ignore", ".*All-NaN slice encountered.*", RuntimeWarning)
    max_ratio = np.nanmax(ratio, axis=1)
    warnings.resetwarnings()
    # now see if the largest ratio of all groups for each pixel
    # exceeds the threshold. There are different threshold for 4+, 3,
    # and 2 usable groups

    num_usable_grps = ndiffs - np.sum(np.isnan(first_diffs_abs), axis=(0, 1))
    fourgrp_cr = (num_usable_grps >= 4) & (max_ratio > twopt_p.normal_rej_thresh)
    threegrp_cr = (num_usable_grps == 3) & (max_ratio > twopt_p.three_diff_rej_thresh)
    twogrp_cr = (num_usable_grps == 2) & (max_ratio > twopt_p.two_diff_rej_thresh)
    # Get a boolean array labeling pixels with at least one CR
    cr_pixel = fourgrp_cr | threegrp_cr | twogrp_cr
    
    return cr_pixel, ratio


def look_for_more_than_one_jump(
    gdq, nints, first_diffs, median_diffs, sigma, first_diffs_finite, twopt_p
):
    """
    Detect jumps using enough diffs in ints to look for more than one jump.

    Parameters
    ----------
    gdq : ndarray
        The group DQ array.
    nints : int
        The number of integrations for exposure.
    first_diffs : ndarray
        The first differences of the groups.
    median_diffs : ndarray
        The median of the first differences.
    sigma : ndarray
        The sigma for each pixel.
    first_diffs_finite : ndarray
        A boolean array where the first diffs are finite.
    twopt_p : TwoPointParams 
        Class containing two point difference parameters.

    Returns
    -------
    gdq : ndarray
        The updated group DQ array.
    """

    # compute 'ratio' for each group. this is the value that will be
    # compared to 'threshold' to classify jumps. subtract the median of
    # first_diffs from first_diffs, take the abs. value and divide by sigma.
    # The jump mask is the ratio greater than the threshold and the
    # difference is usable.  Loop over integrations to minimize the memory
    # footprint.
    jump_mask = np.zeros(first_diffs.shape, dtype=bool)
    for i in range(nints):
        absdiff = np.abs(first_diffs[i] - median_diffs[np.newaxis, :])
        ratio = absdiff / sigma[np.newaxis, :]
        jump_candidates = ratio > twopt_p.normal_rej_thresh
        jump_mask = jump_candidates & first_diffs_finite[i]
        gdq[i, 1:] |= jump_mask * np.uint8(twopt_p.fl_jump)
    return gdq


# XXX develop CI test for this function.
def det_jump_sigma_clipping(
    gdq, nints, ngroups, total_groups, first_diffs_finite, first_diffs, twopt_p
):
    """
    Detect jumps using sigma clipping.

    Parameters
    ----------
    gdq : ndarray
        The group DQ array.
    nints : int
        The number of integrations in an exposure.
    ngroups : int
        The number of groups in an integration
    total_groups : int
        Total usable groups to check.
    first_diffs_finite : ndarray
        A boolean array where the first diffs are finite.
    first_diffs : ndarray
        The first differences of the groups.
    twopt_p : TwoPointParams 
        Class containing two point difference parameters.

    Returns
    -------
    gdq : ndarray
        Flagged group DQ array.
    """
    log.info(" Jump Step using sigma clip {} greater than {}, rejection threshold {}".format(
        str(total_groups), str(twopt_p.minimum_sigclip_groups), str(twopt_p.normal_rej_thresh)))
    warnings.filterwarnings("ignore", ".*All-NaN slice encountered.*", RuntimeWarning)
    warnings.filterwarnings("ignore", ".*Mean of empty slice.*", RuntimeWarning)
    warnings.filterwarnings("ignore", ".*Degrees of freedom <= 0.*", RuntimeWarning)
    warnings.filterwarnings("ignore", ".*Input data contains invalid values*", AstropyUserWarning)

    axis = 0 if twopt_p.only_use_ints else (0, 1)
    clipped_diffs, alow, ahigh = stats.sigma_clip(
        first_diffs, sigma=twopt_p.normal_rej_thresh,
        axis=axis, masked=True, return_bounds=True)

    # get the standard deviation from the bounds of sigma clipping
    stddev = 0.5 * (ahigh - alow) / twopt_p.normal_rej_thresh
    jump_candidates = clipped_diffs.mask
    sat_or_dnu_not_set = gdq[:, 1:] & (twopt_p.fl_sat | twopt_p.fl_dnu) == 0
    jump_mask = jump_candidates & first_diffs_finite & sat_or_dnu_not_set
    del clipped_diffs
    gdq[:, 1:] |= jump_mask * np.uint8(twopt_p.fl_jump)

    # if grp is all jump set to do not use
    for integ in range(nints):
        for grp in range(ngroups):
            if np.all(gdq[integ, grp] & (twopt_p.fl_jump | twopt_p.fl_dnu) != 0):
                # The line below matches the comment above, but not the
                # old logic.  Leaving it for now.
                #gdq[integ, grp] |= twopt_p.fl_dnu
                jump_only = gdq[integ, grp, :, :] == twopt_p.fl_jump
                gdq[integ, grp][jump_only] = 0
                
    warnings.resetwarnings()
    return gdq, stddev


def check_sigma_clip_groups(nints, total_groups, twopt_p):
    """
    Test to see if there are enough groups to use sigma clipping.

    Parameters
    ----------
    nints : int
        The number of integrations per exposure
    total_groups : int
        Total usable groups to check.
    twopt_p : TwoPointParams 
        Class containing two point difference parameters.

    Returns
    -------
    boolean
        Are there enough groups to use sigma clipping.
    """
    test1 = (twopt_p.only_use_ints and nints >= twopt_p.minimum_sigclip_groups)
    test2 = (not twopt_p.only_use_ints and total_groups >= twopt_p.minimum_sigclip_groups)
    return test1 or test2


def flag_four_neighbors(
    gdq, nints, ngroups, first_diffs, median_diffs, sigma,
    row_below_gdq, row_above_gdq, twopt_p):
    """
    Flag four neighbors.

    Parameters
    ----------
    gdq : ndarray
        Group DQ array.
    nints : int
        The number of integrations.
    ngroups : int
        The number of groups.
    first_diffs : ndarray
        The first differences of the groups.
    median_diffs : ndarray
        The median of the first differences.
    sigma : ndarray
        The sigma for each pixel.
    row_below_gdq : ndarray
        Pixels below current row also to be flagged as a CR.
    row_above_gdq : ndarray
        Pixels above current row also to be flagged as a CR.
    twopt_p : TwoPointParams 
        Class containing two point difference parameters.

    Returns
    -------
    gdq : ndarray
        Group DQ array.
    row_below_gdq : ndarray
        Pixels below current row also to be flagged as a CR.
    row_above_gdq : ndarray
        Pixels above current row also to be flagged as a CR.
    """
    for i in range(nints):
        for j in range(ngroups - 1):
            ratio = np.abs(first_diffs[i, j] - median_diffs)/sigma
            jump_set = gdq[i, j + 1] & twopt_p.fl_jump != 0
            flag = (ratio < twopt_p.max_jump_to_flag_neighbors) & \
                (ratio > twopt_p.min_jump_to_flag_neighbors) & \
                (jump_set)

            # Dilate the flag by one pixel in each direction.
            flagsave = flag.copy()
            flag[1:] |= flagsave[:-1]
            flag[:-1] |= flagsave[1:]
            flag[:, 1:] |= flagsave[:, :-1]
            flag[:, :-1] |= flagsave[:, 1:]
            sat_or_dnu_notset = gdq[i, j + 1] & (twopt_p.fl_sat | twopt_p.fl_dnu) == 0
            gdq[i, j + 1][sat_or_dnu_notset & flag] |= twopt_p.fl_jump
            row_below_gdq[i, j + 1][flagsave[0]] = twopt_p.fl_jump
            row_above_gdq[i, j + 1][flagsave[-1]] = twopt_p.fl_jump

    return gdq, row_below_gdq, row_above_gdq


def transient_jumps(gdq, nints, first_diffs, median_diffs, twopt_p):
    """
    Flag n groups after jumps to account for the transient seen after ramp jumps.

    Parameters
    ----------
    gdq : ndarray
        The group DQ array.
    first_diffs : ndarray
        First differences of the groups of the science array.
    median_diffs : ndarray
        The media of the first differences.
    twopt_p : TwoPointParams 
        Class containing two point difference parameters.

    Returns
    -------
    gdq : ndarray
        The group DQ array.
    """
    for i in range(nints):
        ejump = first_diffs[i] - median_diffs[np.newaxis, :]
        jump_set = gdq[i] & twopt_p.fl_jump != 0

        bigjump = np.zeros(jump_set.shape, dtype=bool)
        verybigjump = np.zeros(jump_set.shape, dtype=bool)

        bigjump[1:] = (ejump >= twopt_p.after_jump_flag_e1) & jump_set[1:]
        verybigjump[1:] = (ejump >= twopt_p.after_jump_flag_e2) & jump_set[1:]

        # Propagate flags forward
        propagate_flags(bigjump, twopt_p.after_jump_flag_n1)
        propagate_flags(verybigjump, twopt_p.after_jump_flag_n2)

        # Set the flags for pixels after these jumps that are not
        # already flagged as saturated or do not use.
        sat_or_dnu_notset = gdq[i] & (twopt_p.fl_sat | twopt_p.fl_dnu) == 0
        addflag = (bigjump | verybigjump) & sat_or_dnu_notset
        gdq[i][addflag] |= twopt_p.fl_jump

    return gdq


def check_group_counts(nints, total_sigclip_groups, twopt_p):
    """
    Determine whether there are enough usable groups for the two sigma clip options

    Parameters
    ----------
    nints : int
        Number of integrations in an exposure
    total_sigclip_groups : int
        Total number of sigma-clipped groups
    twopt_p : TwoPointParams 
        Class containing two point difference parameters.

    Returns
    -------
    boolean
        Can the sigma clip options be used?
    """
    test1 = twopt_p.only_use_ints and nints < twopt_p.minimum_sigclip_groups
    test2 = not twopt_p.only_use_ints
    test2 = test2 and total_sigclip_groups < twopt_p.minimum_sigclip_groups

    return test1 or test2


def groups_all_set_dnu(nints, ngroups, gdq, twopt_p):
    """
    Get group totals with various characteristics.

    Parameters
    ----------
    nints : int
        The number of integrations in an exposure.
    ngroups : int
        The number of groups in an integration.
    gdq : ndarray
        Group DQ array.
    twopt_p : TwoPointParams 
        Class containing two point difference parameters.

    Returns
    -------
    ngroups_ans : tuple
        Various group totals.
    """
    # get data, gdq
    num_flagged_grps = 0

    # determine the number of groups with all pixels set to DO_NOT_USE
    max_flagged_grps = 0
    total_flagged_grps = 0
    for integ in range(nints):
        num_flagged_grps = 0
        for grp in range(ngroups):
            if np.all(np.bitwise_and(gdq[integ, grp, :, :], twopt_p.fl_dnu)):
                num_flagged_grps += 1
        if num_flagged_grps > max_flagged_grps:
            max_flagged_grps = num_flagged_grps
        total_flagged_grps += num_flagged_grps
    if twopt_p.only_use_ints:
        total_sigclip_groups = nints
    else:
        total_sigclip_groups = nints * ngroups - num_flagged_grps

    min_usable_groups = ngroups - max_flagged_grps
    total_groups = nints * ngroups - total_flagged_grps
    min_usable_diffs = min_usable_groups - 1
    sig_clip_grps_fails = False
    total_noise_min_grps_fails = False

    ngroups_ans = (min_usable_groups, total_groups, min_usable_diffs, sig_clip_grps_fails,
                   total_noise_min_grps_fails, total_sigclip_groups)

    return ngroups_ans


def set_up_data(dataa, group_dq, read_noise, twopt_p):
    """
    Creates copies, if desired, and squares the read noise.

    Parameters
    ----------
    dataa : ndarray
        The science data.
    group_dq : ndarray
        The group DQ array.
    read_noise : ndarray
        The pixel readnoise reference array.
    twopt_p : TwoPointParams
        Class containing two point difference parameters.

    Returns
    -------
    dat : ndarray
        The science data.
    gdq : ndarray
        The group DQ array.
    read_noise_2 : ndarray
        The square of the read noise reference array.
    """
    # copy data and group DQ array
    if twopt_p.copy_arrs:
        dat = dataa.copy()
        gdq = group_dq.copy()
    else:
        dat = dataa
        gdq = group_dq

    read_noise_2 = read_noise**2

    return dat, gdq, read_noise_2


def propagate_flags(boolean_flag, n_groups_flag):
    """
    Propagate a boolean flag array npix groups along the first axis.

    If the number of groups to propagate is not too large, or if a
    high percentage of pixels are flagged, use boolean or on the
    array.  Otherwise use np.where.  In both cases operate on the
    array in-place.

    Parameters
    ----------
    boolean_flag : 3D boolean array
        Should be True where the flag is to be propagated.
    n_groups_flag : int
        Number of groups to propagate flags forward.
    """
    ngroups = boolean_flag.shape[0]
    jmax = min(n_groups_flag, ngroups - 2)
    # Option A: iteratively propagate all flags forward by one
    # group at a time.  Do this unless we have a lot of groups
    # and cosmic rays are rare.
    if (jmax <= 50 and jmax > 0) or np.mean(boolean_flag) > 1e-3:
        for j in range(jmax):
            boolean_flag[j + 1:] |= boolean_flag[j:-1]
    # Option B: find the flags and propagate them individually.
    elif jmax > 0:
        igrp, icol, irow = np.where(boolean_flag)
        for j in range(len(igrp)):
            boolean_flag[igrp[j]:igrp[j] + n_groups_flag + 1, icol[j], irow[j]] = True
    return


def calc_med_first_diffs(in_first_diffs):
    """
    Calculate the median of `first diffs` along the group and integration axes.

    If there are 4+ usable groups (e.g not flagged as saturated, donotuse,
    or a previously clipped CR), then the group with largest absolute
    first difference will be clipped and the median of the remaining groups
    will be returned. If there are exactly 3 usable groups, the median of
    those three groups will be returned without any clipping. Finally, if
    there are two usable groups, the group with the smallest absolute
    difference will be returned.

    Parameters
    ----------
    in_first_diffs : array, float
        array containing the first differences of adjacent groups
        for a single integration. Can be 3d or 4d
        (nints, ngroups, npix) or (nints, ngroups, npix1, npix2)

    Returns
    -------
    median_diffs : array, float
        array containing the median of the group differences across
        integrations for each pixel in in_first_diffs. Will be either
        1d or 2d depending on the shape of in_first_diffs.
    """
    
    # We will modify our copy of first_diffs by setting some pixels to NaN.
    first_diffs = in_first_diffs.copy()
    nints, ndiffs = first_diffs.shape[:2]
    num_usable_diffs = (ndiffs * nints) - np.sum(np.isnan(first_diffs), axis=(0, 1))

    # Boolean arrays for the number of usable differences
    fourgrps = num_usable_diffs >= 4
    twogrps = num_usable_diffs == 2
    lessthantwogrps = num_usable_diffs < 2

    warnings.filterwarnings("ignore", ".*All-NaN slice encountered.*", RuntimeWarning)

    # Four or more usable diffs: mask the largest difference.
    maxval = np.nanmax(first_diffs, axis=(0, 1))        
    first_diffs[fourgrps & (first_diffs == maxval)] = np.nan
    
    # Three or more usable diffs: take the median
    median_diffs = np.nanmedian(first_diffs, axis=(0, 1))

    # Two usable diffs: take the minimum
    median_diffs[twogrps] = np.nanmin(first_diffs, axis=(0, 1))[twogrps]
    
    # Fewer than two usable diffs: can't do anything.
    median_diffs[lessthantwogrps] = np.nan

    warnings.resetwarnings()

    return median_diffs
