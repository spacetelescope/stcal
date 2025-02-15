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

    An interface between the detect_jumps_data function and the
    find_crs_old function using the TwoPointParams class that makes
    adding and removing parameters when using the two point
    difference without necessitating a change to the find_crs
    function signature.

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
    # START
    dat, gdq, read_noise_2 = set_up_data(dataa, group_dq, read_noise, twopt_p)

    # Get data characteristics
    nints, ngroups, nrows, ncols = dataa.shape
    ndiffs = (ngroups - 1) * nints

    # create arrays for output
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
    if (test_group_counts(nints, total_sigclip_groups, twopt_p)):
        sig_clip_grps_fails = True

    if min_usable_groups < twopt_p.minimum_groups:
        total_noise_min_grps_fails = True

    if total_noise_min_grps_fails and sig_clip_grps_fails:
        log.info("Jump Step was skipped because exposure has less than the minimum number of usable groups")
        dummy = np.zeros((ngroups - 1, nrows, ncols), dtype=np.float32)
        return gdq, row_below_gdq, row_above_gdq, -99, dummy
    else:
        # XXX current dev
        # set 'saturated' or 'do not use' pixels to nan in data
        dat[gdq & (twopt_p.fl_dnu | twopt_p.fl_sat) != 0] = np.nan
        
        # calculate the differences between adjacent groups (first diffs)
        # Bad data will be NaN; np.nanmedian will be used later.
        first_diffs = np.diff(dat, axis=1)
        first_diffs_finite = np.isfinite(first_diffs)
        
        # calc. the median of first_diffs for each pixel along the group axis
        warnings.filterwarnings("ignore", ".*All-NaN slice encountered.*", RuntimeWarning)
        median_diffs = np.nanmedian(first_diffs, axis=(0, 1))
        warnings.resetwarnings()
        # calculate sigma for each pixel
        sigma = np.sqrt(np.abs(median_diffs) + read_noise_2 / twopt_p.nframes)

        # reset sigma so pxels with 0 readnoise are not flagged as jumps
        sigma[sigma == 0.] = np.nan

        # Test to see if there are enough groups to use sigma clipping
        if (twopt_p.only_use_ints and nints >= twopt_p.minimum_sigclip_groups) or \
           (not twopt_p.only_use_ints and total_groups >= twopt_p.minimum_sigclip_groups):
            log.info(" Jump Step using sigma clip {} greater than {}, rejection threshold {}".format(
                str(total_groups), str(twopt_p.minimum_sigclip_groups), str(twopt_p.normal_rej_thresh)))
            warnings.filterwarnings("ignore", ".*All-NaN slice encountered.*", RuntimeWarning)
            warnings.filterwarnings("ignore", ".*Mean of empty slice.*", RuntimeWarning)
            warnings.filterwarnings("ignore", ".*Degrees of freedom <= 0.*", RuntimeWarning)
            warnings.filterwarnings("ignore", ".*Input data contains invalid values*", AstropyUserWarning)

            if twopt_p.only_use_ints:
                clipped_diffs, alow, ahigh = stats.sigma_clip(
                    first_diffs, sigma=twopt_p.normal_rej_thresh,
                    axis=0, masked=True, return_bounds=True)
            else:
                clipped_diffs, alow, ahigh = stats.sigma_clip(
                    first_diffs, sigma=twopt_p.normal_rej_thresh,
                    axis=(0, 1), masked=True, return_bounds=True)
            # get the standard deviation from the bounds of sigma clipping
            stddev = 0.5*(ahigh - alow)/twopt_p.normal_rej_thresh
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
        else:  # There are not enough groups for sigma clipping
            if min_usable_diffs >= twopt_p.min_diffs_single_pass:
                # There are enough diffs in all ints to look for more than one jump

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
                    
            else:  # low number of diffs requires iterative flagging
                
                # calc. the median of first_diffs for each pixel along the group axis
                # Do not overwrite first_diffs, median_diffs, sigma.
                first_diffs_abs = np.abs(first_diffs)
                median_diffs_iter = calc_med_first_diffs(first_diffs_abs)

                # calculate sigma for each pixel
                sigma_iter = np.sqrt(np.abs(median_diffs_iter) + read_noise_2 / twopt_p.nframes)
                # reset sigma so pxels with 0 readnoise are not flagged as jumps
                sigma_iter[sigma_iter == 0.0] = np.nan

                # compute 'ratio' for each group. this is the value that will be
                # compared to 'threshold' to classify jumps. subtract the median of
                # first_diffs from first_diffs, take the abs. value and divide by sigma.
                e_jump = first_diffs_abs - median_diffs_iter[np.newaxis, :, :]
                ratio = np.abs(e_jump) / sigma_iter[np.newaxis, :, :]
                # create a 2d array containing the value of the largest 'ratio' for each pixel
                warnings.filterwarnings("ignore", ".*All-NaN slice encountered.*", RuntimeWarning)
                max_ratio = np.nanmax(ratio, axis=1)
                warnings.resetwarnings()
                # now see if the largest ratio of all groups for each pixel exceeds the threshold.
                # there are different threshold for 4+, 3, and 2 usable groups
                num_unusable_groups = np.sum(np.isnan(first_diffs_abs), axis=(0, 1))
                int4cr, row4cr, col4cr = np.where(
                    np.logical_and(ndiffs - num_unusable_groups >= 4, max_ratio > twopt_p.normal_rej_thresh)
                )
                int3cr, row3cr, col3cr = np.where(
                    np.logical_and(ndiffs - num_unusable_groups == 3, max_ratio > twopt_p.three_diff_rej_thresh)
                )
                int2cr, row2cr, col2cr = np.where(
                    np.logical_and(ndiffs - num_unusable_groups == 2, max_ratio > twopt_p.two_diff_rej_thresh)
                )
                # get the rows, col pairs for all pixels with at least one CR
#                    all_crs_int = np.concatenate((int4cr, int3cr, int2cr))
                all_crs_row = np.concatenate((row4cr, row3cr, row2cr))
                all_crs_col = np.concatenate((col4cr, col3cr, col2cr))

                # iterate over all groups of the pix w/ an initial CR to look for subsequent CRs
                # flag and clip the first CR found. recompute median/sigma/ratio
                # and repeat the above steps of comparing the max 'ratio' for each pixel
                # to the threshold to determine if another CR can be flagged and clipped.
                # repeat this process until no more CRs are found.
                for j in range(len(all_crs_row)):
                    # get arrays of abs(diffs), ratio, readnoise for this pixel.
                    pix_first_diffs = first_diffs_abs[:, :, all_crs_row[j], all_crs_col[j]]
                    pix_ratio = ratio[:, :, all_crs_row[j], all_crs_col[j]]
                    pix_rn2 = read_noise_2[all_crs_row[j], all_crs_col[j]]

                    # Create a mask to flag CRs. pix_cr_mask = 0 denotes a CR
                    pix_cr_mask = np.ones(pix_first_diffs.shape, dtype=bool)

                    # set the largest ratio as a CR
                    location = np.unravel_index(np.nanargmax(pix_ratio), pix_ratio.shape)
                    pix_cr_mask[location] = 0
                    new_CR_found = True

                    # loop and check for more CRs, setting the mask as you go and
                    # clipping the group with the CR. stop when no more CRs are found
                    # or there is only one two diffs left (which means there is
                    # actually one left, since the next CR will be masked after
                    # checking that condition)
                    while new_CR_found and (ndiffs - np.sum(np.isnan(pix_first_diffs)) > 2):
                        new_CR_found = False

                        # set CRs to nans in first diffs to clip them
                        pix_first_diffs[~pix_cr_mask] = np.nan

                        # recalculate median, sigma, and ratio
                        new_pix_median_diffs = calc_med_first_diffs(pix_first_diffs)

                        new_pix_sigma = np.sqrt(np.abs(new_pix_median_diffs) + pix_rn2 / twopt_p.nframes)
                        new_pix_ratio = np.abs(pix_first_diffs - new_pix_median_diffs) / new_pix_sigma

                        # check if largest ratio exceeds threshold appropriate for num remaining groups

                        # select appropriate thresh. based on number of remaining groups
                        rej_thresh = twopt_p.normal_rej_thresh
                        if ndiffs - np.sum(np.isnan(pix_first_diffs)) == 3:
                            rej_thresh = twopt_p.three_diff_rej_thresh
                        if ndiffs - np.sum(np.isnan(pix_first_diffs)) == 2:
                            rej_thresh = twopt_p.two_diff_rej_thresh
                        max_idx = np.nanargmax(new_pix_ratio)
                        location = np.unravel_index(max_idx, new_pix_ratio.shape)
                        if new_pix_ratio[location] > rej_thresh:
                            new_CR_found = True
                            pix_cr_mask[location] = 0
                        unusable_diffs = np.sum(np.isnan(pix_first_diffs))
                    # Found all CRs for this pix - set flags in input DQ array
                    gdq[:, 1:, all_crs_row[j], all_crs_col[j]] = np.bitwise_or(
                        gdq[:, 1:, all_crs_row[j],
                        all_crs_col[j]],
                        twopt_p.fl_jump * np.invert(pix_cr_mask),
                    )
                    
    num_primary_crs = np.sum(gdq & twopt_p.fl_jump == twopt_p.fl_jump)
    
    # Flag the four neighbors using bitwise or, shifting the reference
    # boolean flag on pixel right, then left, then up, then down.
    # Flag neighbors above the threshold for which neither saturation 
    # nor donotuse is set.
    
    if twopt_p.flag_4_neighbors:
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
                
    # Flag n groups after jumps above the specified thresholds to
    # account for the transient seen after ramp jumps.  Again, use
    # boolean arrays; the propagation happens in a separate function.
    
    if twopt_p.after_jump_flag_n1 > 0 or twopt_p.after_jump_flag_n2 > 0:
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
            
    if "stddev" in locals():
        return gdq, row_below_gdq, row_above_gdq, num_primary_crs, stddev

    if twopt_p.only_use_ints:
        dummy = np.zeros((dataa.shape[1] - 1, dataa.shape[2], dataa.shape[3]), dtype=np.float32)
    else:
        dummy = np.zeros((dataa.shape[2], dataa.shape[3]), dtype=np.float32)

    return gdq, row_below_gdq, row_above_gdq, num_primary_crs, dummy
# END


def test_group_counts(nints, total_sigclip_groups, twopt_p):
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
    """Propagate a boolean flag array npix groups along the first axis.
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
    Returns
    -------
    None
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
    """Calculate the median of `first diffs` along the group axis.

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
        for a single integration. Can be 3d or 1d (for a single pix)

    Returns
    -------
    median_diffs : float or array, float
        If the input is a single pixel, a float containing the median for
        the groups in that pixel will be returned. If the input is a 3d
        array of several pixels, a 2d array with the median for each pixel
        will be returned.
    """
    first_diffs = in_first_diffs.copy()
    if first_diffs.ndim == 1:  # in the case where input is a single pixel
        num_usable_groups = len(first_diffs) - np.sum(np.isnan(first_diffs), axis=0)
        if num_usable_groups >= 4:  # if 4+, clip largest and return median
            mask = np.ones_like(first_diffs).astype(bool)
            mask[np.nanargmax(np.abs(first_diffs))] = False  # clip the diff with the largest abs value
            return np.nanmedian(first_diffs[mask])

        if num_usable_groups == 3:  # if 3, no clipping just return median
            return np.nanmedian(first_diffs)

        if num_usable_groups == 2:  # if 2, return diff with minimum abs
            return first_diffs[np.nanargmin(np.abs(first_diffs))]

        return np.nan

    if first_diffs.ndim == 2:  # in the case where input is a single pixel
        nansum = np.sum(np.isnan(first_diffs), axis=(0, 1))
        num_usable_diffs = first_diffs.size - np.sum(np.isnan(first_diffs), axis=(0, 1))
        if num_usable_diffs >= 4:  # if 4+, clip largest and return median
            mask = np.ones_like(first_diffs).astype(bool)
            location = np.unravel_index(np.nanargmax(first_diffs), first_diffs.shape)
            mask[location] = False  # clip the diff with the largest abs value
            return np.nanmedian(first_diffs[mask])
        elif num_usable_diffs == 3:  # if 3, no clipping just return median
            return np.nanmedian(first_diffs)
        elif num_usable_diffs == 2:  # if 2, return diff with minimum abs
            TEST = np.nanargmin(np.abs(first_diffs))
            diff_min_idx = np.nanargmin(first_diffs)
            location = np.unravel_index(diff_min_idx, first_diffs.shape)
            return first_diffs[location]
        else:
            return np.nan

    if first_diffs.ndim == 4:
        # if input is multi-dimensional
        nints, ndiffs, nrows, ncols = first_diffs.shape
        shaped_diffs = np.reshape(first_diffs, ((nints * ndiffs), nrows, ncols))
        num_usable_diffs = (ndiffs * nints) - np.sum(np.isnan(shaped_diffs), axis=0)
        median_diffs = np.zeros((nrows, ncols))  # empty array to store median for each pix

        # process groups with >=4 usable diffs
        row4, col4 = np.where(num_usable_diffs >= 4)  # locations of >= 4 usable diffs pixels
        if len(row4) > 0:
            four_slice = shaped_diffs[:, row4, col4]
            loc0 = np.nanargmax(four_slice, axis=0)
            shaped_diffs[loc0, row4, col4] = np.nan
            median_diffs[row4, col4] = np.nanmedian(shaped_diffs[:, row4, col4], axis=0)

        # process groups with 3 usable groups
        row3, col3 = np.where(num_usable_diffs == 3)  # locations of == 3 usable diff pixels
        if len(row3) > 0:
            three_slice = shaped_diffs[:, row3, col3]
            median_diffs[row3, col3] = np.nanmedian(three_slice, axis=0)  # add median to return arr for these pix

        # process groups with 2 usable groups
        row2, col2 = np.where(num_usable_diffs == 2)  # locations of == 2 usable diff pixels
        if len(row2) > 0:
            two_slice = shaped_diffs[ :, row2, col2]
            two_slice[np.nanargmax(np.abs(two_slice), axis=0),
                      np.arange(two_slice.shape[1])] = np.nan  # mask larger abs. val
            median_diffs[row2, col2] = np.nanmin(two_slice, axis=0)  # add med. to return arr

        # set the medians all groups with less than 2 usable diffs to nan to skip further
        # calculations for these pixels
        row_none, col_none = np.where(num_usable_diffs < 2)
        median_diffs[row_none, col_none] = np.nan

        return median_diffs
