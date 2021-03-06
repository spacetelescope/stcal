"""
Two-Point Difference method for finding outliers in a 4-D ramp data array.
The scheme used in this variation of the method uses numpy array methods
to compute first-differences and find the max outlier in each pixel while
still working in the full 4-D data array. This makes detection of the first
outlier very fast. We then iterate pixel-by-pixel over only those pixels
that are already known to contain an outlier, to look for any additional
outliers and set the appropriate DQ mask for all outliers in the pixel.
This is MUCH faster than doing all the work on a pixel-by-pixel basis.
"""

import logging
import numpy as np

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

HUGE_NUM = np.finfo(np.float32).max


def find_crs(data, group_dq, read_noise, normal_rej_thresh,
             two_diff_rej_thresh, three_diff_rej_thresh, nframes,
             flag_4_neighbors, max_jump_to_flag_neighbors,
             min_jump_to_flag_neighbors, dqflags):
    """
    Find CRs/Jumps in each integration within the input data array. The input
    data array is assumed to be in units of electrons, i.e. already multiplied
    by the gain. We also assume that the read noise is in units of electrons.
    We also assume that there are at least three groups in the integrations.
    This was checked by jump_step before this routine is called.

    Parameters
    ----------
    data: float, 4D array (num_ints, num_groups, num_rows,  num_cols)
        input ramp data

    groupdq : int, 4D array
        group DQ flags

    read_noise : float, 2D array
        The read noise of each pixel

    normal_rej_thresh : float
        cosmic ray sigma rejection threshold

    two_diff_rej_thresh : float
        cosmic ray sigma rejection threshold for ramps having 3 groups

    three_diff_rej_thresh : float
        cosmic ray sigma rejection threshold for ramps having 4 groups

    nframes : int
        The number of frames that are included in the group average

    flag_4_neighbors : bool
        if set to True (default is True), it will cause the four perpendicular
        neighbors of all detected jumps to also be flagged as a jump.

    max_jump_to_flag_neighbors : float
        value in units of sigma that sets the upper limit for flagging of
        neighbors. Any jump above this cutoff will not have its neighbors
        flagged.

    min_jump_to_flag_neighbors : float
        value in units of sigma that sets the lower limit for flagging of
        neighbors (marginal detections). Any primary jump below this value will
        not have its neighbors flagged.

    Returns
    -------
    gdq : int, 4D array
        group DQ array with reset flags

    row_below_gdq : int, 3D array (num_ints, num_groups, num_cols)
        pixels below current row also to be flagged as a CR

    row_above_gdq : int, 3D array (num_ints, num_groups, num_cols)
        pixels above current row also to be flagged as a CR

    """
    gdq = group_dq.copy()

    # Get data characteristics
    nints, ngroups, nrows, ncols = data.shape
    ndiffs = ngroups - 1

    # Create arrays for output
    row_above_gdq = np.zeros((nints, ngroups, ncols), dtype=np.uint8)
    row_below_gdq = np.zeros((nints, ngroups, ncols), dtype=np.uint8)

    # Square the read noise values, for use later
    read_noise_2 = read_noise ** 2

    # Set saturated values in the input data array to NaN, so they don't get
    # used in any of the subsequent calculations
    data[np.where(np.bitwise_and(gdq, dqflags["SATURATED"]))] = np.nan

    # Set pixels flagged as DO_NOT_USE in the input to NaN, so they don't get
    # used in any of the subsequent calculations. MIRI exposures can sometimes
    # have all pixels in the first and last groups flagged with DO_NOT_USE.
    data[np.where(np.bitwise_and(gdq, dqflags["DO_NOT_USE"]))] = np.nan

    # Loop over multiple integrations
    for integ in range(nints):

        log.info(f'Working on integration {integ + 1}:')

        # Compute first differences of adjacent groups up the ramp
        # note: roll the ngroups axis of data array to the end, to make
        # memory access to the values for a given pixel faster.
        # New form of the array has dimensions [nrows, ncols, ngroups].
        first_diffs = np.diff(np.rollaxis(data[integ], axis=0, start=3),
                              axis=2)
        positive_first_diffs = np.abs(first_diffs)

        # sat_groups is a 3D array that is true when the group is saturated
        sat_groups = np.isnan(positive_first_diffs)

        # number_sat_groups is a 2D array with the count of saturated groups
        # for each pixel
        number_sat_groups = sat_groups.sum(axis=2)

        # Make all the first diffs for saturated groups be equal to
        # 100,000 to put them above the good values in the sorted index
        first_diffs[np.isnan(first_diffs)] = 100000.

        # Here we sort the 3D array along the last axis, which is the group
        # axis. np.argsort returns a 3D array with the last axis containing
        # the indices that would yield the groups in order.
        sort_index = np.argsort(positive_first_diffs)

        # median_diffs is a 2D array with the clipped median of each pixel
        median_diffs = get_clipped_median_array(ndiffs, number_sat_groups,
                                                first_diffs, sort_index)

        # Compute uncertainties as the quadrature sum of the poisson noise
        # in the first difference signal and read noise. Because the first
        # differences can be biased by CRs/jumps, we use the median signal
        # for computing the poisson noise. Here we lower the read noise
        # by the square root of number of frames in the group.
        # Sigma is a 2D array.
        sigma = np.sqrt(np.abs(median_diffs) + read_noise_2 / nframes)

        # Reset sigma to exclude pixels with both readnoise and signal=0
        sigma_0_pixels = np.where(sigma == 0.)
        if len(sigma_0_pixels[0] > 0):
            log.debug(f'Found {len(sigma_0_pixels[0])} pixels with sigma=0')
            log.debug('which will be reset so that no jump will be detected')
            sigma[sigma_0_pixels] = HUGE_NUM

        # Compute distance of each sample from the median in units of sigma;
        # note that the use of "abs" means we'll detect positive and negative
        # outliers. ratio is a 2D array with the units of sigma deviation of
        # the difference from the median.
        ratio = np.abs(first_diffs - median_diffs[:, :, np.newaxis]) /\
            sigma[:, :, np.newaxis]
        ratio3d = np.reshape(ratio, (nrows, ncols, ndiffs))

        # Get the group index for each pixel of the largest non-saturated
        # group, assuming the indices are sorted. 2 is subtracted from ngroups
        # because we are using differences and there is one less difference
        # than the number of groups. This is a 2-D array.
        max_value_index = ngroups - 2 - number_sat_groups

        # Extract from the sorted group indices the index of the largest
        # non-saturated group.
        row, col = np.where(number_sat_groups >= 0)
        max_index1d = sort_index[row, col, max_value_index[row, col]]

        # reshape to a 2-D array :
        max_index1 = np.reshape(max_index1d, (nrows, ncols))
        max_ratio2d = np.reshape(ratio3d[row, col, max_index1[row, col]],
                                 (nrows, ncols))
        max_index1d = sort_index[row, col, 1]
        max_index2d = np.reshape(max_index1d, (nrows, ncols))
        last_ratio = np.reshape(ratio3d[row, col, max_index2d[row, col]],
                                (nrows, ncols))

        # Get the row and column indices of pixels whose largest non-saturated
        # ratio is above the threshold, First search all the pixels that have
        # at least four good groups, these will use the normal threshold
        row4cr, col4cr = np.where(np.logical_and(ndiffs -
                                  number_sat_groups >= 4,
                                  max_ratio2d > normal_rej_thresh))

        # For pixels with only three good groups, use the three diff threshold
        row3cr, col3cr = np.where(np.logical_and(ndiffs - number_sat_groups
                                  == 3,
                                  max_ratio2d > three_diff_rej_thresh))

        # Finally, for pixels with only two good groups, compare the SNR of the
        # last good group to the two diff threshold
        row2cr, col2cr = np.where(last_ratio > two_diff_rej_thresh)
        log.info(f'From highest outlier Two-point found {len(row4cr)} pixels \
                 with at least one CR and at least four groups')
        log.info(f'From highest outlier Two-point found {len(row3cr)} pixels \
                 with at least one CR and three groups')
        log.info(f'From highest outlier Two-point found {len(row2cr)} pixels \
                 with at least one CR and two groups')

        # get the rows,col pairs for all pixels with at least one CR
        all_crs_row = np.concatenate((row4cr, row3cr, row2cr))
        all_crs_col = np.concatenate((col4cr, col3cr, col2cr))

        # Loop over all pixels that we found the first CR in
        number_pixels_with_cr = len(all_crs_row)
        for j in range(number_pixels_with_cr):
            # Extract the first diffs for this pixel with at least one CR,
            # yielding a 1D array
            pix_masked_diffs = first_diffs[all_crs_row[j], all_crs_col[j]]

            # Get the scalar readnoise^2 and number of saturated groups for
            # this pixel.
            pix_rn2 = read_noise_2[all_crs_row[j], all_crs_col[j]]
            pix_sat_groups = number_sat_groups[all_crs_row[j],
                                               all_crs_col[j]]

            # Create a CR mask and set 1st CR to be found
            # cr_mask=0 designates a CR
            pix_cr_mask = np.ones(pix_masked_diffs.shape, dtype=bool)
            number_CRs_found = 1
            pix_sorted_index = sort_index[all_crs_row[j], all_crs_col[j], :]

            # setting largest diff to be a CR
            pix_cr_mask[pix_sorted_index[ndiffs - pix_sat_groups - 1]] = 0
            new_CR_found = True

            # Loop and see if there is more than one CR, setting the mask as
            # you go, stop when only 1 diffs is left.
            while new_CR_found and ((ndiffs - number_CRs_found -
                                    pix_sat_groups) > 1):
                new_CR_found = False
                largest_diff = ndiffs - number_CRs_found - pix_sat_groups

                # For this pixel get a new median difference excluding the
                # number of CRs found and the number of saturated groups
                pix_med_diff = get_clipped_median_vector(
                    ndiffs, number_CRs_found + pix_sat_groups,
                    pix_masked_diffs, pix_sorted_index)

                # Recalculate the noise and ratio for this pixel now that we
                # have rejected a CR
                pix_poisson_noise = np.sqrt(np.abs(pix_med_diff))
                pix_sigma = np.sqrt(pix_poisson_noise * pix_poisson_noise +
                                    pix_rn2 / nframes)
                pix_ratio = np.abs(pix_masked_diffs - pix_med_diff) / pix_sigma

                rej_thresh = get_rej_thresh(
                    largest_diff, two_diff_rej_thresh, three_diff_rej_thresh,
                    normal_rej_thresh)

                # Check if largest remaining difference is above threshold
                if pix_ratio[pix_sorted_index[largest_diff - 1]] > rej_thresh:
                    new_CR_found = True
                    pix_cr_mask[pix_sorted_index[largest_diff - 1]] = 0
                    number_CRs_found += 1

            # Found all CRs for this pixel. Set CR flags in input DQ array for
            # this pixel
            gdq[integ, 1:, all_crs_row[j], all_crs_col[j]] = \
                np.bitwise_or(gdq[integ, 1:, all_crs_row[j], all_crs_col[j]],
                              dqflags["JUMP_DET"] * np.invert(pix_cr_mask))

        # Flag neighbors of pixels with detected jumps, if requested
        if flag_4_neighbors:
            cr_group, cr_row, cr_col = np.where(np.bitwise_and(gdq[integ],
                                                dqflags["JUMP_DET"]))
            for j in range(len(cr_group)):

                # Jumps must be in a certain range to have neighbors flagged
                if ratio[cr_row[j], cr_col[j], cr_group[j] - 1] < \
                        max_jump_to_flag_neighbors and \
                        ratio[cr_row[j], cr_col[j], cr_group[j] - 1] > \
                        min_jump_to_flag_neighbors:

                    # This section saves flagged neighbors that are above or
                    # below the current range of row. If this method
                    # running in a single process, the row above and below are
                    # not used. If it is running in multiprocessing mode, then
                    # the rows above and below need to be returned to
                    # find_jumps to use when it reconstructs the full group dq
                    # array from the slices.
                    if cr_row[j] != 0:
                        gdq[integ, cr_group[j], cr_row[j] - 1, cr_col[j]] =\
                            np.bitwise_or(gdq[integ, cr_group[j], cr_row[j] -
                                          1, cr_col[j]], dqflags["JUMP_DET"])
                    else:
                        row_below_gdq[integ, cr_group[j], cr_col[j]] = \
                            dqflags["JUMP_DET"]

                    if cr_row[j] != nrows - 1:
                        gdq[integ, cr_group[j], cr_row[j] + 1, cr_col[j]] = \
                            np.bitwise_or(gdq[integ, cr_group[j], cr_row[j] +
                                          1, cr_col[j]], dqflags["JUMP_DET"])
                    else:
                        row_above_gdq[integ, cr_group[j], cr_col[j]] = \
                            dqflags["JUMP_DET"]

                    # Here we are just checking that we don't flag neighbors of
                    # jumps that are off the detector.
                    if cr_col[j] != 0:
                        gdq[integ, cr_group[j], cr_row[j], cr_col[j] - 1] =\
                            np.bitwise_or(gdq[integ, cr_group[j], cr_row[j],
                                          cr_col[j] - 1], dqflags["JUMP_DET"])

                    if cr_col[j] != ncols - 1:
                        gdq[integ, cr_group[j], cr_row[j], cr_col[j] + 1] =\
                            np.bitwise_or(gdq[integ, cr_group[j], cr_row[j],
                                          cr_col[j] + 1], dqflags["JUMP_DET"])

    # All done
    return gdq, row_below_gdq, row_above_gdq


def get_rej_thresh(num_usable_diffs, two_group_thresh, three_group_thresh,
                   normal_thresh):
    """
    Return the rejection threshold depending on how many useable diffs there
    are left in the pixel.

    Parameters
    ----------
    num_usable_diffs : int
        number of differences in pixel

    two_group_thresh : float
        cosmic ray sigma rejection threshold for ramps having 3 groups

    three_group_thresh : float
        cosmic ray sigma rejection threshold for ramps having 4 groups

    normal_thresh : float
        cosmic ray sigma rejection threshold

    Returns
    -------
    thresh: float
        rejection threshold
    """
    if num_usable_diffs == 2:
        return two_group_thresh
    elif num_usable_diffs == 3:
        return three_group_thresh
    else:
        return normal_thresh


def get_clipped_median_array(num_diffs, diffs_to_ignore, input_array,
                             sorted_index):
    """
    This routine will return the clipped median for input_array which is a
    three dimensional array of first differences. It will ignore the largest
    differences (diffs_to_ignore) for each pixel and compute the median of the
    remaining differences. This is only called once for the entire array.

    Parameters
    ----------
    num_diffs : int
        number of first difference, equal to the number of groups-1

    diffs_to_ignore : int, 2D array
        number of saturated groups per pixerl

    input_array : int, 3D array
        first differences of adjacent groups

    sorted_index : int, 3D array
        first differences, sorted along the groups axis

    Returns
    -------
    pix_med_diff : int, 2D array
        clipped median for the array of first differences
    """
    pix_med_diff = np.zeros_like(diffs_to_ignore)
    pix_med_index = np.zeros_like(diffs_to_ignore)

    # Process pixels with four or more good differences
    row4, col4 = np.where(num_diffs - diffs_to_ignore >= 4)

    # ignore largest value and number of CRs found when finding new median
    # Check to see if this is a 2-D array or 1-D
    # Get the index of the median value always excluding the highest value
    # In addition, decrease the index by 1 for every two diffs_to_ignore,
    # these will be saturated values in this case
    #    row, col = np.indices(diffs_to_ignore.shape)
    pix_med_index[row4, col4] = \
        sorted_index[row4, col4, (num_diffs - (diffs_to_ignore[row4,
                                  col4] + 1)) // 2]

    pix_med_diff[row4, col4] = input_array[row4, col4,
                                           pix_med_index[row4, col4]]

    # For pixels with an even number of differences the median is the mean of
    # the two central values.  So we need to get the value the other central
    # difference one lower in the sorted index that the one found above.
    even_group_rows, even_group_cols = \
        np.where(np.logical_and(num_diffs - diffs_to_ignore - 1 % 2 == 0,
                 num_diffs - diffs_to_ignore >= 4))

    pix_med_index2 = np.zeros_like(pix_med_index)
    pix_med_index2[even_group_rows, even_group_cols] = \
        sorted_index[even_group_rows, even_group_cols,
                     (num_diffs - (diffs_to_ignore[even_group_rows,
                                                   even_group_cols] + 3)) // 2]

    # Average together the two central values
    pix_med_diff[even_group_rows, even_group_cols] = \
        (pix_med_diff[even_group_rows, even_group_cols] +
         input_array[even_group_rows, even_group_cols,
         pix_med_index2[even_group_rows, even_group_cols]]) / 2.0

    # Process pixels with three good differences
    row3, col3 = np.where(num_diffs - diffs_to_ignore == 3)
    # ignore largest value and number of CRs found when finding new median
    # Check to see if this is a 2-D array or 1-D
    # Get the index of the median value always excluding the highest value
    # In addition, decrease the index by 1 for every two diffs_to_ignore,
    # these will be saturated values in this case
    #    row, col = np.indices(diffs_to_ignore.shape)
    if len(row3) > 0:
        pix_med_index[row3, col3] = \
            sorted_index[row3, col3, (num_diffs -
                         (diffs_to_ignore[row3, col3])) // 2]

        pix_med_diff[row3, col3] = \
            input_array[row3, col3, pix_med_index[row3, col3]]

    # Process pixels with two good differences
    row2, col2 = np.where(num_diffs - diffs_to_ignore == 2)
    if len(row2) > 0:
        pix_med_index[row2, col2] = sorted_index[row2, col2, 0]
        pix_med_diff[row2, col2] = input_array[row2, col2,
                                               pix_med_index[row2, col2]]

    return pix_med_diff


def get_clipped_median_vector(num_diffs, diffs_to_ignore, input_vector,
                              sorted_index):
    """
    This routine will return the clipped median for the first differences of
    the input pixel (input_vector). It will ignore the input number of largest
    differences (diffs_to_ignore). As cosmic rays are found, the
    diffs_to_ignore will increase.

    Parameters
    ----------
    num_diffs : int
        number of first difference, equal to the number of groups-1

    diffs_to_ignore : int, 2D array
        number of saturated groups per pixerl

    input_array : int, 1D array
        first differences of adjacent groups for a pixel

    sorted_index : int, 3D array
        first differences, sorted along the groups axis

    Returns
    -------
    pix_med_diff : int, vector
        clipped median for the vector of first differences

    """
    if num_diffs - diffs_to_ignore == 2:
        # For the two diff case we just return the smallest value instead of
        # the median.
        return np.min(input_vector[sorted_index[0:1]])
    elif num_diffs - diffs_to_ignore == 3:
        # For the three diff case we do not reject the largest diff when the
        # median is calculated.
        skip_max_diff = 0
    else:
        # For the four or more diff case we will skip the largest diff.
        skip_max_diff = 1

    # Find the median difference
    pix_med_index = \
        sorted_index[int(((num_diffs - skip_max_diff - diffs_to_ignore) / 2))]

    pix_med_diff = input_vector[pix_med_index]

    # If there is an even number of differences, then average the two values
    # in the middle.
    if (num_diffs - diffs_to_ignore - skip_max_diff) % 2 == 0:  # even number
        pix_med_index2 = \
            sorted_index[int((num_diffs - skip_max_diff - diffs_to_ignore) / 2)
                         - 1]
        pix_med_diff = (pix_med_diff + input_vector[pix_med_index2]) / 2.0

    return pix_med_diff
