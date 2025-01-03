import copy
import logging

import numpy as np
from scipy import ndimage

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def flag_saturated_pixels(
    data, gdq, pdq, sat_thresh, sat_dq, atod_limit, dqflags, n_pix_grow_sat=1, zframe=None, read_pattern=None
):
    """
    Short Summary
    -------------
    Apply flagging for saturation based on threshold values stored in the
    saturation reference file data `sat_thresh` and A/D floor based on testing
    for 0 DN values. For A/D floor flagged groups, the DO_NOT_USE flag is also
    set.

    Parameters
    ----------
    data : float, 4D array
        science array

    gdq : int, 4D array
        group dq array

    pdq : int, 2D array
        pixelg dq array

    sat_thresh : `np.array`
        Pixel-wise threshold for saturation, same shape `data`

    sat_dq : `np.array`
        data quality flags associated with `sat_thresh`

    atod_limit : int
        hard DN limit of 16-bit A-to-D converter

    dqflags : dict
        A dictionary with at least the following keywords:
        DO_NOT_USE, SATURATED, AD_FLOOR, NO_SAT_CHECK

    n_pix_grow_sat : int
        Number of pixels that each flagged saturated pixel should be 'grown',
        to account for charge spilling. Default is 1.

    zframe : float, 3D array
        The ZEROFRAME.

    read_pattern : List[List[float or int]] or None
        The times or indices of the frames composing each group.


    Returns
    -------
    gdq : int, 4D array
        updated group dq array

    pdq : int, 2D array
        updated pixel dq array
    """
    nints, ngroups, nrows, ncols = data.shape
    dnu = int(dqflags["DO_NOT_USE"])
    saturated = int(dqflags["SATURATED"])
    ad_floor = int(dqflags["AD_FLOOR"])
    no_sat_check = int(dqflags["NO_SAT_CHECK"])

    # Identify pixels flagged in reference file as NO_SAT_CHECK,
    no_sat_check_mask = np.bitwise_and(sat_dq, no_sat_check) == no_sat_check

    # Flag pixels in the saturation threshold array with NO_SAT_CHECK,
    # and add them to to no_sat_check_mask.
    sat_dq[np.isnan(sat_thresh)] |= no_sat_check
    no_sat_check_mask |= np.isnan(sat_thresh)

    # Set the saturation check threshold above the A-to-D
    # converter limit so that they don't get flagged as saturated, for
    # pixels in the no_sat_check_mask.
    sat_thresh[no_sat_check_mask] = atod_limit + 1

    for ints in range(nints):
        # Work forward through the groups for initial pass at saturation

        # We want to flag saturation in all subsequent groups after
        # the one in which it was found.  Use this boolean array to
        # keep a running tally of pixels that have saturated.
        previously_saturated = np.zeros(shape=(nrows, ncols), dtype='bool')

        for group in range(ngroups):
            plane = data[ints, group, :, :]

            # for saturation, the flag is set in the current plane
            # and all following planes.

            # Update the running tally of all pixels that have ever
            # experienced saturation to account for this.

            previously_saturated |= (plane >= sat_thresh)
            flagarray = (previously_saturated * saturated).astype(np.uint32)

            gdq[ints, group, :, :] |= flagarray

            # for A/D floor, the flag is only set of the current plane
            flaglowarray = ((plane <= 0)*(ad_floor | dnu)).astype(np.uint32)

            gdq[ints, group, :, :] |= flaglowarray

            del flagarray
            del flaglowarray

            # now, flag any pixels that border saturated pixels (not A/D floor pix)
            if n_pix_grow_sat > 0:
                gdq_slice = gdq[ints, group, :, :]
                adjacent_pixels(gdq_slice, saturated, n_pix_grow_sat, inplace=True)

        # Work backward through the groups for a second pass at saturation
        # This is to flag things that actually saturated in prior groups but
        # were not obvious because of group averaging

        for group in range(ngroups - 2, -1, -1):

            plane = data[ints, group, :, :]
            thisdq = gdq[ints, group, :, :]
            nextdq = gdq[ints, group + 1, :, :]

            # Determine the dilution factor due to group averaging

            # No point in this step if the dilution factor is 1.  In
            # that case, there is no way that we would have missed
            # saturation before but flag it now, since the threshold
            # would be the same.

            if read_pattern is not None:
                # Single value dilution factor for this group
                dilution_factor = np.mean(read_pattern[group]) / read_pattern[group][-1]
                if dilution_factor == 1:
                    continue
                # Broadcast to array size
                dilution_factor = np.where(no_sat_check_mask, 1, dilution_factor)
            else:
                dilution_factor = 1
                continue

            # Find where this plane looks like it might saturate given
            # the dilution factor, *and* this group did not already get
            # flagged as saturated or do not use, *and* the next group
            # was flagged as saturated.  Result of the line below is a
            # boolean array.

            partial_sat = ((plane >= sat_thresh*dilution_factor) & \
                           (thisdq & (saturated | dnu) == 0) & \
                           (nextdq & saturated != 0))

            flagarray = (partial_sat * dnu).astype(np.uint32)
            
            # Grow the newly-flagged saturating pixels
            if n_pix_grow_sat > 0:
                adjacent_pixels(flagarray, dnu, n_pix_grow_sat, inplace=True)

            # Add them to the gdq array
            gdq[ints, group, :, :] |= flagarray

        # Add an additional pass to look for things saturating in the second group
        # that can be particularly tricky to identify
        if ((read_pattern is not None) & (ngroups > 2)):
            dq2 = gdq[ints, 1, :, :]
            dq3 = gdq[ints, 2, :, :]
            
            # Identify groups which we wouldn't expect to saturate by the third group,
            # on the basis of the first group
            scigp1 = data[ints, 0, :, :]
            mask = scigp1 / np.mean(read_pattern[0]) * read_pattern[2][-1] < sat_thresh

            # Identify groups with suspiciously large values in the second group
            # In the limit of groups with just nframe this just checks if second group
            # is over the regular saturation limit.
            scigp2 = data[ints, 1, :, :]
            mask &= scigp2 > sat_thresh / len(read_pattern[1])

            # Identify groups that are saturated in the third group but not yet flagged in the second
            gp3mask = ((np.bitwise_and(dq3, saturated) != 0) & \
                       (np.bitwise_and(dq2, saturated) == 0))
            mask &= gp3mask

            # Flag the 2nd group for the pixels passing that gauntlet
            flagarray = (mask * dnu).astype(np.uint32)

            # Add them to the gdq array
            np.bitwise_or(gdq[ints, 1, :, :], flagarray, gdq[ints, 1, :, :])


        # Check ZEROFRAME.
        if zframe is not None:
            plane = zframe[ints, :, :]
            flagarray, flaglowarray = plane_saturation(plane, sat_thresh, dqflags)
            zdq = flagarray | flaglowarray
            if n_pix_grow_sat > 0:
                adjacent_pixels(zdq, saturated, n_pix_grow_sat, inplace=True)
            plane[zdq != 0] = 0.0
            zframe[ints] = plane

    n_sat = np.any(np.any(np.bitwise_and(gdq, saturated), axis=0), axis=0).sum()
    log.info("Detected %i saturated pixels", n_sat)
    n_floor = np.any(np.any(np.bitwise_and(gdq, ad_floor), axis=0), axis=0).sum()
    log.info("Detected %i A/D floor pixels", n_floor)

    pdq = np.bitwise_or(pdq, sat_dq)

    return gdq, pdq, zframe


def adjacent_pixels(plane_gdq, saturated, n_pix_grow_sat=1, inplace=False):
    """
    plane_gdq : ndarray
        The data quality flags of the current.

    saturated : uint8
        The saturation flag.

    n_pix_grow_sat : int
        Number of pixels that each flagged saturated pixel should be 'grown',
        to account for charge spilling. Default is 1.

    inplace : bool
        Update plane_gdq in place, returning None?  Default False.

    Return
    ------
    sat_pix : ndarray
        The saturated pixels in the current plane.
    """
    if not inplace:
        cgdq = plane_gdq.copy()
    else:
        cgdq = plane_gdq

    only_sat = plane_gdq & saturated > 0
    dilated = only_sat.copy()
    box_dim = (n_pix_grow_sat * 2) + 1

    # The for loops below are equivalent to
    #
    #struct = np.ones((box_dim, box_dim)).astype(bool)
    #dilated = ndimage.binary_dilation(only_sat, structure=struct).astype(only_sat.dtype)
    #
    # The explicit loop over the box, followed by taking care of the
    # array edges, turns out to be faster by around an order of magnitude.
    # There must be poor coding in the underlying routine for
    # ndimage.binary_dilation as of scipy 1.14.1.

    for i in range(box_dim):
        for j in range(box_dim):

            # Explicit binary dilation over the inner ('valid')
            # region of the convolution/filter

            i2 = only_sat.shape[0] - box_dim + i + 1
            j2 = only_sat.shape[1] - box_dim + j + 1

            k1, k2, l1, l2 = [n_pix_grow_sat, -n_pix_grow_sat,
                              n_pix_grow_sat, -n_pix_grow_sat]

            dilated[k1:k2, l1:l2] |= only_sat[i:i2, j:j2]

    for i in range(n_pix_grow_sat - 1, -1, -1):
        for j in range(i + n_pix_grow_sat, -1, -1):

            # March from the limit of the 'valid' region toward
            # each edge.  Maximum filter ensures correct dilation.

            dilated[i] |= ndimage.maximum_filter(only_sat[j], box_dim)
            dilated[:, i] |= ndimage.maximum_filter(only_sat[:, j], box_dim)
            dilated[-i - 1] |= ndimage.maximum_filter(only_sat[-j - 1], box_dim)
            dilated[:, -i - 1] |= ndimage.maximum_filter(only_sat[:, -j - 1], box_dim)

    cgdq[dilated] |= saturated

    if inplace:
        return None
    else:
        return cgdq



def plane_saturation(plane, sat_thresh, dqflags):
    """
    plane : ndarray, 2D float
        The plane to check for saturation and A/D floor.

    sat_thresh : `np.array`
        Pixel-wise threshold for saturation, same shape `data`.

    dims : tuple
        The dimensions of the data array.

    dqflags : dict
        A dictionary with at least the following keywords:
        DO_NOT_USE, SATURATED, AD_FLOOR, NO_SAT_CHECK
    """
    donotuse = dqflags["DO_NOT_USE"]
    saturated = dqflags["SATURATED"]
    ad_floor = dqflags["AD_FLOOR"]

    flagarray = np.zeros(plane.shape, dtype=np.uint32)
    flaglowarray = np.zeros(plane.shape, dtype=np.uint32)

    # Update the 4D gdq array with the saturation flag.
    # check for saturation
    flagarray[:, :] = np.where(plane[:, :] >= sat_thresh, saturated, 0)

    # check for A/D floor
    flaglowarray[:, :] = np.where(plane[:, :] <= 0, ad_floor | donotuse, 0)

    return flagarray, flaglowarray
