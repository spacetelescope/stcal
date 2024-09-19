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
    dnu = dqflags["DO_NOT_USE"]
    saturated = dqflags["SATURATED"]
    ad_floor = dqflags["AD_FLOOR"]
    no_sat_check = dqflags["NO_SAT_CHECK"]

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
        for group in range(ngroups):
            plane = data[ints, group, :, :]

            flagarray, flaglowarray = plane_saturation(plane, sat_thresh, dqflags)

            # for saturation, the flag is set in the current plane
            # and all following planes.
            np.bitwise_or(gdq[ints, group:, :, :], flagarray, gdq[ints, group:, :, :])

            # for A/D floor, the flag is only set of the current plane
            np.bitwise_or(gdq[ints, group, :, :], flaglowarray, gdq[ints, group, :, :])

            del flagarray
            del flaglowarray

            # now, flag any pixels that border saturated pixels (not A/D floor pix)
            if n_pix_grow_sat > 0:
                gdq_slice = copy.copy(gdq[ints, group, :, :]).astype(int)

                gdq[ints, group, :, :] = adjacent_pixels(gdq_slice, saturated, n_pix_grow_sat)

        # Work backward through the groups for a second pass at saturation
        # This is to flag things that actually saturated in prior groups but
        # were not obvious because of group averaging
        for group in range(ngroups-2, -1, -1):
            plane = data[ints, group, :, :]
            thisdq = gdq[ints, group, :, :]
            nextdq = gdq[ints, group + 1, :, :]

            # Determine the dilution factor due to group averaging
            if read_pattern is not None:
                # Single value dilution factor for this group
                dilution_factor = np.mean(read_pattern[group]) / read_pattern[group][-1]
                # Broadcast to array size
                dilution_factor = np.where(no_sat_check_mask, 1, dilution_factor)
            else:
                dilution_factor = 1

            # Find where this plane looks like it might saturate given the dilution factor
            flagarray, _ = plane_saturation(plane, sat_thresh * dilution_factor, dqflags)

            # Find the overlap of where this plane looks like it might saturate, was not currently
            # flagged as saturation or DO_NOT_USE, and the next group had saturation flagged.
            indx = np.where((np.bitwise_and(flagarray, saturated) != 0) & \
                            (np.bitwise_and(thisdq, saturated) == 0) & \
                            (np.bitwise_and(thisdq, dnu) == 0) & \
                            (np.bitwise_and(nextdq, saturated) != 0))

            # Reset flag array to only pixels passing this gauntlet
            flagarray[:] = 0
            flagarray[indx] = 2
            
            # Grow the newly-flagged saturating pixels
            if n_pix_grow_sat > 0:
                flagarray = adjacent_pixels(flagarray, saturated, n_pix_grow_sat)

            # Add them to the gdq array
            np.bitwise_or(gdq[ints, group, :, :], flagarray, gdq[ints, group, :, :])

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
            gp3mask = np.where((np.bitwise_and(dq3, saturated) != 0) & \
                               (np.bitwise_and(dq2, saturated) == 0), True, False)
            mask &= gp3mask

            # Flag the 2nd group for the pixels passing that gauntlet
            flagarray = np.zeros_like(mask,dtype='uint8')
            flagarray[mask] = saturated
            # flag any pixels that border these new pixels
            if n_pix_grow_sat > 0:
                flagarray = adjacent_pixels(flagarray, saturated, n_pix_grow_sat)

            # Add them to the gdq array
            np.bitwise_or(gdq[ints, 1, :, :], flagarray, gdq[ints, 1, :, :])
            

        # Check ZEROFRAME.
        if zframe is not None:
            plane = zframe[ints, :, :]
            flagarray, flaglowarray = plane_saturation(plane, sat_thresh, dqflags)
            zdq = flagarray | flaglowarray
            if n_pix_grow_sat > 0:
                zdq = adjacent_pixels(zdq, saturated, n_pix_grow_sat)
            plane[zdq != 0] = 0.0
            zframe[ints] = plane

    n_sat = np.any(np.any(np.bitwise_and(gdq, saturated), axis=0), axis=0).sum()
    log.info("Detected %i saturated pixels", n_sat)
    n_floor = np.any(np.any(np.bitwise_and(gdq, ad_floor), axis=0), axis=0).sum()
    log.info("Detected %i A/D floor pixels", n_floor)

    pdq = np.bitwise_or(pdq, sat_dq)

    return gdq, pdq, zframe


def adjacent_pixels(plane_gdq, saturated, n_pix_grow_sat):
    """
    plane_gdq : ndarray
        The data quality flags of the current.

    saturated : uint8
        The saturation flag.

    n_pix_grow_sat : int
        Number of pixels that each flagged saturated pixel should be 'grown',
        to account for charge spilling. Default is 1.

    Return
    ------
    sat_pix : ndarray
        The saturated pixels in the current plane.
    """
    cgdq = plane_gdq.copy()
    only_sat = np.bitwise_and(plane_gdq, saturated).astype(np.uint8)
    box_dim = (n_pix_grow_sat * 2) + 1
    struct = np.ones((box_dim, box_dim)).astype(bool)
    dialated = ndimage.binary_dilation(only_sat, structure=struct).astype(only_sat.dtype)
    return np.bitwise_or(cgdq, (dialated * saturated))


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
