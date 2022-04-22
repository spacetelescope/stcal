import numpy as np
import logging

import copy
from scipy import ndimage

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def flag_saturated_pixels(data, gdq, pdq, sat_thresh, sat_dq, atod_limit,
                          dqflags, n_pix_grow_sat=1):
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

    dqflags: dict
        A dictionary with at least the following keywords:
        DO_NOT_USE, SATURATED, AD_FLOOR, NO_SAT_CHECK

    n_pix_grow_sat : int
        Number of pixels that each flagged saturated pixel should be 'grown',
        to account for charge spilling. Default is 1.


    Returns
    -------
    gdq : int, 4D array
        updated group dq array

    pdq : int, 2D array
        updated pixel dq array
    """

    nints = data.shape[0]
    ngroups = data.shape[1]

    donotuse = dqflags['DO_NOT_USE']
    saturated = dqflags['SATURATED']
    ad_floor = dqflags['AD_FLOOR']
    no_sat_check = dqflags['NO_SAT_CHECK']

    # For pixels flagged in reference file as NO_SAT_CHECK,
    # set the saturation check threshold to the A-to-D converter limit.
    sat_thresh[np.bitwise_and(sat_dq, no_sat_check) == no_sat_check] = atod_limit

    # Also reset NaN values in the saturation threshold array to the
    # A-to-D limit and flag them with NO_SAT_CHECK
    sat_dq[np.isnan(sat_thresh)] |= no_sat_check
    sat_thresh[np.isnan(sat_thresh)] = atod_limit

    flagarray = np.zeros(data.shape[-2:], dtype=gdq.dtype)
    flaglowarray = np.zeros(data.shape[-2:], dtype=gdq.dtype)
    for ints in range(nints):
        for group in range(ngroups):
            # Update the 4D gdq array with the saturation flag.
            # check for saturation
            flagarray[:, :] = np.where(data[ints, group, :, :] >= sat_thresh,
                                       saturated, 0)
            # check for A/D floor
            flaglowarray[:, :] = np.where(data[ints, group, :, :] <= 0,
                                          ad_floor | donotuse, 0)
            # for saturation, the flag is set in the current plane
            # and all following planes.
            np.bitwise_or(gdq[ints, group:, :, :], flagarray,
                          gdq[ints, group:, :, :])
            # for A/D floor, the flag is only set of the current plane
            np.bitwise_or(gdq[ints, group, :, :], flaglowarray,
                          gdq[ints, group, :, :])

            # now, flag any pixels that border saturated pixels (not A/D floor pix)
            if n_pix_grow_sat > 0:
                gdq_slice = copy.copy(gdq[ints, group, :, :]).astype(int)
                only_sat = np.bitwise_and(gdq_slice, saturated).astype(np.uint8)
                box_dim = (n_pix_grow_sat * 2) + 1
                struct = np.ones((box_dim, box_dim)).astype(bool)
                dialated = ndimage.binary_dilation(only_sat, structure=struct).astype(only_sat.dtype)
                gdq[ints, group, :, :] = np.bitwise_or(gdq[ints, group, :, :], (dialated * saturated))

    n_sat = np.any(np.any(np.bitwise_and(gdq, saturated), axis=0), axis=0).sum()
    log.info(f'Detected {n_sat} saturated pixels')
    n_floor = np.any(np.any(np.bitwise_and(gdq, ad_floor), axis=0), axis=0).sum()
    log.info(f'Detected {n_floor} A/D floor pixels')

    pdq = np.bitwise_or(pdq, sat_dq)

    return gdq, pdq
