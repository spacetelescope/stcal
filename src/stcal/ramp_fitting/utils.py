#! /usr/bin/env python
#
# utils.py: utility functions
import logging

import numpy as np

log = logging.getLogger(__name__)

# Replace zero or negative variances with this:
LARGE_VARIANCE = 1.0e8
LARGE_VARIANCE_THRESHOLD = 0.01 * LARGE_VARIANCE

__all__ = ["set_if_total_ramp", "set_if_total_integ", "dq_compress_sect", "dq_compress_final"]


def set_if_total_ramp(pixeldq_sect, gdq_sect, flag, set_flag):
    """
    Set set_flag in final_dq if flag is present in all integrations.

    Parameters
    ----------
    pixeldq_sect: ndarray
        2-D array (nrows, ncols) of the integration DQ.

    gdq_dq : ndarray
        3-D array (ngroups, nrows, ncols) of the integration level DQ.

    flag : int
        Flag to check in each integration.

    set_flag : int
        Flag to set if flag is found in each integration.
    """
    # Checking for all groups is the same as checking for all integrations
    # because in both we are checking cubes.  For the integration check the
    # first dimension is the number of integrations, for the ramp check the
    # first dimension is the number of groups.
    set_if_total_integ(pixeldq_sect, gdq_sect, flag, set_flag)


def set_if_total_integ(final_dq, integ_dq, flag, set_flag):
    """
    Set set_flag in final_dq if flag is present in all integrations.

    Parameters
    ----------
    final_dq : ndarray
        2-D array (nrows, ncols) of the final pixel DQ.

    integ_dq : ndarray
        3-D array (nints, nrows, ncols) of the integration level DQ.

    flag : int
        Flag to check in each integration.

    set_flag : int
        Flag to set if flag is found in each integration.
    """
    nints = integ_dq.shape[0]

    # Count the planes carrying the flag one plane at a time. Testing the whole
    # cube at once would allocate two further copies of it, and promote them to
    # the (wider) dtype of the flag values.
    n_set = np.zeros(integ_dq.shape[1:], dtype=np.int32)
    for plane in integ_dq:
        n_set += np.bitwise_and(plane, flag) != 0

    all_set = np.where(n_set == nints)

    # If flag is set in all integrations, then set the set_flag
    final_dq[all_set] = np.bitwise_or(final_dq[all_set], set_flag)


def dq_compress_sect(ramp_data, gdq_sect, pixeldq_sect):
    """
    Set the integration level flags for DO_NOT_USE, JUMP_DET, and SATURATED.

    If any ramp has a jump or saturated, the respective flag will be set for the
    integration.  If all groups in a ramp are flagged as DO_NOT_USE, then the
    integration level DO_NOT_USE flag will be set.  Also, if
    all groups are marked as DO_NOT_USE or SATURATED (as in suppressed one
    groups), then the DO_NOT_USE flag is set.

    Parameters
    ----------
    ramp_data : RampData
        Contains the DQ flag information.

    gdq_sect : ndarray
        The current 3-D (ngroups, nrows, ncols) integration DQ array.

    pixeldq_sect : ndarray
        The 2-D (nrows, ncols) pixel DQ flags for the current integration.

    Return
    ------
    pixeldq_sect : ndarray
        The 2-D (nrows, ncols) pixel DQ flags for the current integration.
    """
    sat = ramp_data.flags_saturated
    jump = ramp_data.flags_jump_det
    dnu = ramp_data.flags_do_not_use
    ngroups, nrows, ncols = gdq_sect.shape

    # Check total SATURATED or DO_NOT_USE
    set_if_total_ramp(pixeldq_sect, gdq_sect, sat | dnu, dnu)

    # A flag is set somewhere in the ramp exactly when it is set in the bitwise
    # OR over the groups. Reducing first keeps the temporary two dimensional:
    # testing the cube directly would copy it, and promote that copy to the
    # (wider) dtype of the flag values.
    any_flag = np.bitwise_or.reduce(gdq_sect, axis=0)

    # If saturation occurs mark the appropriate flag.
    sat_check = np.where(np.bitwise_and(any_flag, sat) != 0)
    pixeldq_sect[sat_check] = np.bitwise_or(pixeldq_sect[sat_check], sat)

    # If jump occurs mark the appropriate flag.
    jump_check = np.where(np.bitwise_and(any_flag, jump) != 0)
    pixeldq_sect[jump_check] = np.bitwise_or(pixeldq_sect[jump_check], jump)

    return pixeldq_sect


def dq_compress_final(dq_int, ramp_data):
    """
    From the integration level DQ flags, set the final pixel DQ flags.

    Parameters
    ----------
    dq_int : ndarray
        The integration level DQ flags, 3-D (nints, nrows, ncols).

    ramp_data : RampData
        Contains the DQ flag information.

    Return
    ------
    final_dq : ndarray
        The final 2-D (nrows, ncols) pixel DQ array.
    """
    final_dq = dq_int[0, :, :]
    nints = dq_int.shape[0]
    for integ in range(1, nints):
        final_dq = np.bitwise_or(final_dq, dq_int[integ, :, :])

    dnu = np.uint32(ramp_data.flags_do_not_use)

    # Remove DO_NOT_USE because it needs special handling.
    # This flag is not set in the final pixel DQ array by simply being set
    # in one of the integrations.
    not_dnu = np.uint32(~dnu)
    final_dq = np.bitwise_and(final_dq, not_dnu)

    # If all integrations are DO_NOT_USE, then set DO_NOT_USE.
    set_if_total_integ(final_dq, dq_int, dnu, dnu)

    return final_dq
