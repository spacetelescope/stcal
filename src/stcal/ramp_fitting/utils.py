#! /usr/bin/env python
#
# utils.py: utility functions
import logging

import numpy as np


log = logging.getLogger(__name__)

# Replace zero or negative variances with this:
LARGE_VARIANCE = 1.0e8
LARGE_VARIANCE_THRESHOLD = 0.01 * LARGE_VARIANCE


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

    # Find where flag is set
    test_dq = np.zeros(integ_dq.shape, dtype=np.uint32)
    test_dq[np.bitwise_and(integ_dq, flag).astype(bool)] = 1

    # Sum over all integrations
    test_sum = test_dq.sum(axis=0)
    all_set = np.where(test_sum == nints)

    # If flag is set in all integrations, then set the set_flag
    final_dq[all_set] = np.bitwise_or(final_dq[all_set], set_flag)


def dq_compress_sect(ramp_data, num_int, gdq_sect, pixeldq_sect):
    """
    This sets the integration level flags for DO_NOT_USE, JUMP_DET and
    SATURATED.  If any ramp has a jump, this flag will be set for the
    integration.  If all groups in a ramp are flagged as DO_NOT_USE, then the
    integration level DO_NOT_USE flag will be set.  If a ramp is saturated in
    group 0, then the integration level flag is marked as SATURATED.  Also, if
    all groups are marked as DO_NOT_USE or SATURATED (as in suppressed one
    groups), then the DO_NOT_USE flag is set.

    Parameters
    ----------
    ramp_data : RampData
        Contains the DQ flag information.

    num_int : int
        The current integration number.

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

    # Assume total saturation if group 0 is SATURATED.
    gdq0_sat = np.bitwise_and(gdq_sect[0], sat)
    pixeldq_sect[gdq0_sat != 0] = np.bitwise_or(pixeldq_sect[gdq0_sat != 0], sat | dnu)

    # If jump occurs mark the appropriate flag.
    jump_loc = np.bitwise_and(gdq_sect, jump)
    jump_check = np.where(jump_loc.sum(axis=0) > 0)
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
    sat = np.uint32(ramp_data.flags_saturated)

    # Remove DO_NOT_USE and SATURATED because they need special handling.
    # These flags are not set in the final pixel DQ array by simply being set
    # in one of the integrations.
    not_sat_or_dnu = np.uint32(~(dnu | sat))
    final_dq = np.bitwise_and(final_dq, not_sat_or_dnu)

    # If all integrations are DO_NOT_USE or SATURATED, then set DO_NOT_USE.
    set_if_total_integ(final_dq, dq_int, dnu | sat, dnu)

    # If all integrations have SATURATED, then set DO_NOT_USE and SATURATED.
    set_if_total_integ(final_dq, dq_int, sat, dnu | sat)

    return final_dq
