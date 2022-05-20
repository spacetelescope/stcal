#! /usr/bin/env python
#
#  ramp_fit.py - calculate weighted mean of slope, based on Massimo
#                Robberto's "On the Optimal Strategy to fit MULTIACCUM
#                ramps in the presence of cosmic rays."
#                (JWST-STScI-0001490,SM-12; 07/25/08).   The derivation
#                is a generalization for >1 cosmic rays, calculating
#                the slope and variance of the slope for each section
#                of the ramp (in between cosmic rays). The intervals are
#                determined from the input data quality arrays.
#
# Note:
# In this module, comments on the 'first group','second group', etc are
#    1-based, unless noted otherwise.

import numpy as np
import logging

from . import gls_fit           # used only if algorithm is "GLS"
from . import ols_fit           # used only if algorithm is "OLS"
from . import ramp_fit_class

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

BUFSIZE = 1024 * 300000  # 300Mb cache size for data section


def create_ramp_fit_class(model, dqflags=None, suppress_one_group=False):
    """
    Create an internal ramp fit class from a data model.

    Parameters
    ----------
    model : data model
        input data model, assumed to be of type RampModel

    dqflags : dict
        The data quality flags needed for ramp fitting.

    suppress_one_group : bool
        Find ramps with only one good group and treat it like it has zero good
        groups.

    Return
    ------
    ramp_data : ramp_fit_class.RampData
        The internal ramp class.
    """
    ramp_data = ramp_fit_class.RampData()

    if not suppress_one_group and hasattr(model.meta.exposure, 'zero_frame'):
        if model.meta.exposure.zero_frame:
            # ZEROFRAME processing here
            zframe_locs, cnt = use_zeroframe_for_saturated_ramps(model, dqflags)
            ramp_data.zframe_locs = zframe_locs
            ramp_data.zframe_cnt = cnt

    # Attribute may not be supported by all pipelines.  Default is NoneType.
    if hasattr(model, 'int_times'):
        int_times = model.int_times
    else:
        int_times = None
    ramp_data.set_arrays(
        model.data, model.err, model.groupdq, model.pixeldq, int_times)

    # Attribute may not be supported by all pipelines.  Default is NoneType.
    if hasattr(model, 'drop_frames1'):
        drop_frames1 = model.exposure.drop_frames1
    else:
        drop_frames1 = None
    ramp_data.set_meta(
        name=model.meta.instrument.name,
        frame_time=model.meta.exposure.frame_time,
        group_time=model.meta.exposure.group_time,
        groupgap=model.meta.exposure.groupgap,
        nframes=model.meta.exposure.nframes,
        drop_frames1=drop_frames1)

    ramp_data.set_dqflags(dqflags)
    ramp_data.start_row = 0
    ramp_data.num_rows = ramp_data.data.shape[2]

    ramp_data.suppress_one_group_ramps = suppress_one_group

    return ramp_data


def ramp_fit(model, buffsize, save_opt, readnoise_2d, gain_2d, algorithm,
             weighting, max_cores, dqflags, suppress_one_group=False):
    """
    Calculate the count rate for each pixel in all data cube sections and all
    integrations, equal to the slope for all sections (intervals between
    cosmic rays) of the pixel's ramp divided by the effective integration time.
    The weighting parameter must currently be set to 'optim', to use the optimal
    weighting (paper by Fixsen, ref. TBA) will be used in the fitting; this is
    currently the only supported weighting scheme.

    Parameters
    ----------
    model : data model
        input data model, assumed to be of type RampModel

    buffsize : int
        size of data section (buffer) in bytes

    save_opt : bool
       calculate optional fitting results

    readnoise_2d : ndarray
        2-D array readnoise for all pixels

    gain_2d : ndarray
        2-D array gain for all pixels

    algorithm : str
        'OLS' specifies that ordinary least squares should be used;
        'GLS' specifies that generalized least squares should be used.

    weighting : str
        'optimal' specifies that optimal weighting should be used;
         currently the only weighting supported.

    max_cores : str
        Number of cores to use for multiprocessing. If set to 'none' (the
        default), then no multiprocessing will be done. The other allowable
        values are 'quarter', 'half', and 'all'. This is the fraction of cores
        to use for multi-proc. The total number of cores includes the SMT cores
        (Hyper Threading for Intel).

    dqflags : dict
        A dictionary with at least the following keywords:
        DO_NOT_USE, SATURATED, JUMP_DET, NO_GAIN_VALUE, UNRELIABLE_SLOPE

    suppress_one_group : bool
        Find ramps with only one good group and treat it like it has zero good
        groups.

    Returns
    -------
    image_info : tuple
        The tuple of computed ramp fitting arrays.

    integ_info : tuple
        The tuple of computed integration fitting arrays.

    opt_info : tuple
        The tuple of computed optional results arrays for fitting.

    gls_opt_model : GLS_RampFitModel object or None (Unused for now)
        Object containing optional GLS-specific ramp fitting data for the
        exposure
    """
    if suppress_one_group and model.data.shape[1] == 1:
        # One group ramp suppression should only be done on data with
        # ramps having more than one group.
        suppress_one_group = False

    # Create an instance of the internal ramp class, using only values needed
    # for ramp fitting from the to remove further ramp fitting dependence on
    # data models.
    ramp_data = create_ramp_fit_class(model, dqflags, suppress_one_group)

    return ramp_fit_data(
        ramp_data, buffsize, save_opt, readnoise_2d, gain_2d,
        algorithm, weighting, max_cores, dqflags)


def ramp_fit_data(ramp_data, buffsize, save_opt, readnoise_2d, gain_2d,
                  algorithm, weighting, max_cores, dqflags):
    """
    This function begins the ramp fit computation after the creation of the
    RampData class.  It determines the proper path for computation to take
    depending on the choice of ramp fitting algorithms (which is only ordinary
    least squares right now) and the choice of single or muliprocessing.


    ramp_data : RampData
        Input data necessary for computing ramp fitting.

    buffsize : int
        size of data section (buffer) in bytes

    save_opt : bool
       calculate optional fitting results

    readnoise_2d : ndarray
        2-D array readnoise for all pixels

    gain_2d : ndarray
        2-D array gain for all pixels

    algorithm : str
        'OLS' specifies that ordinary least squares should be used;
        'GLS' specifies that generalized least squares should be used.

    weighting : str
        'optimal' specifies that optimal weighting should be used;
         currently the only weighting supported.

    max_cores : str
        Number of cores to use for multiprocessing. If set to 'none' (the
        default), then no multiprocessing will be done. The other allowable
        values are 'quarter', 'half', and 'all'. This is the fraction of cores
        to use for multi-proc. The total number of cores includes the SMT cores
        (Hyper Threading for Intel).

    dqflags : dict
        A dictionary with at least the following keywords:
        DO_NOT_USE, SATURATED, JUMP_DET, NO_GAIN_VALUE, UNRELIABLE_SLOPE

    Returns
    -------
    image_info : tuple
        The tuple of computed ramp fitting arrays.

    integ_info : tuple
        The tuple of computed integration fitting arrays.

    opt_info : tuple
        The tuple of computed optional results arrays for fitting.

    gls_opt_model : GLS_RampFitModel object or None (Unused for now)
        Object containing optional GLS-specific ramp fitting data for the
        exposure
    """
    if algorithm.upper() == "GLS":
        image_info, integ_info, gls_opt_info = gls_fit.gls_ramp_fit(
            ramp_data, buffsize, save_opt, readnoise_2d, gain_2d, max_cores)
        opt_info = None
    else:
        # Get readnoise array for calculation of variance of noiseless ramps, and
        #   gain array in case optimal weighting is to be done
        nframes = ramp_data.nframes
        readnoise_2d *= gain_2d / np.sqrt(2. * nframes)

        # Suppress one group ramps, if desired.
        if ramp_data.suppress_one_group_ramps:
            suppress_one_group_saturated_or_jump_ramps(ramp_data)

        # Compute ramp fitting using ordinary least squares.
        image_info, integ_info, opt_info = ols_fit.ols_ramp_fit_multi(
            ramp_data, buffsize, save_opt, readnoise_2d, gain_2d, weighting, max_cores)
        gls_opt_info = None

    return image_info, integ_info, opt_info, gls_opt_info


def suppress_one_group_saturated_or_jump_ramps(ramp_data):
    """
    Finds one group ramps in each integration and suppresses them, i.e. turns
    them into zero group ramps.

    Parameter
    ---------
    ramp_data : RampData
        input data model, assumed to be of type RampModel
    """
    dq = ramp_data.groupdq
    nints, ngroups, nrows, ncols = dq.shape
    sat_flag = ramp_data.flags_saturated
    jump_flag = ramp_data.flags_jump_det

    ramp_data.one_groups = [None] * nints

    for integ in range(nints):
        ramp_data.one_groups[integ] = []
        intdq = dq[integ, :, :, :]

        # Find ramps with only one group that is not saturated and
        # not jump (i.e., only one good group).
        bad_flags = np.bitwise_or(sat_flag, jump_flag)
        bad_groups = np.zeros(intdq.shape, dtype=int)
        bad_groups[np.where(np.bitwise_and(intdq, bad_flags))] = 1
        nbad_groups = bad_groups.sum(axis=0)
        wh_one = np.where(nbad_groups == (ngroups - 1))

        wh1_rows = wh_one[0]
        wh1_cols = wh_one[1]
        for n in range(len(wh1_rows)):
            row = wh1_rows[n]
            col = wh1_cols[n]
            # For ramps that have good 0th group, but the rest of the
            # ramp saturated, mark the 0th groups as saturated, too.

            if ramp_data.groupdq[integ, 0, row, col] == 0:
                ramp_data.groupdq[integ, 0, row, col] = sat_flag
                sat_pix = (row, col)
                ramp_data.one_groups[integ].append(sat_pix)


def use_zeroframe_for_saturated_ramps(model, dqflags):
    """
    For saturated ramps, if there is good data in the ZEROFRAME, replace
    group zero with the data in ZEROFRAME to use the ramp as a one group
    ramp.

    Parameters
    ----------
    model : data model
        input data model, assumed to be of type RampModel

    dqflags : dict
        The data quality flags needed for ramp fitting.

    Return
    ------
    zframe_locs : list
        A 2D list for the location of the ramps using ZEROFRAME data.
        zframe_locs[k] is the list of pixels in the kth integration.
    """
    nints, ngroups, nrows, ncols = model.data.shape
    sat_flag = dqflags["SATURATED"]
    good_flag = dqflags["GOOD"]
    dq = model.groupdq

    zframe_locs = [None] * nints

    cnt = 0
    for integ in range(nints):
        zframe_locs[integ] = []
        intdq = dq[integ, :, :, :]

        # Find ramps with a good zeroeth group, but saturated in
        # the remainder of the ramp.
        wh_sat = groups_saturated_in_integration(intdq, sat_flag, ngroups)

        whs_rows = wh_sat[0]
        whs_cols = wh_sat[1]
        for n in range(len(whs_rows)):
            row = whs_rows[n]
            col = whs_cols[n]

            # For ramps completely saturated look for data in the ZEROFRAME
            # that is non-zero.  If it is non-zero, replace group zero in the
            # ramp with the data in ZEROFRAME.
            if model.zeroframe[integ, row, col] != 0:
                zframe_locs[integ].append((row, col))
                model.data[integ, 0, row, col] = model.zeroframe[integ, row, col]
                model.groupdq[integ, 0, row, col] = good_flag
                cnt = cnt + 1

    return zframe_locs, cnt


def groups_saturated_in_integration(intdq, sat_flag, num_sat_groups):
    """
    Find the ramps in an integration that have num_sat_groups saturated.

    Parameters
    ----------
    intdq : ndarray
        DQ flags for an integration

    sat_flag : uint
        The data quality flag for SATURATED

    num_sat_groups : int
        The number of saturated groups in an integration of interest.
    """
    sat_groups = np.zeros(intdq.shape, dtype=int)
    sat_groups[np.where(np.bitwise_and(intdq, sat_flag))] = 1
    nsat_groups = sat_groups.sum(axis=0)
    wh_nsat_groups = np.where(nsat_groups == num_sat_groups)

    return wh_nsat_groups
