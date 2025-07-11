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

import logging

import numpy as np

from . import (
    likely_fit, # used only if algorithm is "LIKELY"
    ols_fit,    # used only if algorithm is "OLS_C"
    ramp_fit_class,
)

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

BUFSIZE = 1024 * 300000  # 300Mb cache size for data section


def create_ramp_fit_class(model, algorithm, dqflags=None, suppress_one_group=False):
    """
    Create an internal ramp fit class from a data model.

    Parameters
    ----------
    model : data model
        input data model, assumed to be of type RampModel

    algorithm : str
        'OLS_C' specifies that ordinary least squares should be used;
        'LIKELY' specifies that maximum likelihood should be used.

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

    if model.hasattr("average_dark_current"):
        dark_current_array = model.average_dark_current
    else:
        dark_current_array = np.zeros(model.pixeldq.shape, dtype=np.float32)

    if model.hasattr("zeroframe"):
        zeroframe = model.zeroframe
    else:
        zeroframe = None

    orig_gdq = None
    if algorithm.upper() == "OLS_C":
        wh_chargeloss = model.groupdq & dqflags['CHARGELOSS']
        if np.any(wh_chargeloss > 0):
            orig_gdq = model.groupdq.copy()
        del wh_chargeloss

    ramp_data.set_arrays(
        model.data,
        model.groupdq,
        model.pixeldq,
        dark_current_array,
        orig_gdq=orig_gdq,
        zeroframe=zeroframe,
    )

    # Attribute may not be supported by all pipelines.  Default is None.
    if hasattr(model.meta.exposure, "drop_frames1"):
        drop_frames1 = model.meta.exposure.drop_frames1
    else:
        drop_frames1 = None
    ramp_data.set_meta(
        name=model.meta.instrument.name,
        frame_time=model.meta.exposure.frame_time,
        group_time=model.meta.exposure.group_time,
        groupgap=model.meta.exposure.groupgap,
        nframes=model.meta.exposure.nframes,
        drop_frames1=drop_frames1,
    )

    ramp_data.algorithm = algorithm
    if hasattr(model.meta.exposure, "read_pattern"):
        ramp_data.read_pattern = [list(reads) for reads in model.meta.exposure.read_pattern]

    ramp_data.set_dqflags(dqflags)
    ramp_data.start_row = 0
    ramp_data.num_rows = ramp_data.data.shape[2]

    ramp_data.suppress_one_group_ramps = suppress_one_group

    return ramp_data


def ramp_fit(
    model,
    save_opt,
    readnoise_2d,
    gain_2d,
    algorithm,
    weighting,
    max_cores,
    dqflags,
    suppress_one_group=False,
):
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

    save_opt : bool
       calculate optional fitting results

    readnoise_2d : ndarray
        2-D array readnoise for all pixels

    gain_2d : ndarray
        2-D array gain for all pixels

    algorithm : str
        'OLS_C' specifies that ordinary least squares should be used;
        'LIKELY' specifies that maximum likelihood should be used.

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
    """
    if suppress_one_group and model.data.shape[1] == 1:
        # One group ramp suppression should only be done on data with
        # ramps having more than one group.
        suppress_one_group = False

    # Create an instance of the internal ramp class, using only values needed
    # for ramp fitting from the to remove further ramp fitting dependence on
    # data models.
    ramp_data = create_ramp_fit_class(model, algorithm, dqflags, suppress_one_group)

    return ramp_fit_data(
        ramp_data, save_opt, readnoise_2d, gain_2d, algorithm, weighting, max_cores
    )


def ramp_fit_data(
    ramp_data, save_opt, readnoise_2d, gain_2d, algorithm, weighting, max_cores
):
    """
    This function begins the ramp fit computation after the creation of the
    RampData class.  It determines the proper path for computation to take
    depending on the choice of ramp fitting algorithms (which is only ordinary
    least squares right now) and the choice of single or muliprocessing.


    ramp_data : RampData
        Input data necessary for computing ramp fitting.

    save_opt : bool
       calculate optional fitting results

    readnoise_2d : ndarray
        2-D array readnoise for all pixels

    gain_2d : ndarray
        2-D array gain for all pixels

    algorithm : str
        'OLS_C' specifies that ordinary least squares should be used;
        'LIKELY' specifies that maximum likelihood should be used.

    weighting : str
        'optimal' specifies that optimal weighting should be used;
         currently the only weighting supported.

    max_cores : str
        Number of cores to use for multiprocessing. If set to 'none' (the
        default), then no multiprocessing will be done. The other allowable
        values are 'quarter', 'half', and 'all'. This is the fraction of cores
        to use for multi-proc. The total number of cores includes the SMT cores
        (Hyper Threading for Intel).

    Returns
    -------
    image_info : tuple
        The tuple of computed ramp fitting arrays.

    integ_info : tuple
        The tuple of computed integration fitting arrays.

    opt_info : tuple
        The tuple of computed optional results arrays for fitting.
    """
    # For the LIKELY algorithm, due to the jump detection portion of the code
    # a minimum of a four group ramp is needed.
    ngroups = ramp_data.data.shape[1]
    if algorithm.upper() == "LIKELY" and ngroups < likely_fit.LIKELY_MIN_NGROUPS:
        log.info(f"When selecting the LIKELY ramp fitting algorithm the"
                  " ngroups needs to be a minimum of {likely_fit.LIKELY_MIN_NGROUPS},"
                  " but ngroups = {ngroups}.  Due to this, the ramp fitting algorithm"
                  " is being changed to OLS_C")
        algorithm = "OLS_C"
        
    if algorithm.upper() == "LIKELY" and ngroups >= likely_fit.LIKELY_MIN_NGROUPS:
        image_info, integ_info, opt_info = likely_fit.likely_ramp_fit(
            ramp_data, readnoise_2d, gain_2d
        )
    else:
        # Default to OLS_C.
        # Get readnoise array for calculation of variance of noiseless ramps, and
        #   gain array in case optimal weighting is to be done
        nframes = ramp_data.nframes
        readnoise_2d *= gain_2d / np.sqrt(2.0 * nframes)

        # Suppress one group ramps, if desired.
        if ramp_data.suppress_one_group_ramps:
            suppress_one_good_group_ramps(ramp_data)

        # Compute ramp fitting using ordinary least squares.
        image_info, integ_info, opt_info = ols_fit.ols_ramp_fit_multi(
            ramp_data, save_opt, readnoise_2d, gain_2d, weighting, max_cores
        )

    return image_info, integ_info, opt_info


def suppress_one_good_group_ramps(ramp_data):
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
    dnu_flag = ramp_data.flags_do_not_use

    for integ in range(nints):
        # In the current integration find ramps with only one group.
        intdq = dq[integ, :, :, :]
        good_groups = np.zeros(intdq.shape, dtype=int)
        good_groups[intdq == 0] = 1
        ngood_groups = good_groups.sum(axis=0)
        wh_one = np.where(ngood_groups == 1)

        # Suppress the ramps with only one good group by flagging
        # all groups in the ramp as DO_NOT_USE.
        wh1_rows = wh_one[0]
        wh1_cols = wh_one[1]
        for n in range(len(wh1_rows)):
            row = wh1_rows[n]
            col = wh1_cols[n]
            # Find ramps that have good 0th group, but the rest of the
            # ramp flagged.
            good_index = np.where(ramp_data.groupdq[integ, :, row, col] == 0)
            if ramp_data.groupdq[integ, good_index, row, col] == 0:
                ramp_data.groupdq[integ, :, row, col] = np.bitwise_or(
                    ramp_data.groupdq[integ, :, row, col], dnu_flag
                )
