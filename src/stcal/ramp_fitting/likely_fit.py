#! /usr/bin/env python

import logging
import multiprocessing
import time
import sys
import warnings

from multiprocessing import cpu_count
from pprint import pprint

import numpy as np

from . import ramp_fit_class, utils
from .likely_algo_classes import IntegInfo, Ramp_Result, Covar


################## DEBUG ################## 
#                  HELP!!
import sys
sys.path.insert(1, "/Users/kmacdonald/code/common")
from general_funcs import dbg_print, \
                          array_string
################## DEBUG ################## 


DELIM = "=" * 80

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

BUFSIZE = 1024 * 300000  # 300Mb cache size for data section

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def likely_ramp_fit(
    ramp_data, buffsize, save_opt, readnoise_2d, gain_2d, weighting, max_cores
):
    """
    Setup the inputs to ols_ramp_fit with and without multiprocessing. The
    inputs will be sliced into the number of cores that are being used for
    multiprocessing. Because the data models cannot be pickled, only numpy
    arrays are passed and returned as parameters to ols_ramp_fit.

    Parameters
    ----------
    ramp_data : RampData
        Input data necessary for computing ramp fitting.

    buffsize : int
        size of data section (buffer) in bytes (not used)

    save_opt : bool
       calculate optional fitting results

    readnoise_2d : ndarray
        readnoise for all pixels

    gain_2d : ndarray
        gain for all pixels

    algorithm : str
        'OLS' specifies that ordinary least squares should be used;
        'GLS' specifies that generalized least squares should be used.

    weighting : str
        'optimal' specifies that optimal weighting should be used;
         currently the only weighting supported.

    max_cores : str
        Number of cores to use for multiprocessing. If set to 'none' (the default),
        then no multiprocessing will be done. The other allowable values are 'quarter',
        'half', and 'all'. This is the fraction of cores to use for multi-proc. The
        total number of cores includes the SMT cores (Hyper Threading for Intel).

    Returns
    -------
    image_info : tuple
        The tuple of computed ramp fitting arrays.

    integ_info : tuple
        The tuple of computed integration fitting arrays.

    opt_info : tuple
        The tuple of computed optional results arrays for fitting.
    """
    image_info, integ_info, opt_info = None, None, None

    nints, ngroups, nrows, ncols = ramp_data.data.shape

    if ngroups < 2:
        raise ValueError("Likelihood fit requires at least 2 groups.")

    readtimes = get_readtimes(ramp_data)

    covar = Covar(readtimes, pedestal=False)  # XXX Choice of pedestal not given
    integ_class = IntegInfo(nints, nrows, ncols)

    for integ in range(nints):
        data = ramp_data.data[integ, :, :, :]
        gdq = ramp_data.groupdq[integ, :, :, :].copy()
        pdq = ramp_data.pixeldq[:, :].copy()

        # Eqn (5)
        diff = (data[1:] - data[:-1]) / covar.delta_t[:, np.newaxis, np.newaxis]
        alldiffs2use = np.ones(diff.shape, np.uint8)  # XXX May not be necessary

        for row in range(nrows):
            d2use = determine_diffs2use(ramp_data, integ, row, diff)
            d2use_copy = d2use.copy()  # Use to flag jumps
            d2use, countrates = mask_jumps(
                diff[:, row], covar, readnoise_2d[row], gain_2d[row], diffs2use=d2use
            )


            '''
            # XXX SET JUMP_DET
            # Set jump detection flags
            jump_locs = d2use_copy ^ d2use
            jump_locs[jump_locs > 0] = ramp_data.flags_jump_det
            print(f"Row: {row} {jump_locs.shape = }")
            # XXX Need to figure out how to put flags in gdq
            # gdq |= jump_locs
            '''

            alldiffs2use[:, row] = d2use  # XXX May not be necessary

            # XXX According to Brandt feedback
            # rateguess = countrates * (countrates > 0) * darkrate (ramp_data.average_dark_current?)
            rateguess = countrates * (countrates > 0) + ramp_data.average_dark_current[row, :]
            result = fit_ramps(
                diff[:, row],
                covar,
                gain_2d[row],
                readnoise_2d[row],
                diffs2use=d2use,
                countrateguess=rateguess,
            )
            integ_class.get_results(result, integ, row)

        pdq = utils.dq_compress_sect(ramp_data, integ, gdq, pdq)
        integ_class.dq[integ, :, :] = pdq

        del gdq

    integ_info = integ_class.prepare_info()
    image_info = compute_image_info(integ_class, ramp_data)

    return image_info, integ_info, opt_info


def mask_jumps(
    diffs,
    Cov,
    rnoise,
    gain,
    threshold_oneomit=20.25,
    threshold_twoomit=23.8,
    diffs2use=None,
):

    """
    Function mask_jumps implements a likelihood-based, iterative jump
    detection algorithm.

    Parameters
    ----------
    diffs : ndarray
        The group differences of the data array for a given integration and row
        (ngroups-1, ncols).

    Cov : Covar
        The class that computes and contains the covariance matrix info.

    rnoise : ndarray
        The read noise (ncols,)

    gain : ndarray
        The gain (ncols,)

    threshold_oneomit : float
        Minimum chisq improvement to exclude a single resultant difference.
        Default: 20.25.

    threshold_twoomit : float
        Minimum chisq improvement to exclude  two sequential resultant differences.
        Default 23.8.

    d2use : ndarray
        A boolean array definined the segmented ramps for each pixel in a row.
        (ngroups-1, ncols)

    Returns
    -------
    d2use : ndarray
        A boolean array definined the segmented ramps for each pixel in a row.
        (ngroups-1, ncols)

    countrates : ndarray
        Count rate estimates used to estimate the covariance matrix.
        Optional, default is None.

    """
    if Cov.pedestal:
        raise ValueError(
            "Cannot mask jumps with a Covar class that includes a pedestal fit."
        )

    # Force a copy of the input array for more efficient memory access.
    loc_diff = diffs * 1

    # We can use one-omit searches only where the reads immediately
    # preceding and following have just one read.  If a readout
    # pattern has more than one read per resultant but significant
    # gaps between resultants then a one-omit search might still be a
    # good idea even with multiple-read resultants.
    oneomit_ok = Cov.Nreads[1:] * Cov.Nreads[:-1] >= 1
    oneomit_ok[0] = oneomit_ok[-1] = True

    # Other than that, we need to omit two.  If a resultant has more
    # than two reads, we need to omit both differences containing it
    # (one pair of omissions in the differences).
    twoomit_ok = Cov.Nreads[1:-1] > 1

    # This is the array to return: one for resultant differences to
    # use, zero for resultant differences to ignore.
    if diffs2use is None:
        diffs2use = np.ones(loc_diff.shape, np.uint8)

    # We need to estimate the covariance matrix.  I'll use the median
    # here for now to limit problems with the count rate in reads with
    # jumps (which is what we are looking for) since we'll be using
    # likelihoods and chi squared; getting the covariance matrix
    # reasonably close to correct is important.
    countrateguess = np.median(loc_diff, axis=0)[np.newaxis, :]

    # XXX Somehow add the Poisson variance back in.
    countrateguess *= countrateguess > 0

    # boolean arrays to be used later
    recheck = np.ones(loc_diff.shape[1]) == 1
    dropped = np.ones(loc_diff.shape[1]) == 0

    for j in range(loc_diff.shape[0]):
        # No need for indexing on the first pass.
        if j == 0:
            result = fit_ramps(
                loc_diff,
                Cov,
                gain,
                rnoise,
                countrateguess=countrateguess,
                diffs2use=diffs2use,
                detect_jumps=True,
            )
            # Also save the count rates so that we can use them later
            # for debiasing.
            countrate = result.countrate * 1.0
        else:
            result = fit_ramps(
                loc_diff[:, recheck],
                Cov,
                gain[recheck],
                rnoise[recheck],
                countrateguess=countrateguess[:, recheck],
                diffs2use=diffs2use[:, recheck],
                detect_jumps=True,
            )

        # Chi squared improvements
        dchisq_two = result.chisq - result.chisq_twoomit
        dchisq_one = result.chisq - result.chisq_oneomit

        # We want the largest chi squared difference
        best_dchisq_one = np.amax(dchisq_one * oneomit_ok[:, np.newaxis], axis=0)
        best_dchisq_two = np.amax(
            dchisq_two * twoomit_ok[:, np.newaxis], axis=0
        )  # XXX HERE Is this where JUMP_DET is set?

        # Is the best improvement from dropping one resultant
        # difference or two?  Two drops will always offer more
        # improvement than one so penalize them by the respective
        # thresholds.  Then find the chi squared improvement
        # corresponding to dropping either one or two reads, whichever
        # is better, if either exceeded the threshold.
        onedropbetter = (
            best_dchisq_one - threshold_oneomit > best_dchisq_two - threshold_twoomit
        )

        best_dchisq = (
            best_dchisq_one * (best_dchisq_one > threshold_oneomit) * onedropbetter
        )
        best_dchisq += (
            best_dchisq_two * (best_dchisq_two > threshold_twoomit) * (~onedropbetter)
        )

        # If nothing exceeded the threshold set the improvement to
        # NaN so that dchisq==best_dchisq is guaranteed to be False.
        best_dchisq[best_dchisq == 0] = np.nan

        # Now make the masks for which resultant difference(s) to
        # drop, count the number of ramps affected, and drop them.
        # If no ramps were affected break the loop.
        dropone = dchisq_one == best_dchisq
        droptwo = dchisq_two == best_dchisq

        drop = np.any([np.sum(dropone, axis=0), np.sum(droptwo, axis=0)], axis=0)

        if np.sum(drop) == 0:
            break

        # Store the updated counts with omitted reads
        new_cts = np.zeros(np.sum(recheck))
        i_d1 = np.sum(dropone, axis=0) > 0
        new_cts[i_d1] = np.sum(result.countrate_oneomit * dropone, axis=0)[i_d1]
        i_d2 = np.sum(droptwo, axis=0) > 0
        new_cts[i_d2] = np.sum(result.countrate_twoomit * droptwo, axis=0)[i_d2]

        # zero out count rates with drops and add their new values back in
        countrate[recheck] *= drop == 0
        countrate[recheck] += new_cts

        # Drop the read (set diffs2use=0) if the boolean array is True.
        diffs2use[:, recheck] *= ~dropone
        diffs2use[:-1, recheck] *= ~droptwo
        diffs2use[1:, recheck] *= ~droptwo

        # No need to repeat this on the entire ramp, only re-search
        # ramps that had a resultant difference dropped this time.
        dropped[:] = False
        dropped[recheck] = drop
        recheck[:] = dropped

        # Do not try to search for bad resultants if we have already
        # given up on all but one, two, or three resultant differences
        # in the ramp.  If there are only two left we have no way of
        # choosing which one is "good".  If there are three left we
        # run into trouble in case we need to discard two.
        recheck[np.sum(diffs2use, axis=0) <= 3] = False

    return diffs2use, countrate


def get_readtimes(ramp_data):
    """
    Get the read times needed to compute the covariance matrices.  If there is
    already a read_pattern in the ramp_data class, then just get it.  If not, then
    one needs to be constructed.  If one needs to be constructed it is assumed the
    groups are evenly spaced in time, as are the frames that make up the group.  If
    each group has only one frame and no group gap, then a list of the group times
    is returned.  If nframes > 0, then a list of lists of each frame time in each
    group is returned with the assumption:
        group_time = (nframes + groupgap) * frame_time

    Parameters
    ----------
    ramp_data : RampData
        Input data necessary for computing ramp fitting.

    Returns
    -------
    readtimes : list
        A list of frame times for each frame used in the computation of the ramp.
    """
    if ramp_data.read_pattern is not None:
        return ramp_data.read_pattern

    ngroups = ramp_data.data.shape[1]
    # rtimes = [(k + 1) * ramp_data.group_time for k in range(ngroups)]   # XXX Old
    tot_frames = ramp_data.nframes + ramp_data.groupgap
    tot_nreads = np.arange(1, ramp_data.nframes + 1)
    rtimes = [
        (tot_nreads + k * tot_frames) * ramp_data.frame_time for k in range(ngroups)
    ]

    return rtimes


def compute_image_info(integ_class, ramp_data):
    """
    Compute the diffs2use mask based on DQ flags of a row.

    Parameters
    ----------
    integ_class : IntegInfo
        Contains the rateints product calculations.

    ramp_data : RampData
        Input data necessary for computing ramp fitting.

    Returns
    -------
    image_info : tuple
        The list of arrays for the rate product.
    """
    if integ_class.data.shape[0] == 1:
        data = integ_class.data[0, :, :]
        dq = integ_class.dq[0, :, :]
        var_p = integ_class.var_poisson[0, :, :]
        var_r = integ_class.var_rnoise[0, :, :]
        var_e = integ_class.err[0, :, :]
        return (data, dq, var_p, var_r, var_e)

    dq = utils.dq_compress_final(integ_class.dq, ramp_data)

    # XXX Feedback from Brandt that this may not be correct.
    #     He provided another way to combine these computations.
    """
    # print("**** Old Computations ****")
    warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
    warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)
    inv_vp = 1.  / integ_class.var_poisson
    var_p = 1. / inv_vp.sum(axis=0)

    inv_vr = 1.  / integ_class.var_rnoise
    var_r = 1. / inv_vr.sum(axis=0)

    inv_err = 1.  / integ_class.err
    err = 1. / inv_err.sum(axis=0)

    inv_err2 = 1. / (integ_class.err**2)
    err2 = 1. / inv_err2.sum(axis=0)

    slope = integ_class.data * inv_err2
    slope = slope.sum(axis=0) * err2
    warnings.resetwarnings()
    """
    # print("**** New Computations ****")
    inv_err2 = 1.0 / (integ_class.err**2)
    weight = inv_err2 / inv_err2.sum(axis=0)
    weight2 = weight**2

    err2 = np.sum(integ_class.err**2 * weight2, axis=0)

    err = np.sqrt(err2)
    var_p = np.sum(integ_class.var_poisson * weight2, axis=0)
    var_r = np.sum(integ_class.var_rnoise * weight2, axis=0)
    slope = np.sum(integ_class.data * weight, axis=0)

    # XXX Compute NaNs.

    return (slope, dq, var_p, var_r, err)


def determine_diffs2use(ramp_data, integ, row, diffs):
    """
    Compute the diffs2use mask based on DQ flags of a row.

    Parameters
    ----------
    ramp_data : RampData
        Input data necessary for computing ramp fitting.

    integ : int
        The current integration being processed.

    row : int
        The current row being processed.

    diffs : ndarray
        The group differences of the data array for a given integration and row
        (ngroups-1, ncols).

    Returns
    -------
    d2use : ndarray
        A boolean array definined the segmented ramps for each pixel in a row.
        (ngroups-1, ncols)
    """
    _, ngroups, _, ncols = ramp_data.data.shape
    dq = np.zeros(shape=(ngroups, ncols), dtype=np.uint8)
    dq[:, :] = ramp_data.groupdq[integ, :, row, :]
    d2use_tmp = np.ones(shape=diffs.shape, dtype=np.uint8)
    d2use = d2use_tmp[:, row]

    # The JUMP_DET is handled different than other group DQ flags.
    jmp = np.uint8(ramp_data.flags_jump_det)
    other_flags = ~jmp

    # Find all non-jump flags
    oflags_locs = np.zeros(shape=dq.shape, dtype=np.uint8)
    wh_of = np.where(np.bitwise_and(dq, other_flags))
    oflags_locs[wh_of] = 1

    # Find all jump flags
    jmp_locs = np.zeros(shape=dq.shape, dtype=np.uint8)
    wh_j = np.where(np.bitwise_and(dq, jmp))
    jmp_locs[wh_j] = 1

    del wh_of, wh_j

    # Based on flagging, exclude differences associated with flagged groups.

    # If a jump occurs at group k, then the difference
    # group[k] - group[k-1] is excluded.
    d2use[jmp_locs[1:, :] == 1] = 0

    # If a non-jump flag occurs at group k, then the differences
    # group[k+1] - group[k] and group[k] - group[k-1] are excluded.
    d2use[oflags_locs[1:, :] == 1] = 0
    d2use[oflags_locs[:-1, :] == 1] = 0

    return d2use


def inital_countrateguess(covar, diffs, diffs2use):
    """
    Compute the initial count rate.

    Parameters
    ----------
    covar : Covar
        The class that computes and contains the covariance matrix info.

    diffs : ndarray
        The group differences of the data (ngroups-1, nrows, ncols).

    diffs2use : ndarray
        Boolean mask determining with group differences to use (ngroups-1, ncols).

    Returns
    -------
    countrateguess : ndarray
        The initial count rate.
    """
    # initial guess for count rate is the average of the unmasked
    # group differences unless otherwise specified.
    if covar.pedestal:
        num = np.sum((diffs * diffs2use)[1:], axis=0)
        den = np.sum(diffs2use[1:], axis=0)
    else:
        num = np.sum((diffs * diffs2use), axis=0)
        den = np.sum(diffs2use, axis=0)

    countrateguess = num / den
    countrateguess *= countrateguess > 0

    return countrateguess


# RAMP FITTING BEGIN
def fit_ramps(
    diffs,
    covar,
    gain,
    rnoise,  # Referred to as 'sig' in fitramp repo
    countrateguess=None,
    diffs2use=None,
    detect_jumps=False,
    resetval=0,
    resetsig=np.inf,
    rescale=True,
    dn_scale=10.0,
):
    """
    Function fit_ramps on a row of pixels.  Fits ramps to read differences
    using the covariance matrix for the read differences as given by the
    diagonal elements and the off-diagonal elements.

    Parameters
    ----------
    diffs : ndarray
        The group differences of the data (ngroups-1, nrows, ncols).

    covar : Covar
        The class that computes and contains the covariance matrix info.

    gain : ndarray
        The gain (ncols,)

    rnoise : ndarray
        The read noise (ncols,)

    countrateguess : ndarray
        Count rate estimates used to estimate the covariance matrix.
        Optional, default is None.

    diffs2use : ndarray
        Boolean mask determining with group differences to use (ngroups-1, ncols).
        Optional, default is None, which results in a mask of all 1's.

    detect_jumps : boolean
        Run jump detection.
        Optional, default is False.

    resetval : float or ndarray
        Priors on the reset values.  Irrelevant unless pedestal is True.  If an
        ndarray, it has dimensions (ncols).
        Opfional, default is 0.

    resetsig : float or ndarray
        Uncertainties on the reset values.  Irrelevant unless covar.pedestal is True.
        Optional, default np.inf, i.e., reset values have flat priors.

    rescale : boolean
        Scale the covariance matrix internally to avoid possible
        overflow/underflow problems for long ramps.
        Optional, default is True.

    dn_scale : XXX
        XXX

    Returns
    -------
    result : Ramp_Result
        Holds computed ramp fitting information.  XXX - rename
    """
    if diffs2use is None:
        # Use all diffs
        diffs2use = np.ones(diffs.shape, np.uint8)

    # diffs is (ngroups, ncols) of the current row
    if countrateguess is None:
        countrateguess = inital_countrateguess(covar, diffs, diffs2use)

    # XXX Maybe use a better name for this function, like compute_alphas_betas
    alpha_tuple, beta_tuple, scale = compute_alphas_betas(
        countrateguess, gain, rnoise, covar, rescale, diffs, dn_scale
    )
    alpha, alpha_phnoise, alpha_readnoise = alpha_tuple
    beta, beta_phnoise, beta_readnoise = beta_tuple

    ndiffs, npix = diffs.shape

    # Mask group differences that should be ignored.  This is half
    # of what we need to do to mask these group differences; the
    # rest comes later.
    diff_mask = diffs * diffs2use
    beta = beta * diffs2use[1:] * diffs2use[:-1]

    # All definitions and formulas here are in the paper.
    # --- Till line 237: Paper 1 section 4
    theta = compute_thetas(ndiffs, npix, alpha, beta)  # EQNs 38-40
    phi = compute_phis(ndiffs, npix, alpha, beta)  # EQNs 41-43

    sgn = np.ones((ndiffs, npix))
    sgn[::2] = -1

    Phi = compute_Phis(ndiffs, npix, beta, phi, sgn)  # EQN 46
    PhiD = compute_PhiDs(ndiffs, npix, beta, phi, sgn, diff_mask)  # EQN ??
    Theta = compute_Thetas(ndiffs, npix, beta, theta, sgn)  # EQN 47
    ThetaD = compute_ThetaDs(ndiffs, npix, beta, theta, sgn, diff_mask)  # EQN 48

    dB, dC, A, B, C = matrix_computations(
        ndiffs,
        npix,
        sgn,
        diff_mask,
        diffs2use,
        beta,
        phi,
        Phi,
        PhiD,
        theta,
        Theta,
        ThetaD,
    )

    result = get_ramp_result(
        dC,
        dB,
        A,
        B,
        C,
        scale,
        phi,
        theta,
        covar,
        resetval,
        resetsig,
        alpha_phnoise,
        alpha_readnoise,
        beta_phnoise,
        beta_readnoise,
    )

    # --- Beginning at line 250: Paper 1 section 4

    if detect_jumps:
        result = compute_jump_detects(
            result, ndiffs, diffs2use, dC, dB, A, B, C, scale, beta, phi, theta, covar
        )

    return result


# RAMP FITTING END


def compute_jump_detects(
    result, ndiffs, diffs2use, dC, dB, A, B, C, scale, beta, phi, theta, covar
):
    """
    The code below computes the best chi squared, best-fit slope,
    and its uncertainty leaving out each group difference in
    turn.  There are ndiffs possible differences that can be
    omitted.

    Then do it omitting two consecutive reads.  There are ndiffs-1
    possible pairs of adjacent reads that can be omitted.

    This approach would need to be modified if also fitting the
    pedestal, so that condition currently triggers an error.  The
    modifications would make the equations significantly more
    complicated; the matrix equations to be solved by hand would be
    larger.

    Paper II, sections 3.1 and 3.2

    Parameters
    ----------
    result : Ramp_Result
        The results of the ramp fitting for a given row of pixels in an integration.

    ndiffs :  int
        Number of differences.

    diffs2use : ndarray
        Boolean mask determining with group differences to use (ngroups-1, ncols).

    dC : ndarray
        Intermediate computation.

    dB : ndarray
        Intermediate computation.

    A : ndarray
        Intermediate computation.

    B : ndarray
        Intermediate computation.

    C : ndarray
        Intermediate computation.

    scale : ndarray or integer
        Overflow/underflow prevention scale.

    beta : ndarray
        Off diagonal of covariance matrix.

    phi : ndarray
        Intermediate computation.

    theta : ndarray
        Intermediate computation.

    covar : Covar
        The class that computes and contains the covariance matrix info.


    Returns
    -------
    result : Ramp_Result
        The results of the ramp fitting for a given row of pixels in an integration.
    """
    # The algorithms below do not work if we are computing the
    # pedestal here.
    if covar.pedestal:
        raise ValueError("Cannot use jump detection algorithm when fitting pedestals.")

    # XXX need to determine where DQ flagging of JUMP_DET needs to occur.

    # Diagonal elements of the inverse covariance matrix
    Cinv_diag = theta[:-1] * phi[1:] / theta[ndiffs]
    Cinv_diag *= diffs2use

    # Off-diagonal elements of the inverse covariance matrix
    # one spot above and below for the case of two adjacent
    # differences to be masked
    Cinv_offdiag = -beta * theta[:-2] * phi[2:] / theta[ndiffs]

    # Equations in the paper: best-fit a, b
    #
    # Catch warnings in case there are masked group
    # differences, since these will be overwritten later.  No need
    # to warn about division by zero here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = (Cinv_diag * B - dB * dC) / (C * Cinv_diag - dC**2)
        b = (dB - a * dC) / Cinv_diag

        result.countrate_oneomit = a
        result.jumpval_oneomit = b

        # Use the best-fit a, b to get chi squared
        result.chisq_oneomit = (
            A
            + a**2 * C
            - 2 * a * B
            + b**2 * Cinv_diag
            - 2 * b * dB
            + 2 * a * b * dC
        )

        # invert the covariance matrix of a, b to get the uncertainty on a
        result.uncert_oneomit = np.sqrt(Cinv_diag / (C * Cinv_diag - dC**2))
        result.jumpsig_oneomit = np.sqrt(C / (C * Cinv_diag - dC**2))

        result.chisq_oneomit /= scale
        result.uncert_oneomit *= np.sqrt(scale)
        result.jumpsig_oneomit *= np.sqrt(scale)

    # Now for two omissions in a row.  This is more work.  Again,
    # all equations are in the paper.  I first define three
    # factors that will be used more than once to save a bit of
    # computational effort.
    cpj_fac = dC[:-1] ** 2 - C * Cinv_diag[:-1]
    cjck_fac = dC[:-1] * dC[1:] - C * Cinv_offdiag
    bcpj_fac = B * dC[:-1] - dB[:-1] * C

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # best-fit a, b, c
        c = bcpj_fac / cpj_fac - (B * dC[1:] - dB[1:] * C) / cjck_fac
        c /= cjck_fac / cpj_fac - (dC[1:] ** 2 - C * Cinv_diag[1:]) / cjck_fac
        b = (bcpj_fac - c * cjck_fac) / cpj_fac
        a = (B - b * dC[:-1] - c * dC[1:]) / C
        result.countrate_twoomit = a

        # best-fit chi squared
        result.chisq_twoomit = (
            A + a**2 * C + b**2 * Cinv_diag[:-1] + c**2 * Cinv_diag[1:]
        )
        result.chisq_twoomit -= 2 * a * B + 2 * b * dB[:-1] + 2 * c * dB[1:]
        result.chisq_twoomit += (
            2 * a * b * dC[:-1] + 2 * a * c * dC[1:] + 2 * b * c * Cinv_offdiag
        )
        result.chisq_twoomit /= scale

        # uncertainty on the slope from inverting the (a, b, c)
        # covariance matrix
        fac = Cinv_diag[1:] * Cinv_diag[:-1] - Cinv_offdiag**2
        term2 = dC[:-1] * (dC[:-1] * Cinv_diag[1:] - Cinv_offdiag * dC[1:])
        term3 = dC[1:] * (dC[:-1] * Cinv_offdiag - Cinv_diag[:-1] * dC[1:])
        result.uncert_twoomit = np.sqrt(fac / (C * fac - term2 + term3))
        result.uncert_twoomit *= np.sqrt(scale)

    # XXX Maybe this tells where to mask.
    result.fill_masked_reads(diffs2use)

    return result


def compute_alphas_betas(countrateguess, gain, rnoise, covar, rescale, diffs, dn_scale):
    """
    Compute alpha, beta, and scale needed for ramp fit.
    Elements of the covariance matrix.
    Are these EQNs 32 and 33?

    Parameters
    ----------
    countrateguess : ndarray
        Initial guess (ncols,)

    gain : ndarray
        Gain (ncols,)

    rnoise : ndarray
        Read noise (ncols,)

    covar : Covar
        The class that computes and contains the covariance matrix info.

    rescale : bool
        Determination to rescale covariance matrix.

    diffs : ndarray
        The group differences of the data (ngroups-1, nrows, ncols).

    dn_scale : XXX
        XXX

    Returns
    -------
    alpha : ndarray
        Diagonal of covariance matrix.

    beta : ndarray
        Off diagonal of covariance matrix.

    scale : ndarray or integer
        Overflow/underflow prevention scale.

    """
    warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
    warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)

    # import ipdb; ipdb.set_trace()

    alpha_phnoise = countrateguess / gain * covar.alpha_phnoise[:, np.newaxis]
    alpha_readnoise = rnoise**2 * covar.alpha_readnoise[:, np.newaxis]
    alpha = alpha_phnoise + alpha_readnoise

    beta_phnoise = countrateguess / gain * covar.beta_phnoise[:, np.newaxis]
    beta_readnoise = rnoise**2 * covar.beta_readnoise[:, np.newaxis]
    beta = beta_phnoise + beta_readnoise

    warnings.resetwarnings()

    ndiffs, npix = diffs.shape

    # Rescale the covariance matrix to a determinant of 1 to
    # avoid possible overflow/underflow.  The uncertainty and chi
    # squared value will need to be scaled back later.  Note that
    # theta[-1] is the determinant of the covariance matrix.
    #
    # The method below uses the fact that if all alpha and beta
    # are multiplied by f, theta[i] is multiplied by f**i.  Keep
    # a running track of these factors to construct the scale at
    # the end, and keep scaling throughout so that we never risk
    # overflow or underflow.

    if rescale:
        # scale = np.exp(np.mean(np.log(alpha), axis=0))
        theta = np.ones((ndiffs + 1, npix))
        theta[1] = alpha[0]

        scale = theta[0] * 1
        warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
        warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)

        for i in range(2, ndiffs + 1):
            theta[i] = (
                alpha[i - 1] / scale * theta[i - 1]
                - beta[i - 2] ** 2 / scale**2 * theta[i - 2]
            )

            # Scaling every ten steps in safe for alpha up to 1e20
            # or so and incurs a negligible computational cost for
            # the fractional power.

            if i % int(dn_scale) == 0 or i == ndiffs:
                f = theta[i] ** (1 / i)
                scale *= f
                tmp = theta[i] / f
                theta[i - 1] /= tmp
                theta[i - 2] /= tmp / f
                theta[i] = 1

        warnings.resetwarnings()
    else:
        scale = 1

    alpha /= scale
    beta /= scale

    alpha_tuple = (alpha, alpha_phnoise, alpha_readnoise)
    beta_tuple = (beta, beta_phnoise, beta_readnoise)

    return alpha_tuple, beta_tuple, scale


def compute_thetas(ndiffs, npix, alpha, beta):
    """
    EQNs 38-40

    Parameters
    ----------
    ndiffs : int
        Number of differences.

    npix : int
        Number of columns in a row.

    alpha : ndarray
        Diagonal of covariance matrix.

    beta : ndarray
        Off diagonal of covariance matrix.

    Returns
    -------
    theta : ndarray
    """
    theta = np.ones((ndiffs + 1, npix))
    theta[1] = alpha[0]
    for i in range(2, ndiffs + 1):
        theta[i] = alpha[i - 1] * theta[i - 1] - beta[i - 2] ** 2 * theta[i - 2]
    return theta


def compute_phis(ndiffs, npix, alpha, beta):
    """
    EQNs 41-43

    Parameters
    ----------
    ndiffs : int
        Number of differences.

    npix : int
        Number of columns in a row.

    alpha : ndarray
        Diagonal of covariance matrix.

    beta : ndarray
        Off diagonal of covariance matrix.

    Returns
    -------
    phi : ndarray
    """
    phi = np.ones((ndiffs + 1, npix))
    phi[ndiffs - 1] = alpha[ndiffs - 1]
    for i in range(ndiffs - 2, -1, -1):
        phi[i] = alpha[i] * phi[i + 1] - beta[i] ** 2 * phi[i + 2]
    return phi


def compute_Phis(ndiffs, npix, beta, phi, sgn):
    """
    EQN 46

    Parameters
    ----------
    ndiffs : int
        Number of differences.

    npix : int
        Number of columns in a row.

    alpha : ndarray
        Diagonal of covariance matrix.

    beta : ndarray
        Off diagonal of covariance matrix.

    sgn : ndarray
        Oscillating 1, -1 sequence.

    Returns
    -------
    Phi : ndarray
    """
    Phi = np.zeros((ndiffs, npix))
    for i in range(ndiffs - 2, -1, -1):
        Phi[i] = Phi[i + 1] * beta[i] + sgn[i + 1] * beta[i] * phi[i + 2]
    return Phi


def compute_PhiDs(ndiffs, npix, beta, phi, sgn, diff_mask):
    """
    EQN 4, Paper II
    This one is defined later in the paper and is used for jump
    detection and pedestal fitting.

    Parameters
    ----------
    ndiffs : int
        Number of differences.

    npix : int
        Number of columns in a row.

    beta : ndarray
        Off diagonal of covariance matrix.

    phi : ndarray
        Intermediate computation.

    sgn : ndarray
        Oscillating 1, -1 sequence.

    diff_mask : ndarray
        Mask of differences used.

    Returns
    -------
    PhiD: ndarray
    """
    PhiD = np.zeros((ndiffs, npix))
    for i in range(ndiffs - 2, -1, -1):
        PhiD[i] = (PhiD[i + 1] + sgn[i + 1] * diff_mask[i + 1] * phi[i + 2]) * beta[i]
    return PhiD


def compute_Thetas(ndiffs, npix, beta, theta, sgn):
    """
    EQN 47

    Parameters
    ----------
    ndiffs : int
        Number of differences.

    npix : int
        Number of columns in a row.

    beta : ndarray
        Off diagonal of covariance matrix.

    theta : ndarray
        Intermediate computation.

    sgn : ndarray
        Oscillating 1, -1 sequence.

    Returns
    -------
    Theta : ndarray
    """
    Theta = np.zeros((ndiffs, npix))
    Theta[0] = -theta[0]
    for i in range(1, ndiffs):
        Theta[i] = Theta[i - 1] * beta[i - 1] + sgn[i] * theta[i]
    return Theta


def compute_ThetaDs(ndiffs, npix, beta, theta, sgn, diff_mask):
    """
    EQN 48

    Parameters
    ----------
    ndiffs : int
        Number of differences.

    npix : int
        Number of columns in a row.

    beta : ndarray
        Off diagonal of covariance matrix.

    theta : ndarray
        Intermediate computation.

    sgn : ndarray
        Oscillating 1, -1 sequence.

    diff_mask : ndarray
        Mask of differences used.

    Returns
    -------
    ThetaD : ndarray
    """
    ThetaD = np.zeros((ndiffs + 1, npix))
    ThetaD[1] = -diff_mask[0] * theta[0]
    for i in range(1, ndiffs):
        ThetaD[i + 1] = beta[i - 1] * ThetaD[i] + sgn[i] * diff_mask[i] * theta[i]
    return ThetaD


def matrix_computations(
    ndiffs, npix, sgn, diff_mask, diffs2use, beta, phi, Phi, PhiD, theta, Theta, ThetaD
):
    """
    Computing matrix computations needed for ramp fitting.
    EQNs 61-63, 71, 75

    Parameters
    ----------
    ndiffs : int
        Number of differences.

    npix : int
        Number of columns in a row.

    sgn : ndarray
        Oscillating 1, -1 sequence.

    diff_mask : ndarray
        Mask of differences used.

    diff2use : ndarray
        Masked differences.

    beta : ndarray
        Off diagonal of covariance matrix.

    phi : ndarray
        Intermediate computation.

    Phi : ndarray
        Intermediate computation.

    PhiD : ndarray
        Intermediate computation.

    theta : ndarray
        Intermediate computation.

    Theta : ndarray
        Intermediate computation.

    ThetaD : ndarray
        Intermediate computation.

    Returns
    -------
    dB : ndarray
        Intermediate computation.

    dC : ndarray
        Intermediate computation.

    A : ndarray
        Intermediate computation.

    B : ndarray
        Intermediate computation.

    C : ndarray
        Intermediate computation.
    """
    beta_extended = np.ones((ndiffs, npix))
    beta_extended[1:] = beta

    # C' and B' in the paper

    dC = sgn / theta[ndiffs] * (phi[1:] * Theta + theta[:-1] * Phi)
    dC *= diffs2use  # EQN 71

    dB = sgn / theta[ndiffs] * (phi[1:] * ThetaD[1:] + theta[:-1] * PhiD)  # EQN 75

    # {\cal A}, {\cal B}, {\cal C} in the paper

    # EQNs 61-63
    A = 2 * np.sum(
        diff_mask * sgn / theta[-1] * beta_extended * phi[1:] * ThetaD[:-1], axis=0
    )
    A += np.sum(diff_mask**2 * theta[:-1] * phi[1:] / theta[ndiffs], axis=0)

    B = np.sum(diff_mask * dC, axis=0)
    C = np.sum(dC, axis=0)

    return dB, dC, A, B, C


def get_ramp_result(
    dC,
    dB,
    A,
    B,
    C,
    scale,
    phi,
    theta,
    covar,
    resetval,
    resetsig,
    alpha_phnoise,
    alpha_readnoise,
    beta_phnoise,
    beta_readnoise,
):
    """
    Use intermediate computations to fit the ramp and save the results.

    Parameters
    ----------
    dB : ndarray
        Intermediate computation.

    dC : ndarray
        Intermediate computation.

    A : ndarray
        Intermediate computation.

    B : ndarray
        Intermediate computation.

    C : ndarray
        Intermediate computation.

    rescale : boolean
        Scale the covariance matrix internally to avoid possible
        overflow/underflow problems for long ramps.
        Optional, default is True.

    phi : ndarray
        Intermediate computation.

    theta : ndarray
        Intermediate computation.

    covar : Covar
        The class that computes and contains the covariance matrix info.

    resetval : float or ndarray
        Priors on the reset values.  Irrelevant unless pedestal is True.  If an
        ndarray, it has dimensions (ncols).
        Opfional, default is 0.

    resetsig : float or ndarray
        Uncertainties on the reset values.  Irrelevant unless covar.pedestal is True.
        Optional, default np.inf, i.e., reset values have flat priors.

    alpha_phnoise :
    alpha_readnoise :
    beta_phnoise :
    beta_readnoise :

    Returns
    -------
    result : Ramp_Result
        The results of the ramp fitting for a given row of pixels in an integration.
    """
    result = Ramp_Result()

    # Finally, save the best-fit count rate, chi squared, uncertainty
    # in the count rate, and the weights used to combine the
    # groups.

    # XXX pedestal is always False.
    if not covar.pedestal:
        warnings.filterwarnings("ignore", ".*invalid value.*", RuntimeWarning)
        warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)
        invC = 1 / C
        # result.countrate = B / C
        result.countrate = B * invC
        result.chisq = (A - B**2 / C) / scale

        result.uncert = np.sqrt(scale / C)
        result.weights = dC / C

        # XXX VAR
        # alpha_phnoise = countrateguess / gain * covar.alpha_phnoise[:, np.newaxis]
        result.var_poisson = np.sum(result.weights**2 * alpha_phnoise, axis=0)
        result.var_poisson += 2 * np.sum(
            result.weights[1:] * result.weights[:-1] * beta_phnoise, axis=0
        )

        result.var_rnoise = np.sum(result.weights**2 * alpha_readnoise, axis=0)
        result.var_rnoise += 2 * np.sum(
            result.weights[1:] * result.weights[:-1] * beta_readnoise, axis=0
        )

        warnings.resetwarnings()

    # If we are computing the pedestal, then we use the other formulas
    # in the paper.

    else:
        dt = covar.mean_t[0]
        Cinv_11 = theta[0] * phi[1] / theta[ndiffs]

        # Calculate the pedestal and slope using the equations in the paper.
        # Do not compute weights for this case.

        b = dB[0] * C * dt - B * dC[0] * dt + dt**2 * C * resetval / resetsig**2
        b /= C * Cinv_11 - dC[0] ** 2 + dt**2 * C / resetsig**2
        a = B / C - b * dC[0] / C / dt
        result.pedestal = b
        result.countrate = a
        result.chisq = A + a**2 * C + b**2 / dt**2 * Cinv_11
        result.chisq += -2 * b / dt * dB[0] - 2 * a * B + 2 * a * b / dt * dC[0]
        result.chisq /= scale

        # elements of the inverse covariance matrix
        M = [C, dC[0] / dt, Cinv_11 / dt**2 + 1 / resetsig**2]
        detM = M[0] * M[-1] - M[1] ** 2
        result.uncert = np.sqrt(scale * M[-1] / detM)
        result.uncert_pedestal = np.sqrt(scale * M[0] / detM)
        result.covar_countrate_pedestal = -scale * M[1] / detM

    return result


################################################################################
################################## DEBUG #######################################


def dbg_print_info(group_time, readtimes, data, diff):
    print(DELIM)
    print(f"group_time = {group_time}")
    print(DELIM)
    print(f"readtimes = {array_string(np.array(readtimes))}")
    print(DELIM)
    print(f"data = {array_string(data[:, 0, 0])}")
    print(DELIM)
    data_gt = data / group_time
    print(f"data / gt = {array_string(data_gt[:, 0, 0])}")
    print(DELIM)
    print(f"diff = {array_string(diff[:, 0, 0])}")
    print(DELIM)


def array_string(arr, prec=4):
    return np.array2string(arr, precision=prec, max_line_width=np.nan, separator=", ")
