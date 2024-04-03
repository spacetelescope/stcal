#! /usr/bin/env python

import logging
import multiprocessing
import time
import warnings
from multiprocessing import cpu_count

import numpy as np

from . import ramp_fit_class, utils
from .likely_algo_classes import IntegInfo, ImageInfo, Ramp_Result, Covar

################## DEBUG ##################
#                  HELP!!
import ipdb
import sys

sys.path.insert(1, "/Users/kmacdonald/code/common")
from general_funcs import DELIM, dbg_print, array_string

################## DEBUG ##################

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

BUFSIZE = 1024 * 300000  # 300Mb cache size for data section

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def likely_ramp_fit(ramp_data, buffsize, save_opt, readnoise_2d, gain_2d, weighting, max_cores):
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

    # XXX Maybe make this more general.  Since groups are evenly space
    #     in JWST, the readtimes are simply uniformly spaced times based
    #     on group time.  With the knowledge of frame time and the number
    #     of frames, this may be able to be more general.  Ask someone
    #     about this.
    readtimes = [(k+1) * ramp_data.group_time for k in range(ngroups)]
    covar = Covar(readtimes)
    integ_class = IntegInfo(nints, nrows, ncols)

    for integ in range(nints):
        data = ramp_data.data[integ, :, :, :]
        diff = (data[1:] - data[:-1]) / covar.delta_t[:, np.newaxis, np.newaxis]

        for row in range(nrows):
            result = fit_ramps(diff[:, row], covar, readnoise_2d[row])
            integ_class.get_results(result, integ, row)

    integ_info = integ_class.prepare_info()

    return image_info, integ_info, opt_info


def inital_countrateguess(Cov, diffs, diffs2use):
    """
    Compute the initial count rate.

    Parameters
    ----------
    Cov : Covar
        The class that computes and contains the covariance matrix info.

    diffs : ndarray
        The group differences of the data (ngroups-1, nrows, ncols).

    diffs2use : ndarray
        Boolean mask determining with group differences to use (ngroups-1, ncols).
    """
    # initial guess for count rate is the average of the unmasked
    # group differences unless otherwise specified.
    if Cov.pedestal:
        num = np.sum((diffs * diffs2use)[1:], axis=0)
        den = np.sum(diffs2use[1:], axis=0)
        countrateguess = num / den
        '''
        countrateguess = np.sum((diffs * diffs2use)[1:], axis=0) / np.sum(
            diffs2use[1:], axis=0
        )
        '''
    else:
        num = np.sum((diffs * diffs2use), axis=0)
        den = np.sum(diffs2use, axis=0)
        countrateguess = num / den
        '''
        countrateguess = np.sum((diffs * diffs2use), axis=0) / np.sum(
            diffs2use, axis=0
        )
        '''
    countrateguess *= countrateguess > 0

    return countrateguess 


# RAMP FITTING BEGIN
def fit_ramps(
    diffs,
    covar,
    rnoise,
    countrateguess=None,
    diffs2use=None,
    detect_jumps=False,
    resetval=0,
    resetsig=np.inf,
    rescale=True,
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

    rnoise : ndarray
        The read noise (ncols,).  XXX - the name should be changed.

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

    Returns
    -------
    result : Ramp_Result
        Holds computed ramp fitting information.  XXX - rename
    """
    # XXX
    # this needs to be refactored into a well written function,
    # instead of meandering crap, hundreds of lines long.  The
    # next function is at line 544.

    if diffs2use is None:
        # Use all diffs
        diffs2use = np.ones(diffs.shape, np.uint8)

    # diffs is (ngroups, ncols) of the current row
    if countrateguess is None:
        countrateguess = inital_countrateguess(covar, diffs, diffs2use)

    alpha, beta, scale = compute_abs(countrateguess, rnoise, covar, rescale)
    ndiffs, npix = alpha.shape

    # Mask group differences that should be ignored.  This is half
    # of what we need to do to mask these group differences; the
    # rest comes later.
    diff_mask = diffs * diffs2use
    beta = beta * diffs2use[1:] * diffs2use[:-1]

    # All definitions and formulas here are in the paper.
    # --- Till line 284: Paper 1 section 4
    theta = compute_thetas(ndiffs, npix, alpha, beta)  # EQNs 38-40
    phi = compute_phis(ndiffs, npix, alpha, beta)  # EQNs 41-43

    sgn = np.ones((ndiffs, npix))
    sgn[::2] = -1

    Phi = compute_Phis(ndiffs, npix, beta, phi, sgn)  # EQN 46
    PhiD = compute_PhiDs(ndiffs, npix, beta, phi, sgn, diff_mask)  # EQN ??
    Theta = compute_Thetas(ndiffs, npix, beta, theta, sgn)  # EQN 47
    ThetaD = compute_ThetaDs(ndiffs, npix, beta, theta, sgn, diff_mask)  # EQN 48

    dB, dC, A, B, C = matrix_computations(
        ndiffs, npix, sgn, diff_mask, diffs2use, beta, phi, Phi, PhiD, theta, Theta, ThetaD)

    result = get_ramp_result(dC, dB, A, B, C, scale, phi, theta, covar, resetval, resetsig)
    # --- Beginning at line 250: Paper 1 section 4

    # The code below computes the best chi squared, best-fit slope,
    # and its uncertainty leaving out each group difference in
    # turn.  There are ndiffs possible differences that can be
    # omitted.
    #
    # Then do it omitting two consecutive reads.  There are ndiffs-1
    # possible pairs of adjacent reads that can be omitted.
    #
    # This approach would need to be modified if also fitting the
    # pedestal, so that condition currently triggers an error.  The
    # modifications would make the equations significantly more
    # complicated; the matrix equations to be solved by hand would be
    # larger.

    # XXX - This needs to get moved into a separate function.  This section should
    #       be separated anyway, since it's a completely separate function, but the
    #       code itself should be further broken down, as it's a meandering mess
    #       also.  Far too complicated for a single function.
    # Paper II, sections 3.1 and 3.2
    if detect_jumps:

        # The algorithms below do not work if we are computing the
        # pedestal here.

        if covar.pedestal:
            raise ValueError(
                "Cannot use jump detection algorithm when fitting pedestals."
            )

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

        result.fill_masked_reads(diffs2use)

    return result
# RAMP FITTING END

def compute_abs(countrateguess, rnoise, covar, rescale):
    """
    Compute alpha, beta, and scale needed for ramp fit.
    Elements of the covariance matrix.
    Are these EQNs 32 and 33?

    Parameters
    ----------
    countrateguess : ndarray
        Initial guess (ncols,)

    rnoise : ndarray
        Readnoise (ncols,)

    covar : Covar
        The class that computes and contains the covariance matrix info.

    rescale : bool
        Determination to rescale covariance matrix.

    Returns
    -------
    alpha : ndarray
        Diagonal of covariance matrix.

    beta : ndarray
        Off diagonal of covariance matrix.

    scale : ndarray or integer
        Overflow/underflow prevention scale.

    """
    alpha = countrateguess * covar.alpha_phnoise[:, np.newaxis]
    alpha += rnoise**2 * covar.alpha_readnoise[:, np.newaxis]
    beta = countrateguess * covar.beta_phnoise[:, np.newaxis]
    beta += rnoise**2 * covar.beta_readnoise[:, np.newaxis]

    # rescale the covariance matrix to a determinant of order 1 to
    # avoid possible overflow/underflow.  The uncertainty and chi
    # squared value will need to be scaled back later.
    if rescale:
        scale = np.exp(np.mean(np.log(alpha), axis=0))
    else:
        scale = 1

    alpha /= scale
    beta /= scale

    return alpha, beta, scale

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
        ndiffs, npix, sgn, diff_mask, diffs2use, beta, phi, Phi, PhiD, theta, Theta, ThetaD):
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

    dC : ndarray

    A : ndarray

    B : ndarray

    C : ndarray

    """
    beta_extended = np.ones((ndiffs, npix))
    beta_extended[1:] = beta

    # C' and B' in the paper

    dC = sgn / theta[ndiffs] * (phi[1:] * Theta + theta[:-1] * Phi)
    dC *= diffs2use  # EQN 71

    dB = sgn / theta[ndiffs] * (phi[1:] * ThetaD[1:] + theta[:-1] * PhiD)  # EQN 75

    # {\cal A}, {\cal B}, {\cal C} in the paper

    # EQNs 61-63
    A = 2 * np.sum(diff_mask * sgn / theta[-1] * beta_extended * phi[1:] * ThetaD[:-1], axis=0)
    A += np.sum(diff_mask**2 * theta[:-1] * phi[1:] / theta[ndiffs], axis=0)

    B = np.sum(diff_mask * dC, axis=0)
    C = np.sum(dC, axis=0)

    return dB, dC, A, B, C


def get_ramp_result(dC, dB, A, B, C, scale, phi, theta, Cov, resetval, resetsig):
    result = Ramp_Result()

    # Finally, save the best-fit count rate, chi squared, uncertainty
    # in the count rate, and the weights used to combine the
    # groups.

    if not Cov.pedestal:
        result.countrate = B / C
        result.chisq = (A - B**2 / C) / scale
        result.uncert = np.sqrt(scale / C)
        result.weights = dC / C

    # If we are computing the pedestal, then we use the other formulas
    # in the paper.

    else:
        dt = Cov.mean_t[0]
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
