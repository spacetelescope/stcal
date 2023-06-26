"""Ramp fitting routines.

The simulator need not actually fit any ramps, but we would like to do a good
job simulating the noise induced by ramp fitting.  That requires computing the
covariance matrix coming out of ramp fitting.  But that's actually a big part
of the work of ramp fitting.

There are a few different proposed ramp fitting algorithms, differing in their
weights.  The final derived covariances are all somewhat similarly difficult
to compute, however, since we ultimately end up needing to compute

.. math:: (A^T C^{-1} A)^{-1}

for the "optimal" case, or

.. math:: (A^T W^{-1} A)^{-1} A^T W^{-1} C W^{-1} A (A^T W^{-1} A)^{-1}

for some alternative weighting.

We start trying the "optimal" case below.

For the "optimal" case, a challenge is that we don't want to compute
:math:`C^{-1}` for every pixel individually.  Fortunately, we only
need :math:`(A^T C^{-1} A)^{-1}` (which is only a 2x2 matrix) for variances,
and only :math:`(A^T C^{-1} A)^{-1} A^T C^{-1}` for ramp fitting, which is 2xn.
Both of these matrices are effectively single parameter families, depending
after rescaling by the read noise only on the ratio of the read noise and flux.

So the routines in these packages construct these different matrices, store
them, and interpolate between them for different different fluxes and ratios.
"""
from astropy import units as u
import numpy as np
from scipy import interpolate

from . import ols_cas21
from .ols_cas21_util import ma_table_to_tau, ma_table_to_tbar, readpattern_to_matable


def construct_covar(read_noise, flux, ma_table):
    """Constructs covariance matrix for first finite differences of unevenly
    sampled resultants.

    Parameters
    ----------
    read_noise : float
        The read noise (electrons)
    flux : float
        The electrons per second
    ma_table : list[list]
        List of lists specifying the first read and the number of reads in each
        resultant.

    Returns
    -------
    np.ndarray[n_resultant, n_resultant] (float)
        covariance matrix of first finite differences of unevenly sampled
        resultants.
    """
    # read_time = parameters.read_time
    tau = ma_table_to_tau(ma_table)
    tbar = ma_table_to_tbar(ma_table)
    nreads = np.array([x[1] for x in ma_table])
    # from Casertano (2022), using Eqs 16, 19, and replacing with forward
    # differences.
    # diagonal -> (rn)^2/(1/N_i + 1/N_{i-1}) + f(tau_i + tau_{i-1} - 2t_{i-1}).
    # off diagonal: f(t_{i-1} - tau_{i-1}) - (rn)^2/N_{i-1}
    # further off diagonal: 0.
    diagonal = [[read_noise**2 / nreads[0] + flux * tau[0]],
                (read_noise**2 * (1 / nreads[1:] + 1 / nreads[:-1]) + flux * (
                    tau[1:] + tau[:-1] - 2 * tbar[:-1]))]
    cc = np.diag(np.concatenate(diagonal))

    off_diagonal = flux * (tbar[:-1] - tau[:-1]) - read_noise**2 / nreads[:-1]
    cc += np.diag(off_diagonal, 1)
    cc += np.diag(off_diagonal, -1)
    return cc.astype('f4')


def construct_ramp_fitting_matrices(covar, ma_table):
    """Construct :math:`A^T C^{-1} A` and :math:`A^T C^{-1}`, the matrices
    needed to fit ramps from resultants.

    The matrices constructed are those needed for applying to differences
    of resultants; e.g., the results of resultants_to_differences.

    Parameters
    ----------
    covar : np.ndarray[n_resultant, n_resultant] (float)
        covariance of differences of resultants
    ma_table : list[list] giving first read number and number of reads in each
        resultant

    Returns
    -------
    atcinva, atcinv : np.ndarray[2, 2], np.ndarray[2, n_resultant] (float)
        :math:`A^T C^{-1} A` and :math:`A^T C^{-1}`, so that
        pedestal, flux = np.linalg.inv(atcinva).dot(atcinva.dot(differences))
    """

    aa = np.zeros((len(ma_table), 2), dtype='f4')
    tbar = ma_table_to_tbar(ma_table)

    # pedestal; affects only 1st finite difference.
    aa[0, 0] = 1
    # slope; affects all finite differences
    aa[0, 1] = tbar[0]
    aa[1:, 1] = np.diff(tbar)
    cinv = np.linalg.inv(covar)
    # this won't be full rank if it's too small; we'll need some special
    # handling for nearly fully saturated cases, etc..
    atcinv = aa.T.dot(cinv)
    atcinva = atcinv.dot(aa)
    return atcinva, atcinv


def construct_ki_and_variances(atcinva, atcinv, covars):
    """Construct the :math:`k_i` weights and variances for ramp fitting.

    Following Casertano (2022), the ramp fit resultants are k.dot(differences),
    where :math:`k=(A^T C^{-1} A)^{-1} A^T C^{-1}`, and differences is the
    result of resultants_to_differences(resultants).  Meanwhile the variances
    are :math:`k C k^T`.  This function computes these k and variances.

    Parameters
    ----------
    atcinva : np.ndarray[2, 2] (float)
        :math:`A^T C^{-1} A` from construct_ramp_fitting_matrices
    atcinv : np.ndarray[2, n_resultant] (float)
        :math:`A^T C^{-1}` from construct_ramp_fitting_matrices
    covars : list[np.ndarray[n_resultant, n_resultant]]
        covariance matrices to contract against :math:`k` to compute variances

    Returns
    -------
    k : np.ndarray[2, n_resultant]
        :math:`k = (A^T C^{-1} A)^-1 A^T C^{-1}` from Casertano (2022)
    variances : list[np.ndarray[2, 2]] (float)
        :math:`k C_i k^T` for different covariance matrices C_i
        supplied in covars
    """

    k = np.linalg.inv(atcinva).dot(atcinv)
    variances = [k.dot(c).dot(k.T) for c in covars]
    return k, variances


def ki_and_variance_grid(ma_table, flux_on_readvar_pts):
    """Construct a grid of :math:`k` and covariances for the values of
    flux_on_readvar.

    The :math:`k` and corresponding covariances needed to do ramp fitting
    form essentially a one dimensional family in the flux in the ramp divided
    by the square of the read noise.  This function constructs these quantities
    for a large number of different flux / read_noise^2 to be used in
    interpolation.

    Parameters
    ----------
    ma_table : list[list] (int)
        a list of the first read and number of reads in each resultant
    flux_on_readvar_pts : array_like (float)
        values of flux / read_noise**2 for which :math:`k` and variances are
        desired.

    Returns
    -------
    kigrid : np.ndarray[len(flux_on_readvar_pts), 2, n_resultants] (float)
        :math:`k` for each value of flux_on_readvar_pts
    vargrid : np.ndarray[len(flux_on_readvar_pts), n_covar, 2, 2] (float)
        covariance of pedestal and slope corresponding to each value of
        flux_on_readvar_pts.  n_covar = 3, for the contributions from
        read_noise, Poisson noise, and the sum.
    """
    # the ramp fitting covariance matrices make a one-dimensional
    # family.  If we divide out the read variance, the single parameter
    # is flux / read_noise**2
    cc_rn = construct_covar(1, 0, ma_table)
    cc_flux = construct_covar(0, 1, ma_table)
    outki = []
    outvar = []
    for flux_on_readvar in flux_on_readvar_pts:
        cc_flux_scaled = cc_flux * flux_on_readvar
        atcinva, atcinv = construct_ramp_fitting_matrices(
            cc_rn + cc_flux_scaled, ma_table)
        covars = [cc_rn, cc_flux_scaled, cc_rn + cc_flux_scaled]
        ki, variances = construct_ki_and_variances(atcinva, atcinv, covars)
        outki.append(ki)
        outvar.append(variances)
    return np.array(outki), np.array(outvar)


def resultants_to_differences(resultants):
    """Convert resultants to their finite differences.

    This is essentially np.diff(...), but retains the first
    resultant.  The resulting structure has tri-diagonal covariance,
    which can be a little useful.

    Parameters
    ----------
    resultants : np.ndarray[n_resultant, nx, ny] (float)
        The resultants

    Returns
    -------
    differences : np.ndarray[n_resultant, nx, ny] (float)
        Differences of resultants
    """
    return np.vstack([resultants[0][None, :],
                      np.diff(resultants, axis=0)])


def fit_ramps_casertano(resultants, dq, read_noise, ma_table=None, read_pattern=None):
    """Fit ramps following Casertano+2022, including averaging partial ramps.

    Ramps are broken where dq != 0, and fits are performed on each sub-ramp.
    Resultants containing multiple ramps have their ramp fits averaged using
    inverse variance weights based on the variance in the individual slope fits
    due to read noise.

    Parameters
    ----------
    resultants : np.ndarry[nresultants, ...]
        the resultants in electrons
    dq : np.ndarry[nresultants, ...]
        the dq array.  dq != 0 implies bad pixel / CR.
    read noise: float
        the read noise in electrons
    ma_table : list[list[int]] or None
        The MA table prescription. If None, use `read_pattern`.
        One of `ma_table` or `read_pattern` must be defined.
    read_pattern : list[list[int]] or None
        The read pattern prescription. If None, use `ma_table`.
        One of `ma_table` or `read_pattern` must be defined.

    Returns
    -------
    par : np.ndarray[..., 2] (float)
        the best fit pedestal and slope for each pixel
    var : np.ndarray[..., 3, 2, 2] (float)
        the covariance matrix of par, for each of three noise terms:
        the read noise, Poisson source noise, and total noise.
    """

    # Get the Multi-accum table, either as given or from the read pattern
    if ma_table is None:
        if read_pattern is not None:
            ma_table = readpattern_to_matable(read_pattern)
    if ma_table is None:
        raise RuntimeError('One of `ma_table` or `read_pattern` must be given.')

    resultants_unit = getattr(resultants, 'unit', None)
    if resultants_unit is not None:
        resultants = resultants.to(u.electron).value

    resultants = np.array(resultants).astype('f4')

    dq = np.array(dq).astype('i4')

    if np.ndim(read_noise) <= 1:
        read_noise = read_noise * np.ones(resultants.shape[1:])
    read_noise = np.array(read_noise).astype('f4')

    origshape = resultants.shape
    if len(resultants.shape) == 1:
        # single ramp.
        resultants = resultants.reshape(origshape + (1,))
        dq = dq.reshape(origshape + (1,))
        read_noise = read_noise.reshape(origshape[1:] + (1,))

    rampfitdict = ols_cas21.fit_ramps(
        resultants.reshape(resultants.shape[0], -1),
        dq.reshape(resultants.shape[0], -1),
        read_noise.reshape(-1),
        ma_table)

    par = np.zeros(resultants.shape[1:] + (2,), dtype='f4')
    var = np.zeros(resultants.shape[1:] + (3, 2, 2), dtype='f4')

    npix = resultants.reshape(resultants.shape[0], -1).shape[1]
    # we need to do some averaging to merge the results in each ramp.
    # inverse variance weights based on slopereadvar
    weight = ((rampfitdict['slopereadvar'] != 0) / (
        rampfitdict['slopereadvar'] + (rampfitdict['slopereadvar'] == 0)))
    totweight = np.bincount(rampfitdict['pix'], weights=weight, minlength=npix)
    totval = np.bincount(rampfitdict['pix'],
                         weights=weight * rampfitdict['slope'],
                         minlength=npix)
    # fill in the averaged slopes
    par.reshape(npix, 2)[:, 1] = (
        totval / (totweight + (totweight == 0)))

    # read noise variances
    totval = np.bincount(
        rampfitdict['pix'], weights=weight ** 2 * rampfitdict['slopereadvar'],
        minlength=npix)
    var.reshape(npix, 3, 2, 2)[:, 0, 1, 1] = (
        totval / (totweight ** 2 + (totweight == 0)))
    # poisson noise variances
    totval = np.bincount(
        rampfitdict['pix'],
        weights=weight ** 2 * rampfitdict['slopepoissonvar'], minlength=npix)
    var.reshape(npix, 3, 2, 2)[..., 1, 1, 1] = (
        totval / (totweight ** 2 + (totweight == 0)))

    var[..., 1, 1, 1] *= par[..., 1]  # multiply Poisson term by flux
    var[..., 2, 1, 1] = var[..., 0, 1, 1] + var[..., 1, 1, 1]

    if resultants.shape != origshape:
        par = par[0]
        var = var[0]

    if resultants_unit is not None:
        par = par * resultants_unit

    return par, var


def fit_ramps_casertano_no_dq(resultants, read_noise, ma_table):
    """Fit ramps following Casertano+2022, only using full ramps.

    This is a simpler implementation of fit_ramps_casertano, which doesn't
    address the case of partial ramps broken by CRs.  This case is easier
    and can be done reasonably efficiently in pure python; results can be
    compared with fit_ramps_casertano in for the case of unbroken ramps.

    Parameters
    ----------
    resultants : np.ndarry[nresultants, npixel]
        the resultants in electrons
    read noise: float
        the read noise in electrons
    ma_table : list[list[int]]
        the ma table prescription

    Returns
    -------
    par : np.ndarray[nx, ny, 2] (float)
        the best fit pedestal and slope for each pixel
    var : np.ndarray[nx, ny, 3, 2, 2] (float)
        the covariance matrix of par, for each of three noise terms:
        the read noise, Poisson source noise, and total noise.
    """
    nadd = len(resultants.shape) - 1
    if np.ndim(read_noise) <= 1:
        read_noise = np.array(read_noise).reshape((1,) * nadd)
    smax = resultants[-1]
    s = smax / np.sqrt(read_noise**2 + smax)  # Casertano+2022 Eq. 44
    ptable = np.array([  # Casertano+2022, Table 2
        [-np.inf, 0], [5, 0.4], [10, 1], [20, 3], [50, 6], [100, 10]])
    pp = ptable[np.searchsorted(ptable[:, 0], s) - 1, 1]
    nn = np.array([x[1] for x in ma_table])  # number of reads in each resultant
    tbar = ma_table_to_tbar(ma_table)
    tau = ma_table_to_tau(ma_table)
    tbarmid = (tbar[0] + tbar[-1]) / 2
    if nadd > 0:
        newshape = ((-1,) + (1,) * nadd)
        nn = nn.reshape(*newshape)
        tbar = tbar.reshape(*newshape)
        tau = tau.reshape(*newshape)
        tbarmid = tbarmid.reshape(*newshape)
    ww = (  # Casertano+22, Eq. 45
        (1 + pp)[None, ...] * nn
        / (1 + pp[None, ...] * nn)
        * np.abs(tbar - tbarmid) ** pp[None, ...])

    # Casertano+22 Eq. 35
    f0 = np.sum(ww, axis=0)
    f1 = np.sum(ww * tbar, axis=0)
    f2 = np.sum(ww * tbar**2, axis=0)
    # Casertano+22 Eq. 36
    dd = f2 * f0 - f1 ** 2
    bad = dd == 0
    dd[bad] = 1
    # Casertano+22 Eq. 37
    kk = (f0[None, ...] * tbar - f1[None, ...]) * ww / (
        dd[None, ...])
    # shape: [n_resultant, ny, nx]
    ff = np.sum(kk * resultants, axis=0)  # Casertano+22 Eq. 38
    # Casertano+22 Eq. 39
    vr = np.sum(kk**2 / nn, axis=0) * read_noise**2
    # Casertano+22 Eq. 40
    vs1 = np.sum(kk**2 * tau, axis=0)
    vs2inner = np.cumsum(kk * tbar, axis=0)
    vs2inner = np.concatenate([0 * vs2inner[0][None, ...], vs2inner[:-1, ...]], axis=0)
    vs2 = 2 * np.sum(vs2inner * kk, axis=0)
    # sum_{i=1}^{j-1} K_i \bar{t}_i
    # this is the inner of the two sums in the 2nd term of Eq. 40
    # Casertano+22 has some discussion of whether it's more efficient to do
    # this as an explicit double sum or to construct the inner sum separately.
    # We've made a lot of other quantities that are [nr, ny, nx] in size,
    # so I don't feel bad about making another.  Clearly a memory optimized
    # code would work a lot harder to reuse a lot of variables above!

    vs = (vs1 + vs2) * ff
    vs = np.clip(vs, 0, np.inf)
    # we can estimate negative flux, but we really shouldn't add variance for
    # that case!

    # match return values from RampFitInterpolator.fit_ramps
    # we haven't explicitly calculated here the pedestal, its
    # uncertainty, or covariance terms.  We just fill
    # with zeros.

    par = np.zeros(ff.shape + (2,), dtype='f4')
    var = np.zeros(ff.shape + (3, 2, 2), dtype='f4')
    par[..., 1] = ff
    var[..., 0, 1, 1] = vr
    var[..., 1, 1, 1] = vs
    var[..., 2, 1, 1] = vr + vs

    return par, var
