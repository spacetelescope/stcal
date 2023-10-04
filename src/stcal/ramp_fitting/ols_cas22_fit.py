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

from . import ols_cas22
from .ols_cas22_util import ma_table_to_tau, ma_table_to_tbar, read_pattern_to_ma_table, ma_table_to_read_pattern


def fit_ramps_casertano(resultants, dq, read_noise, read_time, ma_table=None, read_pattern=None, use_jump=False):
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
    read_noise : float
        the read noise in electrons
    read_time : float
        Read time. For Roman data this is the FRAME_TIME keyword.
    ma_table : list[list[int]] or None
        The MA table prescription. If None, use `read_pattern`.
        One of `ma_table` or `read_pattern` must be defined.
    read_pattern : list[list[int]] or None
        The read pattern prescription. If None, use `ma_table`.
        One of `ma_table` or `read_pattern` must be defined.
    use_jump : bool
        If True, use the jump detection algorithm to identify CRs.
        If False, use the DQ array to identify CRs.

    Returns
    -------
    par : np.ndarray[..., 2] (float)
        the best fit pedestal and slope for each pixel
    var : np.ndarray[..., 3, 2, 2] (float)
        the covariance matrix of par, for each of three noise terms:
        the read noise, Poisson source noise, and total noise.
    """

    # Get the Multi-accum table, either as given or from the read pattern
    if read_pattern is None:
        if ma_table is not None:
            read_pattern = ma_table_to_read_pattern(ma_table)
    if read_pattern is None:
        raise RuntimeError('One of `ma_table` or `read_pattern` must be given.')

    resultants_unit = getattr(resultants, 'unit', None)
    if resultants_unit is not None:
        resultants = resultants.to(u.electron).value

    resultants = np.array(resultants).astype(np.float32)

    dq = np.array(dq).astype(np.int32)
    if np.ndim(read_noise) <= 1:
        read_noise = read_noise * np.ones(resultants.shape[1:])
    read_noise = np.array(read_noise).astype(np.float32)

    orig_shape = resultants.shape
    if len(resultants.shape) == 1:
        # single ramp.
        resultants = resultants.reshape(origshape + (1,))
        dq = dq.reshape(origshape + (1,))
        read_noise = read_noise.reshape(origshape[1:] + (1,))

    ramp_fits, parameters, variances = ols_cas22.fit_ramps(
        resultants.reshape(resultants.shape[0], -1),
        dq.reshape(resultants.shape[0], -1),
        read_noise.reshape(-1),
        read_time,
        read_pattern,
        use_jump)

    if resultants.shape != orig_shape:
        parameters = parameters[0]
        variances = variances[0]

    if resultants_unit is not None:
        parameters = parameters * resultants_unit

    return parameters, variances


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
