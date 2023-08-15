import numpy as np
cimport numpy as np
cimport cython

from stcal.ramp_fitting.ols_cas22_util import ma_table_to_tau, ma_table_to_tbar

from stcal.ramp_fitting.ols_cas22._core cimport make_ramp


@cython.boundscheck(False)
@cython.wraparound(False)
def fit_ramps(np.ndarray[float, ndim=2] resultants,
              np.ndarray[int, ndim=2] dq,
              np.ndarray[float, ndim=1] read_noise, read_time,
              ma_table):
    """Fit ramps using the Casertano+22 algorithm.

    This implementation fits all ramp segments between bad pixels
    marked in the dq image with values not equal to zero.  So the
    number of fit ramps can be larger than the number of pixels.
    The derived slopes, corresponding variances, and the locations of
    the ramps in each pixel are given in the returned dictionary.

    Parameters
    ----------
    resultants : np.ndarry[nresultants, npixel]
        the resultants in electrons
    dq : np.ndarry[nresultants, npixel]
        the dq array.  dq != 0 implies bad pixel / CR.
    read noise : float
        the read noise in electrons
    read_time : float
        Time to perform a readout. For Roman data, this is FRAME_TIME.
    ma_table : list[list[int]]
        the ma table prescription

    Returns
    -------
    dictionary containing the following keywords:
    slope : np.ndarray[nramp]
        slopes fit for each ramp
    slopereadvar : np.ndarray[nramp]
        variance in slope due to read noise
    slopepoissonvar : np.ndarray[nramp]
        variance in slope due to Poisson noise, divided by the slope
        i.e., the slope poisson variance is coefficient * flux; this term
        is the coefficient.
    pix : np.ndarray[nramp]
        the pixel each ramp is in
    resstart : np.ndarray[nramp]
        The first resultant in this ramp
    resend : np.ndarray[nramp]
        The last resultant in this ramp.
    """
    cdef int nresultant = len(ma_table)
    if nresultant != resultants.shape[0]:
        raise RuntimeError(f'MA table length {nresultant} does not '
                           f'match number of resultants {resultants.shape[0]}')

    cdef np.ndarray[int] nn = np.array([x[1] for x in ma_table]).astype('i4')
    # number of reads in each resultant
    cdef np.ndarray[float] tbar = ma_table_to_tbar(ma_table, read_time).astype('f4')
    cdef np.ndarray[float] tau = ma_table_to_tau(ma_table, read_time).astype('f4')
    cdef int npixel = resultants.shape[1]
    cdef int nramp = (np.sum(dq[0, :] == 0) +
                      np.sum((dq[:-1, :] != 0) & (dq[1:, :] == 0)))
    cdef np.ndarray[float] slope = np.zeros(nramp, dtype='f4')
    cdef np.ndarray[float] slopereadvar = np.zeros(nramp, dtype='f4')
    cdef np.ndarray[float] slopepoissonvar = np.zeros(nramp, dtype='f4')
    cdef np.ndarray[int] resstart = np.zeros(nramp, dtype='i4') - 1
    cdef np.ndarray[int] resend = np.zeros(nramp, dtype='i4') - 1
    cdef np.ndarray[int] pix = np.zeros(nramp, dtype='i4') - 1
    cdef int i, j
    cdef int inramp = -1
    cdef int rampnum = 0
    for i in range(npixel):
        inramp = 0
        for j in range(nresultant):
            if (not inramp) and (dq[j, i] == 0):
                inramp = 1
                pix[rampnum] = i
                resstart[rampnum] = j
            elif (not inramp) and (dq[j, i] != 0):
                continue
            elif inramp and (dq[j, i] == 0):
                continue
            elif inramp and (dq[j, i] != 0):
                inramp = 0
                resend[rampnum] = j - 1
                rampnum += 1
            else:
                raise ValueError('unhandled case')
        if inramp:
            resend[rampnum] = j
            rampnum += 1
    # we should have just filled out the starting and stopping locations
    # of each ramp.

    for i in range(nramp):
        slope[i], slopereadvar[i], slopepoissonvar[i] = make_ramp(
            resultants[:, pix[i]],
            resstart[i], resend[i],
            read_noise[pix[i]], tbar, tau, nn).fit()

    return dict(slope=slope, slopereadvar=slopereadvar,
                slopepoissonvar=slopepoissonvar,
                pix=pix, resstart=resstart, resend=resend)
