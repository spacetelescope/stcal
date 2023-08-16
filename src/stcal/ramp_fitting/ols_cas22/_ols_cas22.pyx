import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
cimport cython

from stcal.ramp_fitting.ols_cas22_util import ma_table_to_tau, ma_table_to_tbar

from stcal.ramp_fitting.ols_cas22._core cimport make_ramp


cdef inline (vector[int], vector[int], vector[int]) end_points(int n_ramp,
                                                               int n_pixel,
                                                               int n_resultants,
                                                               int[:, :] dq):

    cdef vector[int] start = vector[int](n_ramp, -1)
    cdef vector[int] end = vector[int](n_ramp, -1)
    cdef vector[int] pix = vector[int](n_ramp, -1)

    cdef int i, j
    cdef int in_ramp = -1
    cdef int ramp_num = 0
    for i in range(n_pixel):
        in_ramp = 0
        for j in range(n_resultants):
            if (not in_ramp) and (dq[j, i] == 0):
                in_ramp = 1
                pix[ramp_num] = i
                start[ramp_num] = j
            elif (not in_ramp) and (dq[j, i] != 0):
                continue
            elif in_ramp and (dq[j, i] == 0):
                continue
            elif in_ramp and (dq[j, i] != 0):
                in_ramp = 0
                end[ramp_num] = j - 1
                ramp_num += 1
            else:
                raise ValueError('unhandled case')
        if in_ramp:
            end[ramp_num] = j
            ramp_num += 1

    return start, end, pix


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
    cdef int n_resultants = len(ma_table)
    if n_resultants != resultants.shape[0]:
        raise RuntimeError(f'MA table length {n_resultants} does not '
                           f'match number of resultants {resultants.shape[0]}')

    cdef np.ndarray[int] n_reads = np.array([x[1] for x in ma_table]).astype('i4')
    # number of reads in each resultant

    cdef np.ndarray[float] t_bar = ma_table_to_tbar(ma_table, read_time).astype('f4')
    cdef np.ndarray[float] tau = ma_table_to_tau(ma_table, read_time).astype('f4')
    cdef int n_pixel = resultants.shape[1]
    cdef int n_ramp = (np.sum(dq[0, :] == 0) +
                       np.sum((dq[:-1, :] != 0) & (dq[1:, :] == 0)))

    # numpy arrays so that we get numpy arrays out
    cdef np.ndarray[float] slope = np.zeros(n_ramp, dtype=np.float32)
    cdef np.ndarray[float] slope_read_var = np.zeros(n_ramp, dtype=np.float32)
    cdef np.ndarray[float] slope_poisson_var = np.zeros(n_ramp, dtype=np.float32)

    cdef vector[int] start, end, pix
    start, end, pix = end_points(n_ramp, n_pixel, n_resultants, dq)

    for i in range(n_ramp):
        slope[i], slope_read_var[i], slope_poisson_var[i] = make_ramp(
            resultants[:, pix[i]],
            start[i], end[i],
            read_noise[pix[i]], t_bar, tau, n_reads).fit()

    return dict(slope=slope, slopereadvar=slope_read_var,
                slopepoissonvar=slope_poisson_var,
                pix=pix, resstart=start, resend=end)
