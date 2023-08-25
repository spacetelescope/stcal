import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.stack cimport stack
cimport cython

from stcal.ramp_fitting.ols_cas22._core cimport (
    Fits, RampIndex, make_threshold, read_data, init_ramps)
from stcal.ramp_fitting.ols_cas22._fixed cimport make_fixed, Fixed
from stcal.ramp_fitting.ols_cas22._ramp cimport make_ramp


@cython.boundscheck(False)
@cython.wraparound(False)
def fit_ramps(np.ndarray[float, ndim=2] resultants,
              np.ndarray[int, ndim=2] dq,
              np.ndarray[float, ndim=1] read_noise, read_time,
              ma_table,
              int use_jumps=False):
    """Fit ramps using the Casertano+22 algorithm.

    This implementation fits all ramp segments between bad pixels
    marked in the dq image with values not equal to zero.  So the
    number of fit ramps can be larger than the number of pixels.
    The derived slopes, corresponding variances, and the locations of
    the ramps in each pixel are given in the returned dictionary.

    Parameters
    ----------
    resultants : np.ndarry[n_resultants, n_pixel]
        the resultants in electrons
    dq : np.ndarry[n_resultants, n_pixel]
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

    # Pre-compute data for all pixels
    cdef Fixed fixed = make_fixed(read_data(ma_table, read_time),
                                  make_threshold(5.5, 1/3.0),
                                  use_jumps)

    # Compute all the initial sets of ramps
    cdef vector[stack[RampIndex]] pixel_ramps = init_ramps(dq)

    # Set up the output lists
    #    Thes are python lists because cython does not support templating
    #    types baised on Python types like what numpy arrays are.
    #    This is an annoying limitation.
    slopes = []
    read_vars = []
    poisson_vars = []

    # Perform all of the fits
    cdef Fits fits
    cdef int index
    for index in range(n_resultants):
        # Fit all the ramps for the given pixel
        fits = make_ramp(fixed, read_noise,
                         resultants[:, index]).fits(pixel_ramps[index])

        # Cast into numpy arrays for output
        slopes.append(np.array(<float [:fits.slope.size()]> fits.slope.data()))
        read_vars.append(np.array(<float [:fits.read_var.size()]> fits.read_var.data()))
        poisson_vars.append(np.array(<float [:fits.poisson_var.size()]>
                                     fits.poisson_var.data()))

    return dict(slope=slopes, slopereadvar=read_vars,
                slopepoissonvar=poisson_vars)
