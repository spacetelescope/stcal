import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.stack cimport stack
from libcpp.list cimport list as cpp_list
from libcpp.deque cimport deque
cimport cython

from stcal.ramp_fitting.ols_cas22._core cimport (
    RampFits, RampIndex, make_threshold, read_data, init_ramps)
from stcal.ramp_fitting.ols_cas22._fixed cimport make_fixed, Fixed
from stcal.ramp_fitting.ols_cas22._pixel cimport make_pixel


@cython.boundscheck(False)
@cython.wraparound(False)
def fit_ramps(np.ndarray[float, ndim=2] resultants,
              np.ndarray[int, ndim=2] dq,
              np.ndarray[float, ndim=1] read_noise, read_time,
              list[list[int]] read_pattern,
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
    read_pattern : list[list[int]]
        the read pattern for the image

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
    cdef int n_resultants = len(read_pattern)
    if n_resultants != resultants.shape[0]:
        raise RuntimeError(f'MA table length {n_resultants} does not '
                           f'match number of resultants {resultants.shape[0]}')

    # Pre-compute data for all pixels
    cdef Fixed fixed = make_fixed(read_data(read_pattern, read_time),
                                  make_threshold(5.5, 1/3.0),
                                  use_jumps)

    # Compute all the initial sets of ramps
    cdef deque[stack[RampIndex]] pixel_ramps = init_ramps(dq)

    cdef cpp_list[cpp_list[float]] slopes, read_vars, poisson_vars
    cdef cpp_list[cpp_list[int]] starts, ends

    # Perform all of the fits
    cdef RampFits ramp_fits
    cdef int index
    for index in range(n_resultants):
        # Fit all the ramps for the given pixel
        ramp_fits = make_pixel(fixed, read_noise,
                               resultants[:, index]).fit_ramps(pixel_ramps[index])

        # Build the output arrays
        slopes.push_back(ramp_fits.slope)
        read_vars.push_back(ramp_fits.read_var)
        poisson_vars.push_back(ramp_fits.poisson_var)
        starts.push_back(ramp_fits.start)
        ends.push_back(ramp_fits.end)

    return dict(slope=slopes, read_var=read_vars,
                poisson_var=poisson_vars, start=starts, end=ends)
