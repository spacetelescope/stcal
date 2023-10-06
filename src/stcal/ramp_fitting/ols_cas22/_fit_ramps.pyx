import numpy as np
cimport numpy as np
from libcpp cimport bool
from libcpp.stack cimport stack
from libcpp.list cimport list as cpp_list
from libcpp.deque cimport deque
cimport cython

from stcal.ramp_fitting.ols_cas22._core cimport (RampFits, RampIndex, Thresh,
                                                 metadata_from_read_pattern, init_ramps,
                                                 Parameter, Variance, RampJumpDQ)
from stcal.ramp_fitting.ols_cas22._fixed cimport fixed_values_from_metadata, FixedValues
from stcal.ramp_fitting.ols_cas22._pixel cimport make_pixel

from typing import NamedTuple


# Fix the default Threshold values at compile time these values cannot be overridden
#   dynamically at runtime.
DEF DefaultIntercept = 5.5
DEF DefaultConstant = 1/3.0

class RampFitOutputs(NamedTuple):
    """
    Simple tuple wrapper for outputs from the ramp fitting algorithm
        This clarifies the meaning of the outputs via naming them something
        descriptive.
    """

    fits: list
    parameters: np.ndarray
    variances: np.ndarray
    dq: np.ndarray
    # def __init__(self, fits, parameters, variances, dq):
    #     self.fits = fits
    #     self.parameters = parameters
    #     self.variances = variances
    #     self.dq = dq


@cython.boundscheck(False)
@cython.wraparound(False)
def fit_ramps(np.ndarray[float, ndim=2] resultants,
              np.ndarray[int, ndim=2] dq,
              np.ndarray[float, ndim=1] read_noise,
              float read_time,
              list[list[int]] read_pattern,
              bool use_jump=False,
              float intercept=DefaultIntercept,
              float constant=DefaultConstant):
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
    read_noise : np.ndarray[n_pixel]
        the read noise in electrons for each pixel
    read_time : float
        Time to perform a readout. For Roman data, this is FRAME_TIME.
    read_pattern : list[list[int]]
        the read pattern for the image
    use_jump : bool
        If True, use the jump detection algorithm to identify CRs.
        If False, use the DQ array to identify CRs.
    intercept : float
        The intercept value for the threshold function. Default=5.5
    constant : float
        The constant value for the threshold function. Default=1/3.0

    Returns
    -------
    A list of RampFits objects, one for each pixel.
    """
    cdef int n_pixels, n_resultants
    n_resultants = resultants.shape[0]
    n_pixels = resultants.shape[1]

    if n_resultants != len(read_pattern):
        raise RuntimeError(f'The read pattern length {len(read_pattern)} does not '
                           f'match number of resultants {n_resultants}')

    # Pre-compute data for all pixels
    cdef FixedValues fixed = fixed_values_from_metadata(metadata_from_read_pattern(read_pattern, read_time),
                                                        Thresh(intercept, constant),
                                                        use_jump)

    # Compute all the initial sets of ramps
    cdef deque[stack[RampIndex]] pixel_ramps = init_ramps(dq)

    # Use list because this might grow very large which would require constant
    #    reallocation. We don't need random access, and this gets cast to a python
    #    list in the end.
    cdef cpp_list[RampFits] ramp_fits

    cdef np.ndarray[float, ndim=2] parameters = np.zeros((n_pixels, 2), dtype=np.float32)
    cdef np.ndarray[float, ndim=2] variances = np.zeros((n_pixels, 3), dtype=np.float32)

    # Copy the dq array so we can modify it without fear
    cdef np.ndarray[int, ndim=2] fit_dq = dq.copy()

    # Perform all of the fits
    cdef RampFits fit
    cdef int index
    for index in range(n_pixels):
        # Fit all the ramps for the given pixel
        fit = make_pixel(fixed, read_noise[index],
                         resultants[:, index]).fit_ramps(pixel_ramps[index])

        parameters[index, Parameter.slope] = fit.average.slope

        variances[index, Variance.read_var] = fit.average.read_var
        variances[index, Variance.poisson_var] = fit.average.poisson_var
        variances[index, Variance.total_var] = fit.average.read_var + fit.average.poisson_var

        for jump in fit.jumps:
            fit_dq[jump, index] = RampJumpDQ.JUMP_DET

        ramp_fits.push_back(fit)

    return RampFitOutputs(ramp_fits, parameters, variances, fit_dq)
