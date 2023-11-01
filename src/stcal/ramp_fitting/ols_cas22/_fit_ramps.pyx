import numpy as np
cimport numpy as np
from libcpp cimport bool
from libcpp.list cimport list as cpp_list
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

    Attributes
    ----------
        fits: list of RampFits
            the raw ramp fit outputs, these are all structs which will get mapped to
            python dictionaries.
        parameters: np.ndarray[n_pixel, 2]
            the slope and intercept for each pixel's ramp fit. see Parameter enum
            for indexing indicating slope/intercept in the second dimension.
        variances: np.ndarray[n_pixel, 3]
            the read, poisson, and total variances for each pixel's ramp fit.
            see Variance enum for indexing indicating read/poisson/total in the
            second dimension.
        dq: np.ndarray[n_resultants, n_pixel]
            the dq array, with additional flags set for jumps detected by the
            jump detection algorithm.
    """
    # fits: list
    parameters: np.ndarray
    variances: np.ndarray
    dq: np.ndarray


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
        This implementation uses the Cas22 algorithm to fit ramps, where
        ramps are fit between bad resultants marked by dq flags for each pixel
        which are not equal to zero. If use_jump is True, it additionally uses
        jump detection to mark additional resultants for each pixel as bad if
        a jump is suspected in them.

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
    A RampFitOutputs tuple
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

    # Use list because this might grow very large which would require constant
    #    reallocation. We don't need random access, and this gets cast to a python
    #    list in the end.
    # cdef cpp_list[RampFits] ramp_fits

    cdef np.ndarray[float, ndim=2] parameters = np.zeros((n_pixels, 2), dtype=np.float32)
    cdef np.ndarray[float, ndim=2] variances = np.zeros((n_pixels, 3), dtype=np.float32)

    # Perform all of the fits
    cdef RampFits fit
    cdef int index
    for index in range(n_pixels):
        # Fit all the ramps for the given pixel
        fit = make_pixel(fixed, read_noise[index],
                         resultants[:, index]).fit_ramps(init_ramps(dq, n_resultants, index))

        parameters[index, Parameter.slope] = fit.average.slope

        variances[index, Variance.read_var] = fit.average.read_var
        variances[index, Variance.poisson_var] = fit.average.poisson_var
        variances[index, Variance.total_var] = fit.average.read_var + fit.average.poisson_var

        for jump in fit.jumps:
            dq[jump, index] = RampJumpDQ.JUMP_DET

        # ramp_fits.push_back(fit)

    # return RampFitOutputs(ramp_fits, parameters, variances, dq)
    return RampFitOutputs(parameters, variances, dq)
