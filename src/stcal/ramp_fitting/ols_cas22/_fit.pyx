# cython: language_level=3str

"""
External interface module for the Casertano+22 ramp fitting algorithm with jump detection.
    This module is intended to contain everything needed by external code.

Enums
-----
Parameter :
    Enumerate the index for the output parameters array.

Variance :
    Enumerate the index for the output variances array.

Classes
-------
RampFitOutputs : NamedTuple
    Simple tuple wrapper for outputs from the ramp fitting algorithm
        This clarifies the meaning of the outputs via naming them something
        descriptive.

(Public) Functions
------------------
fit_ramps : function
    Fit ramps using the Castenario+22 algorithm to a set of pixels accounting
    for jumps (if use_jump is True) and bad pixels (via the dq array). This
    is the primary externally callable function.
"""
from __future__ import annotations

import numpy as np

cimport numpy as cnp
from cython cimport boundscheck, wraparound
from libcpp cimport bool
from libcpp.list cimport list as cpp_list

from stcal.ramp_fitting.ols_cas22._jump cimport (
    JumpFits,
    Thresh,
    fill_fixed_values,
    fit_jumps,
    n_fixed_offsets,
    n_pixel_offsets,
)
from stcal.ramp_fitting.ols_cas22._ramp cimport ReadPattern, from_read_pattern

from typing import NamedTuple

# Initialize numpy for cython use in this module
cnp.import_array()


cpdef enum Parameter:
    intercept
    slope
    n_param


cpdef enum Variance:
    read_var
    poisson_var
    total_var
    n_var


class RampFitOutputs(NamedTuple):
    """
    Simple tuple wrapper for outputs from the ramp fitting algorithm
        This clarifies the meaning of the outputs via naming them something
        descriptive.

    Attributes
    ----------
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
        fits: list of RampFits
            the raw ramp fit outputs, these are all structs which will get mapped to
            python dictionaries.
    """
    parameters: np.ndarray
    variances: np.ndarray
    dq: np.ndarray
    fits: list | None = None


@boundscheck(False)
@wraparound(False)
def fit_ramps(float[:, :] resultants,
              cnp.ndarray[int, ndim=2] dq,
              float[:] read_noise,
              float read_time,
              list[list[int]] read_pattern,
              bool use_jump=False,
              float intercept=5.5,
              float constant=1/3,
              bool include_diagnostic=False):
    """Fit ramps using the Casertano+22 algorithm.
        This implementation uses the Cas22 algorithm to fit ramps, where
        ramps are fit between bad resultants marked by dq flags for each pixel
        which are not equal to zero. If use_jump is True, it additionally uses
        jump detection to mark additional resultants for each pixel as bad if
        a jump is suspected in them.

    Parameters
    ----------
    resultants : float[n_resultants, n_pixel]
        the resultants in electrons (Note that this can be based as any sort of
        array, such as a numpy array. The memory view is just for efficiency in
        cython)
    dq : np.ndarry[n_resultants, n_pixel]
        the dq array.  dq != 0 implies bad pixel / CR. (Kept as a numpy array
        so that it can be passed out without copying into new numpy array, will
        be working on memory views of this array)
    read_noise : float[n_pixel]
        the read noise in electrons for each pixel (same note as the resultants)
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
    include_diagnostic : bool
        If True, include the raw ramp fits in the output. Default=False

    Returns
    -------
    A RampFitOutputs tuple
    """
    cdef int n_pixels, n_resultants
    n_resultants = resultants.shape[0]
    n_pixels = resultants.shape[1]

    # Raise error if input data is inconsistent
    if n_resultants != len(read_pattern):
        raise RuntimeError(f'The read pattern length {len(read_pattern)} does not '
                           f'match number of resultants {n_resultants}')

    # Compute the main metadata from the read pattern and cast it to memory views
    cdef ReadPattern metadata = from_read_pattern(read_pattern, read_time, n_resultants)
    cdef float[:] t_bar = metadata.t_bar
    cdef float[:] tau = metadata.tau
    cdef int[:] n_reads = metadata.n_reads

    # Setup pre-compute arrays for jump detection
    cdef float[:, :] fixed
    cdef float[:, :] pixel
    if use_jump:
        # Initialize arrays for the jump detection pre-computed values
        fixed = np.empty((n_fixed_offsets, n_resultants - 1), dtype=np.float32)
        pixel = np.empty((n_pixel_offsets, n_resultants - 1), dtype=np.float32)

        # Pre-compute the values from the read pattern
        fixed = fill_fixed_values(fixed, t_bar, tau, n_reads, n_resultants)
    else:
        # "Initialize" the arrays when not using jump detection, they need to be
        #    initialized because they do get passed around, but they don't need
        #    to actually have any entries
        fixed = np.empty((0, 0), dtype=np.float32)
        pixel = np.empty((0, 0), dtype=np.float32)

    # Create a threshold struct
    cdef Thresh thresh = Thresh(intercept, constant)

    # Create variable to old the diagnostic data
    # Use list because this might grow very large which would require constant
    #    reallocation. We don't need random access, and this gets cast to a python
    #    list in the end.
    cdef cpp_list[JumpFits] ramp_fits

    # Initialize the output arrays. Note that the fit intercept is currently always
    #    zero, where as every variance is calculated and set. This means that the
    #    parameters need to be filled with zeros, where as the variances can just
    #    be allocated
    cdef float[:, :] parameters = np.zeros((n_pixels, Parameter.n_param), dtype=np.float32)
    cdef float[:, :] variances = np.empty((n_pixels, Variance.n_var), dtype=np.float32)

    # Cast the enum values into integers for indexing (otherwise compiler complains)
    #   These will be optimized out
    cdef int slope = Parameter.slope
    cdef int read_var = Variance.read_var
    cdef int poisson_var = Variance.poisson_var
    cdef int total_var = Variance.total_var

    # Pull memory view of dq for speed of access later
    #   changes to this array will backpropagate to the original numpy array
    cdef int[:, :] dq_ = dq

    # Run the jump fitting algorithm for each pixel
    cdef JumpFits fit
    cdef int index
    for index in range(n_pixels):
        # Fit all the ramps for the given pixel
        fit = fit_jumps(resultants[:, index],
                        dq_[:, index],
                        read_noise[index],
                        t_bar,
                        tau,
                        n_reads,
                        n_resultants,
                        fixed,
                        pixel,
                        thresh,
                        use_jump,
                        include_diagnostic)

        # Extract the output fit's parameters
        parameters[index, slope] = fit.average.slope

        # Extract the output fit's variances
        variances[index, read_var] = fit.average.read_var
        variances[index, poisson_var] = fit.average.poisson_var
        variances[index, total_var] = fit.average.read_var + fit.average.poisson_var

        # Store diagnostic data if requested
        if include_diagnostic:
            ramp_fits.push_back(fit)

    # Cast memory views into numpy arrays for ease of use in python.
    return RampFitOutputs(np.array(parameters, dtype=np.float32),
                          np.array(variances, dtype=np.float32),
                          dq,
                          ramp_fits if include_diagnostic else None)
