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
import cython
from cython.cimports.libcpp import bool as cpp_bool
from cython.cimports.libcpp.list import list as cpp_list
from cython.cimports.libcpp.vector import vector
from cython.cimports.stcal.ramp_fitting.ols_cas22._jump import (
    JumpFits,
    Thresh,
    _fill_fixed_values,
    fit_jumps,
)
from cython.cimports.stcal.ramp_fitting.ols_cas22._ramp import _fill_metadata


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.ccall
def fit_ramps(
    resultants: cython.float[:, :],
    dq: cython.int[:, :],
    read_noise: cython.float[:],
    read_time: cython.float,
    read_pattern: vector[vector[cython.int]],
    parameters: cython.float[:, :],
    variances: cython.float[:, :],
    t_bar: cython.float[:],
    tau: cython.float[:],
    n_reads: cython.int[:],
    single_pixel: cython.float[:, :],
    double_pixel: cython.float[:, :],
    single_fixed: cython.float[:, :],
    double_fixed: cython.float[:, :],
    use_jump: cpp_bool,
    intercept: cython.float,
    constant: cython.float,
    include_diagnostic: cpp_bool,
) -> cpp_list[JumpFits]:
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
    n_resultants: cython.int = resultants.shape[0]
    n_pixels: cython.int = resultants.shape[1]

    # Compute the main metadata from the read pattern and cast it to memory views
    _fill_metadata(t_bar, tau, n_reads, read_pattern, read_time, n_resultants)

    if use_jump:
        # Pre-compute the values from the read pattern
        _fill_fixed_values(single_fixed, double_fixed, t_bar, tau, n_reads, n_resultants)

    # Create a threshold struct
    thresh: Thresh = Thresh(intercept, constant)

    # Create variable to old the diagnostic data
    # Use list because this might grow very large which would require constant
    #    reallocation. We don't need random access, and this gets cast to a python
    #    list in the end.
    ramp_fits: cpp_list[JumpFits] = cpp_list[JumpFits]()

    # Run the jump fitting algorithm for each pixel
    fit: JumpFits
    index: cython.int
    for index in range(n_pixels):
        # Fit all the ramps for the given pixel
        fit = fit_jumps(
            parameters[index, :],
            variances[index, :],
            resultants[:, index],
            dq[:, index],
            read_noise[index],
            t_bar,
            tau,
            n_reads,
            n_resultants,
            single_pixel,
            double_pixel,
            single_fixed,
            double_fixed,
            thresh,
            use_jump,
            include_diagnostic,
        )

        # Store diagnostic data if requested
        if include_diagnostic:
            ramp_fits.push_back(fit)

    # Cast memory views into numpy arrays for ease of use in python.
    return ramp_fits
