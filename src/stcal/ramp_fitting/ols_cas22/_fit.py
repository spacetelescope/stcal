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

from typing import TYPE_CHECKING, NamedTuple

import cython
import numpy as np
from cython.cimports import numpy as cnp
from cython.cimports.libcpp.list import list as cpp_list
from cython.cimports.stcal.ramp_fitting.ols_cas22._jump import (
    JumpFits,
    Parameter,
    Thresh,
    Variance,
    _fill_fixed_values,
    fit_jumps,
    n_fixed_offsets,
    n_pixel_offsets,
)
from cython.cimports.stcal.ramp_fitting.ols_cas22._ramp import _fill_metadata

if TYPE_CHECKING:
    from cython.cimports.libcpp import bool as cpp_bool
    from cython.cimports.libcpp.vector import vector

# Initialize numpy for cython use in this module
cnp.import_array()


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


_slope = cython.declare(cython.int, Parameter.slope)

_read_var = cython.declare(cython.int, Variance.read_var)
_poisson_var = cython.declare(cython.int, Variance.poisson_var)
_total_var = cython.declare(cython.int, Variance.total_var)


@cython.boundscheck(False)
@cython.wraparound(False)
def fit_ramps(
    resultants: cython.float[:, :],
    dq: cython.int[:, :],
    read_noise: cython.float[:],
    read_time: cython.float,
    read_pattern: vector[vector[cython.int]],
    use_jump: cpp_bool = False,
    intercept: cython.float = 5.5,
    constant: cython.float = 1 / 3,
    include_diagnostic: cpp_bool = False,
) -> RampFitOutputs:
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

    # Raise error if input data is inconsistent
    if n_resultants != len(read_pattern):
        msg = (
            f"The read pattern length {len(read_pattern)} does "
            f"not match number of resultants {n_resultants}"
        )
        raise RuntimeError(msg)

    # Compute the main metadata from the read pattern and cast it to memory views
    t_bar: cython.float[:] = np.empty(n_resultants, dtype=np.float32)
    tau: cython.float[:] = np.empty(n_resultants, dtype=np.float32)
    n_reads: cython.int[:] = np.empty(n_resultants, dtype=np.int32)
    _fill_metadata(t_bar, tau, n_reads, read_pattern, read_time, n_resultants)

    # Setup pre-compute arrays for jump detection
    single_pixel: cython.float[:, :]
    double_pixel: cython.float[:, :]
    single_fixed: cython.float[:, :]
    double_fixed: cython.float[:, :]
    if use_jump:
        # Initialize arrays for the jump detection pre-computed values
        single_pixel = np.empty((n_pixel_offsets, n_resultants - 1), dtype=np.float32)
        double_pixel = np.empty((n_pixel_offsets, n_resultants - 2), dtype=np.float32)

        single_fixed = np.empty((n_fixed_offsets, n_resultants - 1), dtype=np.float32)
        double_fixed = np.empty((n_fixed_offsets, n_resultants - 2), dtype=np.float32)

        # Pre-compute the values from the read pattern
        _fill_fixed_values(single_fixed, double_fixed, t_bar, tau, n_reads, n_resultants)
    else:
        # "Initialize" the arrays when not using jump detection, they need to be
        #    initialized because they do get passed around, but they don't need
        #    to actually have any entries
        single_pixel = np.empty((0, 0), dtype=np.float32)
        double_pixel = np.empty((0, 0), dtype=np.float32)

        single_fixed = np.empty((0, 0), dtype=np.float32)
        double_fixed = np.empty((0, 0), dtype=np.float32)

    # Create a threshold struct
    thresh: Thresh = Thresh(intercept, constant)

    # Create variable to old the diagnostic data
    # Use list because this might grow very large which would require constant
    #    reallocation. We don't need random access, and this gets cast to a python
    #    list in the end.
    ramp_fits: cpp_list[JumpFits] = cpp_list[JumpFits]()

    # Initialize the output arrays. Note that the fit intercept is currently always
    #    zero, where as every variance is calculated and set. This means that the
    #    parameters need to be filled with zeros, where as the variances can just
    #    be allocated
    parameters: cython.float[:, :] = np.zeros((n_pixels, Parameter.n_param), dtype=np.float32)
    variances: cython.float[:, :] = np.empty((n_pixels, Variance.n_var), dtype=np.float32)

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
    return RampFitOutputs(
        np.array(parameters, dtype=np.float32),
        np.array(variances, dtype=np.float32),
        np.array(dq, dtype=np.uint32),
        ramp_fits if include_diagnostic else None,
    )
