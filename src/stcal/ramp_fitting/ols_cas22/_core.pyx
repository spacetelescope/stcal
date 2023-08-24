"""
Define the basic types and functions for the CAS22 algorithm with jump detection

Structs:
-------
    RampIndex
        int start: starting index of the ramp in the resultants
        int end: ending index of the ramp in the resultants
    Thresh
        float intercept: intercept of the threshold
        float constant: constant of the threshold
    Fit
        float slope: slope of a single ramp
        float read_var: read noise variance of a single ramp
        float poisson_var: poisson noise variance of single ramp
    Fits
        vector[float] slope: slopes of the ramps for a single pixel
        vector[float] read_var: read noise variances of the ramps for a single pixel
        vector[float] poisson_var: poisson noise variances of the ramps for a single pixel

Functions:
----------
    get_power
        Return the power from Casertano+22, Table 2
    threshold
        Compute jump threshold
    reverse_fits
        Reverse a Fits struct
"""
from libc.math cimport log10
import numpy as np
cimport numpy as np

from stcal.ramp_fitting.ols_cas22._core cimport RampIndex, Thresh, Fit, Fits, get_power, threshold, reverse_fits


# Casertano+2022, Table 2
cdef float[2][6] PTABLE = [
    [-np.inf, 5, 10, 20, 50, 100],
    [0,     0.4,  1,  3,  6,  10]]


cdef inline float get_power(float s):
    """
    Return the power from Casertano+22, Table 2

    Parameters
    ----------
    s: float
        signal from the resultants

    Returns
    -------
    signal power from Table 2
    """
    cdef int i
    for i in range(6):
        if s < PTABLE[0][i]:
            return PTABLE[1][i - 1]

    return PTABLE[1][i]


cdef inline float threshold(Thresh thresh, float slope):
    """
    Compute jump threshold

    Parameters
    ----------
    thresh : Thresh
        threshold parameters struct:
    slope : float
        slope of the ramp in question

    Returns
    -------
        intercept - constant * log10(slope)
    """
    return thresh.intercept - thresh.constant * log10(slope)


cdef inline Fits reverse_fits(Fits fits):
    """
    Reverse a Fits struct
        The jump detection step computes the ramps in reverse time order for each pixel.
        This reverses the results of the fit to match the original time order, which is
        much faster than prepending to a C++ vector.

    Parameters
    ----------
    fits : Fits
        fits struct to reverse

    Returns
    -------
    reversed fits struct
    """
    return Fits(fits.slope[::-1], fits.read_var[::-1], fits.poisson_var[::-1])
