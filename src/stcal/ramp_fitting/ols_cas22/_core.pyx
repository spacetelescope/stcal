"""
Define the basic types and functions for the CAS22 algorithm with jump detection

Structs
-------
    RampIndex
        int start: starting index of the ramp in the resultants
        int end: ending index of the ramp in the resultants

            Note that the Python range would be [start:end+1] for any ramp index.
    RampFit
        float slope: slope of a single ramp
        float read_var: read noise variance of a single ramp
        float poisson_var: poisson noise variance of single ramp
    RampFits
        vector[RampFit] fits: ramp fits (in time order) for a single pixel
        vector[RampIndex] index: ramp indices (in time order) for a single pixel
        RampFit average: average ramp fit for a single pixel
    ReadPatternMetata
        vector[float] t_bar: mean time of each resultant
        vector[float] tau: variance time of each resultant
        vector[int] n_reads: number of reads in each resultant

            Note that these are entirely computed from the read_pattern and
            read_time (which should be constant for a given telescope) for the
            given observation.
    Thresh
        float intercept: intercept of the threshold
        float constant: constant of the threshold

Enums
-----
    Diff
        This is the enum to track the index for single vs double difference related
        computations.

        single: single difference
        double: double difference

    Parameter
        This is the enum to track the index of the computed fit parameters for
        the ramp fit.

        intercept: the intercept of the ramp fit
        slope: the slope of the ramp fit

    Variance
        This is the enum to track the index of the computed variance values for
        the ramp fit.

        read_var: read variance computed
        poisson_var: poisson variance computed
        total_var: total variance computed (read_var + poisson_var)

    RampJumpDQ
        This enum is to specify the DQ flags for Ramp/Jump detection

        JUMP_DET: jump detected

Functions
---------
    get_power
        Return the power from Casertano+22, Table 2
    threshold
        Compute jump threshold
        - cpdef gives a python wrapper, but the python version of this method
          is considered private, only to be used for testing
    init_ramps
        Find initial ramps for each pixel, accounts for DQ flags
        - A python wrapper, _init_ramps_list, that adjusts types so they can
          be directly inspected in python exists for testing purposes only.
    metadata_from_read_pattern
        Read the read pattern and derive the baseline metadata parameters needed
        - cpdef gives a python wrapper, but the python version of this method
          is considered private, only to be used for testing
"""
from libcpp.vector cimport vector
from libc.math cimport log10

import numpy as np
cimport numpy as np
cimport cython

from stcal.ramp_fitting.ols_cas22._core cimport Thresh, ReadPatternMetadata


# Casertano+2022, Table 2
cdef float[2][6] PTABLE = [
    [-np.inf, 5, 10, 20, 50, 100],
    [0,     0.4,  1,  3,  6,  10]]


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline float get_power(float signal):
    """
    Return the power from Casertano+22, Table 2

    Parameters
    ----------
    signal: float
        signal from the resultants

    Returns
    -------
    signal power from Table 2
    """
    cdef int i
    for i in range(6):
        if signal < PTABLE[0][i]:
            return PTABLE[1][i - 1]

    return PTABLE[1][i]


cpdef inline float threshold(Thresh thresh, float slope):
    """
    Compute jump threshold

    Parameters
    ----------
    thresh : Thresh
        threshold parameters struct
    slope : float
        slope of the ramp in question

    Returns
    -------
        intercept - constant * log10(slope)
    """
    # clip slope in 1, 1e4
    slope = slope if slope > 1 else 1
    slope = slope if slope < 1e4 else 1e4
    return thresh.intercept - thresh.constant * log10(slope)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef inline vector[RampIndex] init_ramps(int[:, :] dq, int n_resultants, int index_pixel):
    """
    Create the initial ramp stack for each pixel
        if dq[index_resultant, index_pixel] == 0, then the resultant is in a ramp
        otherwise, the resultant is not in a ramp

    Parameters
    ----------
    dq : int[n_resultants, n_pixel]
        DQ array
    n_resultants : int
        Number of resultants
    index_pixel : int
        The index of the pixel to find ramps for

    Returns
    -------
    vector of RampIndex objects
        - vector with entry for each ramp found (last entry is last ramp found)
        - RampIndex with start and end indices of the ramp in the resultants
    """
    cdef vector[RampIndex] ramps = vector[RampIndex]()

    # Note: if start/end are -1, then no value has been assigned
    # ramp.start == -1 means we have not started a ramp
    # dq[index_resultant, index_pixel] == 0 means resultant is in ramp
    cdef RampIndex ramp = RampIndex(-1, -1)
    for index_resultant in range(n_resultants):
        if ramp.start == -1:
            # Looking for the start of a ramp
            if dq[index_resultant, index_pixel] == 0:
                # We have found the start of a ramp!
                ramp.start = index_resultant
            else:
                # This is not the start of the ramp yet
                continue
        else:
            # Looking for the end of a ramp
            if dq[index_resultant, index_pixel] == 0:
                # This pixel is in the ramp do nothing
                continue
            else:
                # This pixel is not in the ramp
                # => index_resultant - 1 is the end of the ramp
                ramp.end = index_resultant - 1

                # Add completed ramp to stack and reset ramp
                ramps.push_back(ramp)
                ramp = RampIndex(-1, -1)

    # Handle case where last resultant is in ramp (so no end has been set)
    if ramp.start != -1 and ramp.end == -1:
        # Last resultant is end of the ramp => set then add to stack
        ramp.end = n_resultants - 1
        ramps.push_back(ramp)

    return ramps


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef ReadPatternMetadata metadata_from_read_pattern(list[list[int]] read_pattern, float read_time):
    """
    Derive the input data from the the read pattern

        read pattern is a list of resultant lists, where each resultant list is
        a list of the reads in that resultant.

    Parameters
    ----------
    read pattern: list[list[int]]
        read pattern for the image
    read_time : float
        Time to perform a readout.

    Returns
    -------
    ReadPatternMetadata struct:
        vector[float] t_bar: mean time of each resultant
        vector[float] tau: variance time of each resultant
        vector[int] n_reads: number of reads in each resultant
    """
    cdef int n_resultants = len(read_pattern)
    cdef ReadPatternMetadata data = ReadPatternMetadata(vector[float](n_resultants),
                                                        vector[float](n_resultants),
                                                        vector[int](n_resultants))

    cdef int index, n_reads
    cdef list[int] resultant
    for index, resultant in enumerate(read_pattern):
            n_reads = len(resultant)

            data.n_reads[index] = n_reads
            data.t_bar[index] = read_time * np.mean(resultant)
            data.tau[index] = np.sum((2 * (n_reads - np.arange(n_reads)) - 1) * resultant) * read_time / n_reads**2

    return data
