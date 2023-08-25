"""
Define the basic types and functions for the CAS22 algorithm with jump detection

Structs:
-------
    RampIndex
        int start: starting index of the ramp in the resultants
        int end: ending index of the ramp in the resultants
    RampFit
        float slope: slope of a single ramp
        float read_var: read noise variance of a single ramp
        float poisson_var: poisson noise variance of single ramp
    RampFits
        cpp_list[float] slope: slopes of the ramps for a single pixel
        cpp_list[float] read_var: read noise variances of the ramps for a single
                                  pixel
        cpp_list[float] poisson_var: poisson noise variances of the ramps for a
                                     single pixel
    DerivedData
        vector[float] t_bar: mean time of each resultant
        vector[float] tau: variance time of each resultant
        vector[int] n_reads: number of reads in each resultant

Objects
-------
    Thresh : class
        Hold the threshold parameters and compute the threshold

Functions:
----------
    get_power
        Return the power from Casertano+22, Table 2
    threshold
        Compute jump threshold
    init_ramps
        Find initial ramps for each pixel
    read_ma_table
        Read the MA table and Derive the necessary data from it
"""
from libc.math cimport log10
import numpy as np
cimport numpy as np

from stcal.ramp_fitting.ols_cas22._core cimport Thresh, DerivedData


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


cdef class Thresh:
    cdef inline float run(Thresh self, float slope):
        """
        Compute jump threshold

        Parameters
        ----------
        slope : float
            slope of the ramp in question

        Returns
        -------
            intercept - constant * log10(slope)
        """
        return self.intercept - self.constant * log10(slope)


cdef Thresh make_threshold(float intercept, float constant):
    """
    Create a Thresh object

    Parameters
    ----------
    intercept : float
        intercept of the threshold
    constant : float
        constant of the threshold

    Returns
    -------
    Thresh object
    """

    thresh = Thresh()
    thresh.intercept = intercept
    thresh.constant = constant

    return thresh


cdef inline vector[stack[RampIndex]] init_ramps(int[:, :] dq):
    """
    Create the initial ramp stack for each pixel
        if dq[index_resultant, index_pixel] == 0, then the resultant is in a ramp
        otherwise, the resultant is not in a ramp

    Parameters
    ----------
    dq : int[n_resultants, n_pixel]
        DQ array

    Returns
    -------
    Vector of stacks of RampIndex objects
        - Vector with entry for each pixel
        - Stack with entry for each ramp found (top of stack is last ramp found)
        - RampIndex with start and end indices of the ramp in the resultants
    """
    cdef int n_pixel, n_resultants

    n_resultants, n_pixel = np.array(dq).shape
    cdef vector[stack[RampIndex]] pixel_ramps = vector[stack[RampIndex]](n_pixel)

    cdef int index_resultant, index_pixel
    cdef stack[RampIndex] ramps
    cdef RampIndex ramp

    for index_pixel in range(n_pixel):
        ramps = stack[RampIndex]()

        # Note: if start/end are -1, then no value has been assigned
        # ramp.start == -1 means we have not started a ramp
        # dq[index_resultant, index_pixel] == 0 means resultant is in ramp
        ramp = RampIndex(-1, -1)
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
                    ramps.push(ramp)
                    ramp = RampIndex(-1, -1)

        # Handle case where last resultant is in ramp (so no end has been set)
        if ramp.start != -1 and ramp.end == -1:
            # Last resultant is end of the ramp => set then add to stack
            ramp.end = n_resultants - 1
            ramps.push(ramp)

        # Add ramp stack for pixel to vector
        pixel_ramps.push_back(ramps)

    return pixel_ramps


cdef inline DerivedData read_data(list[list[int]] ma_table, float read_time):
    """
    Derive the input data from the MA table

    Note the MA table is a list of pairs of ints for each resultant:
        (first read index, number of reads in resultant)

    Parameters
    ----------
    ma_table : list[list[int]]
        MA table
    read_time : float
        Time to perform a readout.

    Returns
    -------
    DerivedData struct:
        vector[float] t_bar: mean time of each resultant
        vector[float] tau: variance time of each resultant
        vector[int] n_reads: number of reads in each resultant
    """
    cdef int n_resultants = len(ma_table)
    cdef DerivedData data = DerivedData(vector[float](n_resultants),
                                        vector[float](n_resultants),
                                        vector[int](n_resultants))

    cdef int index
    cdef list[int] entry
    for index, entry in enumerate(ma_table):
        data.n_reads[index] = entry[1]
        data.t_bar[index] = read_time *(entry[0] + (entry[1] - 1) / 2.0)
        data.tau[index] = data.t_bar[index] - (entry[1] - 1) * ((entry[1] + 1) *
                                                                read_time /
                                                                (6 * entry[1]))

    return data
