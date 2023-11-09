"""
Define the data which is fixed for all pixels to compute the CAS22 algorithm with
    jump detection

Objects
-------
FixedValues : class
    Class to contain the data fixed for all pixels and commonly referenced
    universal values for jump detection

Functions
---------
    fixed_values_from_metadata : function
        Fast constructor for FixedValues from the read pattern metadata
            - cpdef gives a python wrapper, but the python version of this method
              is considered private, only to be used for testing
"""
import numpy as np
cimport numpy as cnp

from cython cimport boundscheck, wraparound

from libc.math cimport NAN

from stcal.ramp_fitting.ols_cas22._fixed cimport FixedOffsets



@boundscheck(False)
@wraparound(False)
cpdef inline float[:, :] fill_fixed_values(float[:, :] fixed,
                                           float[:] t_bar,
                                           float[:] tau,
                                           int[:] n_reads,
                                           int n_resultants):
    """
    Compute the difference offset of t_bar

    Returns
    -------
    [
        <t_bar[i+1] - t_bar[i]>,
        <t_bar[i+2] - t_bar[i]>,
        <t_bar[i+1] - t_bar[i]> ** 2,
        <t_bar[i+2] - t_bar[i]> ** 2,
        <(1/n_reads[i+1] + 1/n_reads[i])>,
        <(1/n_reads[i+2] + 1/n_reads[i])>,
        <(tau[i] + tau[i+1] - 2 * min(t_bar[i], t_bar[i+1]))>,
        <(tau[i] + tau[i+2] - 2 * min(t_bar[i], t_bar[i+2]))>,
    ]
    """

    cdef int single_t_bar_diff = FixedOffsets.single_t_bar_diff
    cdef int double_t_bar_diff = FixedOffsets.double_t_bar_diff
    cdef int single_t_bar_diff_sqr = FixedOffsets.single_t_bar_diff_sqr
    cdef int double_t_bar_diff_sqr = FixedOffsets.double_t_bar_diff_sqr
    cdef int single_read_recip = FixedOffsets.single_read_recip
    cdef int double_read_recip = FixedOffsets.double_read_recip
    cdef int single_var_slope_val = FixedOffsets.single_var_slope_val
    cdef int double_var_slope_val = FixedOffsets.double_var_slope_val

    # Coerce division to be using floats
    cdef float num = 1

    cdef int i
    for i in range(n_resultants - 1):
        fixed[single_t_bar_diff, i] = t_bar[i + 1] - t_bar[i]
        fixed[single_t_bar_diff_sqr, i] = fixed[single_t_bar_diff, i] ** 2
        fixed[single_read_recip, i] = (num / n_reads[i + 1]) + (num / n_reads[i])
        fixed[single_var_slope_val, i] = tau[i + 1] + tau[i] - 2 * min(t_bar[i + 1], t_bar[i])

        if i < n_resultants - 2:
            fixed[double_t_bar_diff, i] = t_bar[i + 2] - t_bar[i]
            fixed[double_t_bar_diff_sqr, i] = fixed[double_t_bar_diff, i] ** 2
            fixed[double_read_recip, i] = (num / n_reads[i + 2]) + (num / n_reads[i])
            fixed[double_var_slope_val, i] = tau[i + 2] + tau[i] - 2 * min(t_bar[i + 2], t_bar[i])
        else:
            # Last double difference is undefined
            fixed[double_t_bar_diff, i] = NAN
            fixed[double_t_bar_diff_sqr, i] = NAN
            fixed[double_read_recip, i] = NAN
            fixed[double_var_slope_val, i] = NAN

    return fixed
