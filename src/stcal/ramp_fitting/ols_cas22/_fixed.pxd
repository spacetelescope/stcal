from libcpp cimport bool

from stcal.ramp_fitting.ols_cas22._jump cimport Thresh
from stcal.ramp_fitting.ols_cas22._read_pattern cimport ReadPattern


cdef class FixedValues:
    cdef bool use_jump
    cdef ReadPattern data
    cdef Thresh threshold

    cdef float[:, :] t_bar_diffs
    cdef float[:, :] t_bar_diff_sqrs
    cdef float[:, :] read_recip_coeffs
    cdef float[:, :] var_slope_coeffs


cpdef FixedValues fixed_values_from_metadata(ReadPattern data, Thresh threshold, bool use_jump)
