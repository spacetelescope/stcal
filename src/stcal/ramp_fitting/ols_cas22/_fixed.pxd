from libcpp cimport bool

from stcal.ramp_fitting.ols_cas22._core cimport ReadPatternMetadata
from stcal.ramp_fitting.ols_cas22._jump cimport Thresh


cdef class FixedValues:
    cdef bool use_jump
    cdef ReadPatternMetadata data
    cdef Thresh threshold

    cdef float[:, :] t_bar_diffs
    cdef float[:, :] t_bar_diff_sqrs
    cdef float[:, :] read_recip_coeffs
    cdef float[:, :] var_slope_coeffs

    cdef float[:, :] t_bar_diff_vals(FixedValues self)
    cdef float[:, :] read_recip_vals(FixedValues self)
    cdef float[:, :] var_slope_vals(FixedValues self)


cpdef FixedValues fixed_values_from_metadata(ReadPatternMetadata data, Thresh threshold, bool use_jump)
