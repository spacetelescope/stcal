from libcpp cimport bool

from stcal.ramp_fitting.ols_cas22._core cimport Thresh, ReadPatternMetadata


cdef class FixedValues:
    cdef bool use_jump
    cdef ReadPatternMetadata data
    cdef Thresh threshold

    cdef float[:, :] t_bar_diff
    cdef float[:, :] recip
    cdef float[:, :] slope_var

    cdef float[:, :] t_bar_diff_val(FixedValues self)
    cdef float[:, :] recip_val(FixedValues self)
    cdef float[:, :] slope_var_val(FixedValues self)


cdef FixedValues fixed_values_from_metadata(ReadPatternMetadata data, Thresh threshold, bool use_jump)
