from libcpp cimport bool

from stcal.ramp_fitting.ols_cas22._core cimport Thresh, ReadPatternMetadata


cdef class Fixed:
    cdef bool use_jump
    cdef ReadPatternMetadata data
    cdef Thresh threshold

    cdef float[:, :] t_bar_diff
    cdef float[:, :] recip
    cdef float[:, :] slope_var

    cdef float[:, :] t_bar_diff_val(Fixed self)
    cdef float[:, :] recip_val(Fixed self)
    cdef float[:, :] slope_var_val(Fixed self)


cdef Fixed make_fixed(ReadPatternMetadata data, Thresh threshold, bool use_jump)
