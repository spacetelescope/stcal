from libcpp cimport bool

from stcal.ramp_fitting.ols_cas22._core cimport Thresh, DerivedData


cdef class Fixed:
    cdef bool use_jump
    cdef float[:] t_bar, tau
    cdef int[:] n_reads
    cdef Thresh threshold

    cdef float[:] t_bar_1, t_bar_2
    cdef float[:] t_bar_1_sq, t_bar_2_sq
    cdef float[:] recip_1, recip_2
    cdef float[:] slope_var_1, slope_var_2

    cdef float[:] t_bar_diff(Fixed self, int offset)
    cdef float[:] t_bar_diff_sq(Fixed self, int offset)
    cdef float[:] recip_val(Fixed self, int offset)
    cdef float[:] slope_var_val(Fixed self, int offset)

    cdef float correction(Fixed self, int i, int j)


cdef Fixed make_fixed(DerivedData data, Thresh threshold, bool use_jump)
