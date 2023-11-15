# cython: language_level=3str

from libcpp.vector cimport vector


cpdef void _fill_metadata(list[list[int]] read_pattern,
                          float read_time,
                          float[:] t_bar,
                          float[:] tau,
                          int[:] n_reads)


cdef struct RampIndex:
    int start
    int end


ctypedef vector[RampIndex] RampQueue


cpdef RampQueue init_ramps(int[:] dq,
                           int n_resultants)


cdef struct RampFit:
    float slope
    float read_var
    float poisson_var


cdef RampFit fit_ramp(float[:] resultants_,
                      float[:] t_bar_,
                      float[:] tau_,
                      int[:] n_reads,
                      float read_noise,
                      RampIndex ramp)
