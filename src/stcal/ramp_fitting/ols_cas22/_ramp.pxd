# cython: language_level=3str

from libcpp.vector cimport vector


cdef struct RampIndex:
    int start
    int end


cdef struct RampFit:
    float slope
    float read_var
    float poisson_var


ctypedef vector[RampIndex] RampQueue


cdef class ReadPattern:
    cdef float[::1] t_bar
    cdef float[::1] tau
    cdef int[::1] n_reads


cpdef RampQueue init_ramps(int[:] dq,
                           int n_resultants)


cpdef ReadPattern from_read_pattern(list[list[int]] read_pattern,
                                    float read_time,
                                    int n_resultants)


cdef RampFit fit_ramp(float[:] resultants_,
                      float[:] t_bar_,
                      float[:] tau_,
                      int[:] n_reads,
                      float read_noise,
                      RampIndex ramp)
