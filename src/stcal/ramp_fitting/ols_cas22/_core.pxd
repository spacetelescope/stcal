from libcpp.vector cimport vector
cdef class Ramp:
    cdef public int start, end
    cdef public float read_noise
    cdef public float [:] resultants,
    cdef public vector[float] t_bar, tau
    cdef public vector[int] n_reads

    cdef (float, float, float) fit(Ramp self)


cdef Ramp make_ramp(
    float [:] resultants, int start, int end, float read_noise,
    vector[float] t_bar, vector[float] tau, vector[int] n_reads)
