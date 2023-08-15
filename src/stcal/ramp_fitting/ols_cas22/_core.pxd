cdef class Ramp:
    cdef public int start, end
    cdef public float read_noise
    cdef public float [:] resultants, t_bar, tau
    cdef public int [:] n_reads

    cdef (float, float, float) fit(Ramp self)


cdef Ramp make_ramp(
    float [:] resultants, int start, int end, float read_noise,
    float [:] t_bar, float [:] tau, int [:] n_reads)
