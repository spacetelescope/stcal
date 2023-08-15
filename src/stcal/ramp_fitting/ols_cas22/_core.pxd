cdef class Ramp:
    cdef public:
        int start, end
        float read_noise
        float [:] resultants, t_bar, tau
        int [:] n_reads


cdef Ramp make_ramp(
    float [:] resultants, int start, int end, float read_noise,
    float [:] t_bar, float [:] tau, int [:] n_reads)