cdef class ReadPattern:
    cdef float[::1] t_bar
    cdef float[::1] tau
    cdef int[::1] n_reads


cpdef ReadPattern from_read_pattern(list[list[int]] read_pattern, float read_time, int n_resultants)