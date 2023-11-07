cimport numpy as cnp


cdef struct Thresh:
    cnp.float32_t intercept
    cnp.float32_t constant


cpdef cnp.float32_t threshold(Thresh thresh, cnp.float32_t slope)