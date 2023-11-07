cimport numpy as cnp

from libcpp.vector cimport vector


cdef struct RampIndex:
    cnp.int32_t start
    cnp.int32_t end


ctypedef vector[RampIndex] RampQueue


cpdef RampQueue init_ramps(cnp.int32_t[:, :] dq, cnp.int32_t n_resultants, cnp.int32_t index_pixel)
