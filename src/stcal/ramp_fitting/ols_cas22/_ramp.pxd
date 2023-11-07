from libcpp.vector cimport vector


cdef struct RampIndex:
    int start
    int end


ctypedef vector[RampIndex] RampQueue


cpdef RampQueue init_ramps(int[:, :] dq, int n_resultants, int index_pixel)
