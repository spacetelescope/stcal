from libcpp.vector cimport vector

from stcal.ramp_fitting.ols_cas22._pixel cimport Pixel


cdef struct RampIndex:
    int start
    int end


cdef struct RampFit:
    float slope
    float read_var
    float poisson_var


ctypedef vector[RampIndex] RampQueue


cpdef RampQueue init_ramps(int[:, :] dq, int n_resultants, int index_pixel)
cdef RampFit fit_ramp(Pixel pixel, RampIndex ramp)
