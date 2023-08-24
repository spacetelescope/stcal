from libcpp.stack cimport stack

from stcal.ramp_fitting.ols_cas22._core cimport Fit, Fits, RampIndex, Thresh
from stcal.ramp_fitting.ols_cas22._fixed cimport Fixed

cdef class Ramp:
    cdef Fixed fixed
    cdef public float read_noise
    cdef public float [:] resultants

    # Computed and cached values for jump detection
    #    single -> j = i + 1
    #    double -> j = i + 2

    # single and double delta + slope
    #    (resultants[j] - resultants[i]/(t_bar[j] - t_bar[i])
    cdef public float[:] delta_1, delta_2 

    # single and double sigma terms
    #    read_noise * recip[i]
    cdef public float[:] sigma_1, sigma_2

    cdef float[:] resultants_diff(Ramp self, int offset)
    cdef Fit fit(Ramp self, RampIndex ramp)

    cdef float[:] stats(Ramp self, float slope, RampIndex ramp)
    cdef Fits fits(Ramp self, stack[RampIndex] ramps, Thresh thresh)


cdef Ramp make_ramp(Fixed fixed, float read_noise, float [:] resultants)