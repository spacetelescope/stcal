from libcpp cimport bool

from stcal.ramp_fitting.ols_cas22._fixed cimport FixedValues
from stcal.ramp_fitting.ols_cas22._ramp cimport RampIndex, RampFit

cdef class Pixel:
    cdef FixedValues fixed
    cdef float read_noise
    cdef float[:] resultants

    cdef float[:, :] local_slopes
    cdef float[:, :] var_read_noise


cpdef Pixel make_pixel(FixedValues fixed, float read_noise, float [:] resultants)
