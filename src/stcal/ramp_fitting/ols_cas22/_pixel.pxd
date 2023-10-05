from libcpp.stack cimport stack

from stcal.ramp_fitting.ols_cas22._core cimport RampFit, RampFits, RampIndex
from stcal.ramp_fitting.ols_cas22._fixed cimport FixedValues

cdef class Pixel:
    cdef FixedValues fixed
    cdef float read_noise
    cdef float [:] resultants

    cdef float[:, :] delta
    cdef float[:, :] sigma

    cdef float[:, :] delta_val(Pixel self)
    cdef RampFit fit_ramp(Pixel self, RampIndex ramp)

    cdef float correction(Pixel self, RampIndex ramp, int index, int diff)
    cdef float stat(Pixel self, float slope, RampIndex ramp, int index, int diff)
    cdef float[:] stats(Pixel self, float slope, RampIndex ramp)
    cdef RampFits fit_ramps(Pixel self, stack[RampIndex] ramps)


cdef Pixel make_pixel(FixedValues fixed, float read_noise, float [:] resultants)
