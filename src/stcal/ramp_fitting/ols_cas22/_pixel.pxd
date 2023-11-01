from libcpp cimport bool
from libcpp.stack cimport stack

from stcal.ramp_fitting.ols_cas22._core cimport RampFit, RampFits, RampIndex
from stcal.ramp_fitting.ols_cas22._fixed cimport FixedValues

cdef class Pixel:
    cdef FixedValues fixed
    cdef float read_noise
    cdef float [:] resultants

    cdef float[:, :] local_slopes
    cdef float[:, :] var_read_noise

    cdef float[:, :] local_slope_vals(Pixel self)
    cdef RampFit fit_ramp(Pixel self, RampIndex ramp)

    cdef float correction(Pixel self, RampIndex ramp, float slope)
    cdef float stat(Pixel self, float slope, RampIndex ramp, int index, int diff)
    cdef float[:] stats(Pixel self, float slope, RampIndex ramp)
    cdef RampFits fit_ramps(Pixel self, stack[RampIndex] ramps, bool include_diagnostic)


cpdef Pixel make_pixel(FixedValues fixed, float read_noise, float [:] resultants)
