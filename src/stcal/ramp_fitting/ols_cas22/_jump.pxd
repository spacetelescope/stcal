from libcpp cimport bool
from libcpp.vector cimport vector

from stcal.ramp_fitting.ols_cas22._ramp cimport RampFit, RampQueue
from stcal.ramp_fitting.ols_cas22._fixed cimport FixedValues

cdef struct Thresh:
    float intercept
    float constant


cdef struct RampFits:
    RampFit average
    vector[int] jumps
    vector[RampFit] fits
    RampQueue index

cdef RampFits fit_jumps(float[:] resultants,
                        float read_noise,
                        RampQueue ramps,
                        FixedValues fixed,
                        float[:, :] pixel,
                        Thresh thresh,
                        bool include_diagnostic)
