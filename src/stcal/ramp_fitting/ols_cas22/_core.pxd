import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.stack cimport stack
from libcpp cimport bool

from stcal.ramp_fitting.ols_cas22._fixed cimport Fixed

cdef struct RampIndex:
    int start
    int end


cdef struct Thresh:
    float intercept
    float constant


cdef struct Fit:
    float slope
    float read_var
    float poisson_var


cdef struct Fits:
    vector[float] slope
    vector[float] read_var
    vector[float] poisson_var

cdef float get_power(float s)
cdef float threshold(Thresh thresh, float slope)
cdef Fits reverse_fits(Fits fits)
