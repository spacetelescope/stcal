import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.stack cimport stack
from libcpp cimport bool

from stcal.ramp_fitting.ols_cas22._fixed cimport Fixed

cdef struct RampIndex:
    int start
    int end


cdef struct Fit:
    float slope
    float read_var
    float poisson_var


cdef struct Fits:
    vector[float] slope
    vector[float] read_var
    vector[float] poisson_var


cdef struct DerivedData:
    vector[float] t_bar
    vector[float] tau
    vector[int] n_reads


cdef class Thresh:
    cdef float intercept
    cdef float constant

    cdef float run(Thresh self, float slope)

cdef Thresh make_threshold(float intercept, float constant)
cdef float get_power(float s)
cdef Fits reverse_fits(Fits fits)
cdef vector[stack[RampIndex]] init_ramps(int[:, :] dq)
cdef DerivedData read_data(list[list[int]] ma_table, float read_time)
