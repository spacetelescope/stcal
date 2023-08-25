import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.stack cimport stack
from libcpp.list cimport list as cpp_list
from libcpp.deque cimport deque
from libcpp cimport bool

from stcal.ramp_fitting.ols_cas22._fixed cimport Fixed

cdef struct RampIndex:
    int start
    int end


cdef struct RampFit:
    float slope
    float read_var
    float poisson_var


cdef struct RampFits:
    cpp_list[float] slope
    cpp_list[float] read_var
    cpp_list[float] poisson_var
    cpp_list[int] start
    cpp_list[int] end


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
cdef deque[stack[RampIndex]] init_ramps(int[:, :] dq)
cdef DerivedData read_data(list[list[int]] ma_table, float read_time)
