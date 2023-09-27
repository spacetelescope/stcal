from libcpp.vector cimport vector
from libcpp.stack cimport stack
from libcpp.deque cimport deque


cdef struct RampIndex:
    int start
    int end


cdef struct RampFit:
    float slope
    float read_var
    float poisson_var


cdef struct AverageRampFit:
    float slope
    float read_var
    float poisson_var


cdef struct RampFits:
    vector[RampFit] fits
    vector[RampIndex] index
    AverageRampFit average


cdef struct DerivedData:
    vector[float] t_bar
    vector[float] tau
    vector[int] n_reads


cdef struct Thresh:
    float intercept
    float constant


cdef enum Diff:
    single = 0
    double = 1


cdef float threshold(Thresh thresh, float slope)
cdef float get_power(float s)
cdef deque[stack[RampIndex]] init_ramps(int[:, :] dq)
cdef DerivedData read_data(list[list[int]] read_pattern, float read_time)
