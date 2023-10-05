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


cdef struct RampFits:
    vector[RampFit] fits
    vector[RampIndex] index
    vector[int] jumps
    RampFit average


cdef struct ReadPatternMetadata:
    vector[float] t_bar
    vector[float] tau
    vector[int] n_reads


cdef struct Thresh:
    float intercept
    float constant


cpdef enum Diff:
    single = 0
    double = 1


cpdef enum Parameter:
    intercept = 0
    slope = 1


cpdef enum Variance:
    read_var = 0
    poisson_var = 1
    total_var = 2


cdef float threshold(Thresh thresh, float slope)
cdef float get_power(float s)
cdef deque[stack[RampIndex]] init_ramps(int[:, :] dq)
cdef ReadPatternMetadata metadata_from_read_pattern(list[list[int]] read_pattern, float read_time)
