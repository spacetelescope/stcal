from libcpp.vector cimport vector
from stcal.ramp_fitting.ols_cas22._ramp cimport RampQueue


cdef struct RampFit:
    float slope
    float read_var
    float poisson_var


cdef struct RampFits:
    RampFit average
    vector[int] jumps
    vector[RampFit] fits
    RampQueue index


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


cpdef enum RampJumpDQ:
    JUMP_DET = 4


cdef float get_power(float s)
