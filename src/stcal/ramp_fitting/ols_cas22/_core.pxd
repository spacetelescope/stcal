from libcpp.vector cimport vector


cdef struct RampIndex:
    int start
    int end


cdef struct RampFit:
    float slope
    float read_var
    float poisson_var


cdef struct RampFits:
    RampFit average
    vector[int] jumps
    vector[RampFit] fits
    vector[RampIndex] index


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


cpdef enum RampJumpDQ:
    JUMP_DET = 4


cpdef float threshold(Thresh thresh, float slope)
cdef float get_power(float s)
cpdef vector[RampIndex] init_ramps(int[:, :] dq, int n_resultants, int index_pixel)
cpdef ReadPatternMetadata metadata_from_read_pattern(list[list[int]] read_pattern, float read_time)
