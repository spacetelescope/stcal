cdef struct Thresh:
    float intercept
    float constant


cpdef float threshold(Thresh thresh, float slope)