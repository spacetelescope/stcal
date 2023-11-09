# cython: language_level=3str

from libcpp cimport bool
from libcpp.vector cimport vector

from stcal.ramp_fitting.ols_cas22._ramp cimport RampFit, RampQueue


cpdef enum:
    JUMP_DET = 4

cdef struct Thresh:
    float intercept
    float constant


cdef struct JumpFits:
    RampFit average
    vector[int] jumps
    vector[RampFit] fits
    RampQueue index


cdef JumpFits fit_jumps(float[:] resultants,
                        int[:] dq,
                        float read_noise,
                        RampQueue ramps,
                        float[:] t_bar,
                        float[:] tau,
                        int[:] n_reads,
                        int n_resultants,
                        float[:, :] fixed,
                        float[:, :] pixel,
                        Thresh thresh,
                        bool use_jump,
                        bool include_diagnostic)
