# cython: language_level=3str

from libcpp cimport bool
from libcpp.vector cimport vector

from stcal.ramp_fitting.ols_cas22._ramp cimport RampFit, RampQueue


cpdef enum FixedOffsets:
    t_bar_diff
    t_bar_diff_sqr
    read_recip
    var_slope_val
    n_fixed_offsets


cpdef enum PixelOffsets:
    local_slope
    var_read_noise
    n_pixel_offsets


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


cpdef void _fill_fixed_values(float[:, :] single_fixed,
                              float[:, :] double_fixed,
                              float[:] t_bar,
                              float[:] tau,
                              int[:] n_reads,
                              int n_resultants)


cdef JumpFits fit_jumps(float[:] resultants,
                        int[:] dq,
                        float read_noise,
                        float[:] t_bar,
                        float[:] tau,
                        int[:] n_reads,
                        int n_resultants,
                        float[:, :] single_pixel,
                        float[:, :] double_pixel,
                        float[:, :] single_fixed,
                        float[:, :] double_fixed,
                        Thresh thresh,
                        bool use_jump,
                        bool include_diagnostic)
