# cython: language_level=3str

from libcpp cimport bool
from libcpp.vector cimport vector

from stcal.ramp_fitting.ols_cas22._ramp cimport RampFit, RampQueue


cpdef enum FixedOffsets:
    single_t_bar_diff
    double_t_bar_diff
    single_t_bar_diff_sqr
    double_t_bar_diff_sqr
    single_read_recip
    double_read_recip
    single_var_slope_val
    double_var_slope_val
    n_fixed_offsets


cpdef enum PixelOffsets:
    single_local_slope
    double_local_slope
    single_var_read_noise
    double_var_read_noise
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


cpdef float[:, :] fill_fixed_values(float[:, :] fixed,
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
                        float[:, :] fixed,
                        float[:, :] pixel,
                        Thresh thresh,
                        bool use_jump,
                        bool include_diagnostic)
