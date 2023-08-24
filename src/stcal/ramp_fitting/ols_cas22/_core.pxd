import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.stack cimport stack
from libcpp cimport bool


cdef struct RampIndex:
    int start
    int end


cdef struct Thresh:
    float intercept
    float constant


cdef struct Fit:
    float slope
    float read_var
    float poisson_var

cdef class Fixed:
    # Fixed parameters for all pixels input
    cdef public bool use_jump
    cdef public float[:] t_bar, tau
    cdef public int[:] n_reads

    # Computed and cached values for jump detection
    #    single -> j = i + 1
    #    double -> j = i + 2

    # single and double differences of t_bar
    #    t_bar[j] - t_bar[i]
    cdef public float[:] t_bar_1, t_bar_2

    # squared single and double differences of t_bar
    #     (t_bar[j] - t_bar[i])**2
    cdef public float[:] t_bar_1_sq, t_bar_2_sq

    # single and double reciprical sum values
    #    ((1/n_reads[i]) + (1/n_reads[j]))
    cdef public float[:] recip_1, recip_2

    # single and double slope var terms
    #    (tau[i] + tau[j] - min(t_bar[i], t_bar[j])) * correction(i, j)
    cdef public float[:] slope_var_1, slope_var_2

    cdef float[:] t_bar_diff(Fixed self, int offset)
    cdef float[:] t_bar_diff_sq(Fixed self, int offset)
    cdef float[:] recip_val(Fixed self, int offset)
    cdef float[:] slope_var_val(Fixed self, int offset)

    cdef float correction(Fixed self, int i, int j)


cdef Fixed make_fixed(float[:] t_bar, float[:] tau, int[:] n_reads, bool use_jump)


cdef class Ramp:
    cdef Fixed fixed
    cdef public float read_noise
    cdef public float [:] resultants

    # Computed and cached values for jump detection
    #    single -> j = i + 1
    #    double -> j = i + 2

    # single and double delta + slope
    #    (resultants[j] - resultants[i]/(t_bar[j] - t_bar[i])
    cdef public float[:] delta_1, delta_2 

    # single and double sigma terms
    #    read_noise * recip[i]
    cdef public float[:] sigma_1, sigma_2

    cdef float[:] resultants_diff(Ramp self, int offset)
    cdef (float, float, float) fit(Ramp self, int start, int end)

    cdef float[:] stats(Ramp self, float slope, int start, int end)
    cdef (stack[float], stack[float], stack[float]) fits(Ramp self, stack[RampIndex] ramps, Thresh thresh)


cdef Ramp make_ramp(Fixed fixed, float read_noise, float [:] resultants)
