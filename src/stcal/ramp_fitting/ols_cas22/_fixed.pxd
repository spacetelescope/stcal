from libcpp cimport bool

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