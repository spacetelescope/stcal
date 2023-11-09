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


cpdef float[:, :] fill_fixed_values(float[:, :] fixed,
                                    float[:] t_bar,
                                    float[:] tau,
                                    int[:] n_reads,
                                    int n_resultants)
