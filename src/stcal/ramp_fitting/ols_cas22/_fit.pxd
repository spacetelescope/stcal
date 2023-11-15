# cython: language_level=3str


cpdef enum Parameter:
    intercept
    slope
    n_param


cpdef enum Variance:
    read_var
    poisson_var
    total_var
    n_var


cpdef enum:
    JUMP_DET = 4


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
