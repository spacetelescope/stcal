cdef (float, float, float) fit_one_ramp(
        float [:] resultants, int start, int end, float read_noise,
        float [:] t_bar, float [:] tau, int [:] nn)