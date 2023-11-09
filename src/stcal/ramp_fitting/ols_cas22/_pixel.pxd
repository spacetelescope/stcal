cpdef enum PixelOffsets:
    single_local_slope
    double_local_slope
    single_var_read_noise
    double_var_read_noise
    n_pixel_offsets

cpdef float[:, :] fill_pixel_values(float[:, :] pixel,
                                    float[:] resultants,
                                    float[:, :] fixed,
                                    float read_noise,
                                    int n_resultants)