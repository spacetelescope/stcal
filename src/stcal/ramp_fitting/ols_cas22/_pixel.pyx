"""
Define the C class for the Cassertano22 algorithm for fitting ramps with jump detection

Objects
-------
Pixel : class
    Class to handle ramp fit with jump detection for a single pixel
    Provides fits method which fits all the ramps for a single pixel

Functions
---------
    make_pixel : function
        Fast constructor for a Pixel class from input data.
            - cpdef gives a python wrapper, but the python version of this method
              is considered private, only to be used for testing
"""
from libc.math cimport NAN
from cython cimport boundscheck, wraparound, cdivision

from stcal.ramp_fitting.ols_cas22._fixed cimport FixedOffsets
from stcal.ramp_fitting.ols_cas22._pixel cimport PixelOffsets


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef inline float[:, :] fill_pixel_values(float[:, :] pixel,
                                           float[:] resultants,
                                           float[:, :] fixed,
                                           float read_noise,
                                           int n_resultants):
    """
    Compute the local slopes between resultants for the pixel

    Returns
    -------
    [
        <(resultants[i+1] - resultants[i])> / <(t_bar[i+1] - t_bar[i])>,
        <(resultants[i+2] - resultants[i])> / <(t_bar[i+2] - t_bar[i])>,
        read_noise ** 2 / <(t_bar[i+1] - t_bar[i])>,
        read_noise ** 2 / <(t_bar[i+2] - t_bar[i])>,
    ]
    """
    cdef int single_slope = PixelOffsets.single_local_slope
    cdef int double_slope = PixelOffsets.double_local_slope
    cdef int single_var = PixelOffsets.single_var_read_noise
    cdef int double_var = PixelOffsets.double_var_read_noise

    cdef int single_t_bar_diff = FixedOffsets.single_t_bar_diff
    cdef int double_t_bar_diff = FixedOffsets.double_t_bar_diff
    cdef int single_read_recip = FixedOffsets.single_read_recip
    cdef int double_read_recip = FixedOffsets.double_read_recip

    cdef float read_noise_sqr = read_noise ** 2

    cdef int i
    for i in range(n_resultants - 1):
        pixel[single_slope, i] = (resultants[i + 1] - resultants[i]) / fixed[single_t_bar_diff, i]
        pixel[single_var, i] = read_noise_sqr * fixed[single_read_recip, i]

        if i < n_resultants - 2:
            pixel[double_slope, i] = (resultants[i + 2] - resultants[i]) / fixed[double_t_bar_diff, i]
            pixel[double_var, i] = read_noise_sqr * fixed[double_read_recip, i]
        else:
            # The last double difference is undefined
            pixel[double_slope, i] = NAN
            pixel[double_var, i] = NAN

    return pixel
