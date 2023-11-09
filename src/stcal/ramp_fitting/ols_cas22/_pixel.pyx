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

from stcal.ramp_fitting.ols_cas22._core cimport Diff
from stcal.ramp_fitting.ols_cas22._pixel cimport PixelOffsets


@boundscheck(False)
@wraparound(False)
@cdivision(True)
cpdef inline float[:, :] fill_pixel_values(float[:, :] pixel,
                                           float[:] resultants,
                                           float[:, :] t_bar_diffs,
                                           float[:, :] read_recip_coeffs,
                                           float read_noise,
                                           int n_resultants):
    """
    Compute the local slopes between resultants for the pixel

    Returns
    -------
    [
        <(resultants[i+1] - resultants[i])> / <(t_bar[i+1] - t_bar[i])>,
        <(resultants[i+2] - resultants[i])> / <(t_bar[i+2] - t_bar[i])>,
    ]
    """
    cdef int single = Diff.single
    cdef int double = Diff.double

    cdef int single_slope = PixelOffsets.single_local_slope
    cdef int double_slope = PixelOffsets.double_local_slope
    cdef int single_var = PixelOffsets.single_var_read_noise
    cdef int double_var = PixelOffsets.double_var_read_noise

    # cdef float[:, :] pixel = np.empty((n_offsets, n_resultants - 1), dtype=np.float32)
    cdef float read_noise_sqr = read_noise ** 2

    cdef int i
    for i in range(n_resultants - 1):
        pixel[single_slope, i] = (resultants[i + 1] - resultants[i]) / t_bar_diffs[single, i]

        if i < n_resultants - 2:
            pixel[double_slope, i] = (resultants[i + 2] - resultants[i]) / t_bar_diffs[double, i]
        else:
            pixel[double_slope, i] = NAN  # last double difference is undefined

        pixel[single_var, i] = read_noise_sqr * read_recip_coeffs[single, i]
        pixel[double_var, i] = read_noise_sqr * read_recip_coeffs[double, i]

    return pixel
