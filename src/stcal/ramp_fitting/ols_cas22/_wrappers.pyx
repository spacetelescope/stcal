import numpy as np
cimport numpy as np

from libcpp cimport bool

from stcal.ramp_fitting.ols_cas22._core cimport ReadPatternMetadata, Thresh

from stcal.ramp_fitting.ols_cas22._fixed cimport FixedValues
from stcal.ramp_fitting.ols_cas22._fixed cimport fixed_values_from_metadata as c_fixed_values_from_metadata

from stcal.ramp_fitting.ols_cas22._pixel cimport Pixel
from stcal.ramp_fitting.ols_cas22._pixel cimport make_pixel as c_make_pixel


def fixed_values_from_metadata(np.ndarray[float, ndim=1] t_bar,
                               np.ndarray[float, ndim=1] tau,
                               np.ndarray[int, ndim=1] n_reads,
                               float intercept,
                               float constant,
                               bool use_jump):

    cdef ReadPatternMetadata data = ReadPatternMetadata(t_bar, tau, n_reads)
    cdef Thresh threshold = Thresh(intercept, constant)

    cdef FixedValues fixed = c_fixed_values_from_metadata(data, threshold, use_jump)

    cdef float intercept_ = fixed.threshold.intercept
    cdef float constant_ = fixed.threshold.constant

    cdef np.ndarray[float, ndim=2] t_bar_diffs
    cdef np.ndarray[float, ndim=2] t_bar_diff_sqrs
    cdef np.ndarray[float, ndim=2] read_recip_coeffs
    cdef np.ndarray[float, ndim=2] var_slope_coeffs

    if use_jump:
        t_bar_diffs = np.array(fixed.t_bar_diffs, dtype=np.float32)
        t_bar_diff_sqrs = np.array(fixed.t_bar_diff_sqrs, dtype=np.float32)
        read_recip_coeffs = np.array(fixed.read_recip_coeffs, dtype=np.float32)
        var_slope_coeffs = np.array(fixed.var_slope_coeffs, dtype=np.float32)
    else:
        try:
            fixed.t_bar_diffs
        except AttributeError:
            t_bar_diffs = np.array([[np.nan],[np.nan]], dtype=np.float32)
        else:
            raise AttributeError("t_bar_diffs should not exist")

        try:
            fixed.t_bar_diff_sqrs
        except AttributeError:
            t_bar_diff_sqrs = np.array([[np.nan],[np.nan]], dtype=np.float32)
        else:
            raise AttributeError("t_bar_diff_sqrs should not exist")

        try:
            fixed.read_recip_coeffs
        except AttributeError:
            read_recip_coeffs = np.array([[np.nan],[np.nan]], dtype=np.float32)
        else:
            raise AttributeError("read_recip_coeffs should not exist")

        try:
            fixed.var_slope_coeffs
        except AttributeError:
            var_slope_coeffs = np.array([[np.nan],[np.nan]], dtype=np.float32)
        else:
            raise AttributeError("var_slope_coeffs should not exist")

    return dict(data=fixed.data,
                intercept=intercept_,
                constant=constant_,
                t_bar_diffs=t_bar_diffs,
                t_bar_diff_sqrs=t_bar_diff_sqrs,
                read_recip_coeffs=read_recip_coeffs,
                var_slope_coeffs=var_slope_coeffs)


def make_pixel(np.ndarray[float, ndim=1] resultants,
               np.ndarray[float, ndim=1] t_bar,
               np.ndarray[float, ndim=1] tau,
               np.ndarray[int, ndim=1] n_reads,
               float read_noise,
               float intercept,
               float constant,
               bool use_jump):

    cdef ReadPatternMetadata data = ReadPatternMetadata(t_bar, tau, n_reads)
    cdef Thresh threshold = Thresh(intercept, constant)

    cdef FixedValues fixed = c_fixed_values_from_metadata(data, threshold, use_jump)

    cdef Pixel pixel = c_make_pixel(fixed, read_noise, resultants)

    cdef np.ndarray[float, ndim=1] resultants_ = np.array(pixel.resultants, dtype=np.float32)

    cdef np.ndarray[float, ndim=2] local_slopes
    cdef np.ndarray[float, ndim=2] var_read_noise

    if use_jump:
        local_slopes = np.array(pixel.local_slopes, dtype=np.float32)
        var_read_noise = np.array(pixel.var_read_noise, dtype=np.float32)
    else:
        try:
            pixel.local_slopes
        except AttributeError:
            local_slopes = np.array([[np.nan],[np.nan]], dtype=np.float32)
        else:
            raise AttributeError("local_slopes should not exist")

        try:
            pixel.var_read_noise
        except AttributeError:
            var_read_noise = np.array([[np.nan],[np.nan]], dtype=np.float32)
        else:
            raise AttributeError("var_read_noise should not exist")

    # only return computed values (assume fixed is correct)
    return dict(resultants=resultants_,
                read_noise=pixel.read_noise,
                local_slopes=local_slopes,
                var_read_noise=var_read_noise)
