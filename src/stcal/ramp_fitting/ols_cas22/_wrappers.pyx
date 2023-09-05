import numpy as np
cimport numpy as np

from libcpp cimport bool
from libcpp.stack cimport stack
from libcpp.deque cimport deque

from stcal.ramp_fitting.ols_cas22._core cimport RampIndex, DerivedData, Thresh, RampFit
from stcal.ramp_fitting.ols_cas22._core cimport read_data as c_read_data
from stcal.ramp_fitting.ols_cas22._core cimport init_ramps as c_init_ramps
from stcal.ramp_fitting.ols_cas22._core cimport make_threshold as c_make_threshold

from stcal.ramp_fitting.ols_cas22._fixed cimport Fixed
from stcal.ramp_fitting.ols_cas22._fixed cimport make_fixed as c_make_fixed

from stcal.ramp_fitting.ols_cas22._pixel cimport Pixel
from stcal.ramp_fitting.ols_cas22._pixel cimport make_pixel as c_make_pixel


def read_data(list[list[int]] read_pattern, float read_time):
    return c_read_data(read_pattern, read_time)


def init_ramps(np.ndarray[int, ndim=2] dq):
    cdef deque[stack[RampIndex]] raw = c_init_ramps(dq)

    # Have to turn deque and stack into python compatible objects
    cdef RampIndex index
    cdef stack[RampIndex] ramp
    cdef list out = []
    cdef list stack_out
    for ramp in raw:
        stack_out = []
        while not ramp.empty():
            index = ramp.top()
            ramp.pop()
            # So top of stack is first item of list
            stack_out = [index] + stack_out

        out.append(stack_out)

    return out


def make_threshold(float intercept, float constant):
    return c_make_threshold(intercept, constant)


def run_threshold(Thresh threshold, float slope):
    return threshold.run(slope)


def make_fixed(np.ndarray[float, ndim=1] t_bar,
               np.ndarray[float, ndim=1] tau,
               np.ndarray[int, ndim=1] n_reads,
               float intercept,
               float constant,
               bool use_jump):

    cdef DerivedData data = DerivedData(t_bar, tau, n_reads)
    cdef Thresh threshold = c_make_threshold(intercept, constant)

    cdef Fixed fixed = c_make_fixed(data, threshold, use_jump)

    cdef float intercept_ = fixed.threshold.intercept
    cdef float constant_ = fixed.threshold.constant

    cdef np.ndarray[float, ndim=1] t_bar_1, t_bar_2
    cdef np.ndarray[float, ndim=1] t_bar_1_sq, t_bar_2_sq
    cdef np.ndarray[float, ndim=1] recip_1, recip_2
    cdef np.ndarray[float, ndim=1] slope_var_1, slope_var_2

    if use_jump:
        t_bar_1 = np.array(fixed.t_bar_1, dtype=np.float32)
        t_bar_2 = np.array(fixed.t_bar_2, dtype=np.float32)
        t_bar_1_sq = np.array(fixed.t_bar_1_sq, dtype=np.float32)
        t_bar_2_sq = np.array(fixed.t_bar_2_sq, dtype=np.float32)

        recip_1 = np.array(fixed.recip_1, dtype=np.float32)
        recip_2 = np.array(fixed.recip_2, dtype=np.float32)

        slope_var_1 = np.array(fixed.slope_var_1, dtype=np.float32)
        slope_var_2 = np.array(fixed.slope_var_2, dtype=np.float32)
    else:
        try:
            fixed.t_bar_1
        except AttributeError:
            t_bar_1 = np.zeros(1, np.float32)
        else:
            raise AttributeError("t_bar_1 should not exist")

        try:
            fixed.t_bar_2
        except AttributeError:
            t_bar_2 = np.zeros(1, np.float32)
        else:
            raise AttributeError("t_bar_2 should not exist")

        try:
            fixed.t_bar_1_sq
        except AttributeError:
            t_bar_1_sq = np.zeros(1, np.float32)
        else:
            raise AttributeError("t_bar_1_sq should not exist")

        try:
            fixed.t_bar_2_sq
        except AttributeError:
            t_bar_2_sq = np.zeros(1, np.float32)
        else:
            raise AttributeError("t_bar_2_sq should not exist")

        try:
            fixed.recip_1
        except AttributeError:
            recip_1 = np.zeros(1, np.float32)
        else:
            raise AttributeError("recip_1 should not exist")

        try:
            fixed.recip_2
        except AttributeError:
            recip_2 = np.zeros(1, np.float32)
        else:
            raise AttributeError("recip_2 should not exist")

        try:
            fixed.slope_var_1
        except AttributeError:
            slope_var_1 = np.zeros(1, np.float32)
        else:
            raise AttributeError("slope_var_1 should not exist")

        try:
            fixed.slope_var_2
        except AttributeError:
            slope_var_2 = np.zeros(1, np.float32)
        else:
            raise AttributeError("slope_var_2 should not exist")


    return dict(data=fixed.data,
                intercept=intercept_,
                constant=constant_,
                t_bar_1=t_bar_1,
                t_bar_2=t_bar_2,
                t_bar_1_sq=t_bar_1_sq,
                t_bar_2_sq=t_bar_2_sq,
                recip_1=recip_1,
                recip_2=recip_2,
                slope_var_1=slope_var_1,
                slope_var_2=slope_var_2)


def make_pixel(np.ndarray[float, ndim=1] resultants,
               np.ndarray[float, ndim=1] t_bar,
               np.ndarray[float, ndim=1] tau,
               np.ndarray[int, ndim=1] n_reads,
               float read_noise,
               float intercept,
               float constant,
               bool use_jump):

    cdef DerivedData data = DerivedData(t_bar, tau, n_reads)
    cdef Thresh threshold = c_make_threshold(intercept, constant)

    cdef Fixed fixed = c_make_fixed(data, threshold, use_jump)

    cdef Pixel pixel = c_make_pixel(fixed, read_noise, resultants)

    cdef np.ndarray[float, ndim=1] resultants_ = np.array(pixel.resultants, dtype=np.float32)

    cdef np.ndarray[float, ndim=1] delta_1, delta_2
    cdef np.ndarray[float, ndim=1] sigma_1, sigma_2

    if use_jump:
        delta_1 = np.array(pixel.delta_1, dtype=np.float32)
        delta_2 = np.array(pixel.delta_2, dtype=np.float32)
        sigma_1 = np.array(pixel.sigma_1, dtype=np.float32)
        sigma_2 = np.array(pixel.sigma_2, dtype=np.float32)
    else:
        try:
            pixel.delta_1
        except AttributeError:
            delta_1 = np.zeros(1, np.float32)
        else:
            raise AttributeError("delta_1 should not exist")

        try:
            pixel.delta_2
        except AttributeError:
            delta_2 = np.zeros(1, np.float32)
        else:
            raise AttributeError("delta_2 should not exist")

        try:
            pixel.sigma_1
        except AttributeError:
            sigma_1 = np.zeros(1, np.float32)
        else:
            raise AttributeError("sigma_1 should not exist")

        try:
            pixel.sigma_2
        except AttributeError:
            sigma_2 = np.zeros(1, np.float32)
        else:
            raise AttributeError("sigma_2 should not exist")

    # only return computed values (assume fixed is correct)
    return dict(resultants=resultants_,
                read_noise=pixel.read_noise,
                delta_1=delta_1,
                delta_2=delta_2,
                sigma_1=sigma_1,
                sigma_2=sigma_2)


def fit_ramp(np.ndarray[float, ndim=1] resultants,
             np.ndarray[float, ndim=1] t_bar,
             np.ndarray[float, ndim=1] tau,
             np.ndarray[int, ndim=1] n_reads,
             float read_noise,
             int start,
             int end):

    cdef DerivedData data = DerivedData(t_bar, tau, n_reads)
    cdef Thresh threshold = c_make_threshold(0, 1)
    cdef Fixed fixed = c_make_fixed(data, threshold, False)

    cdef Pixel pixel = c_make_pixel(fixed, read_noise, resultants)
    cdef RampIndex ramp_index = RampIndex(start, end)

    cdef RampFit ramp_fit = pixel.fit_ramp(ramp_index)

    return ramp_fit
