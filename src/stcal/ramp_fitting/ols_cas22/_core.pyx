from libc.math cimport sqrt, fabs, log10
from libcpp.vector cimport vector
from libcpp.stack cimport stack
from libcpp cimport bool
import numpy as np
cimport numpy as np
cimport cython

from stcal.ramp_fitting.ols_cas22._core cimport RampIndex, Thresh, Fit, Fits


# Casertano+2022, Table 2
cdef float[2][6] PTABLE = [
    [-np.inf, 5, 10, 20, 50, 100],
    [0,     0.4,  1,  3,  6,  10]]
cdef int PTABLE_LENGTH = 6

cdef inline float get_weight_power(float s):
    cdef int i
    for i in range(PTABLE_LENGTH):
        if s < PTABLE[0][i]:
            return PTABLE[1][i - 1]
    return PTABLE[1][i]


cdef float threshold(Thresh thresh, float slope):
    return thresh.intercept - thresh.constant * log10(slope)


cdef Fits reverse_fits(Fits fits):
    return Fits(fits.slope[::-1], fits.read_var[::-1], fits.poisson_var[::-1])
