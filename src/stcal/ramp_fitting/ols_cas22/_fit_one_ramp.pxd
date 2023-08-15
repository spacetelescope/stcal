from stcal.ramp_fitting.ols_cas22._core cimport Ramp

cdef (float, float, float) fit_one_ramp(Ramp ramp)