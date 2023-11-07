import numpy as np

from libc.math cimport log10


from stcal.ramp_fitting.ols_cas22._jump cimport Thresh

cpdef inline float threshold(Thresh thresh, float slope):
    """
    Compute jump threshold

    Parameters
    ----------
    thresh : Thresh
        threshold parameters struct
    slope : float
        slope of the ramp in question

    Returns
    -------
        intercept - constant * log10(slope)
    """
    slope = slope if slope > 1 else 1
    slope = slope if slope < 1e4 else 1e4

    return thresh.intercept - thresh.constant * log10(slope)