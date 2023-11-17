"""
This subpackage exists to hold the Cython implementation of the OLS cas22 algorithm
    This subpackage is private, and should not be imported directly by users. Instead,
    import from stcal.ramp_fitting.ols_cas22.
"""
from ._fit import FixedOffsets, Parameter, PixelOffsets, Variance, fit_ramps

__all__ = [
    "fit_ramps",
    "Parameter",
    "Variance",
    "PixelOffsets",
    "FixedOffsets",
]
