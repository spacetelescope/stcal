"""This module exists to keep the interfaces the same as before a refactoring."""
import warnings

from .ols_cas22 import fit_ramps as fit_ramps_casertano

__all__ = ["fit_ramps_casertano"]

warnings.warn(
    "The module stcal.ramp_fitting.ols_cas22_fit is deprecated. "
    "Please use stcal.ramp_fitting.ols_cas22 instead.",
    DeprecationWarning,
    stacklevel=2,
)
