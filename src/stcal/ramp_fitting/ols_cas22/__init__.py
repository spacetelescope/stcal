from enum import Enum

import numpy as np

from ._fit import JUMP_DET, Parameter, Variance, fit_ramps


class DefaultThreshold(Enum):
    INTERCEPT = np.float32(5.5)
    CONSTANT = np.float32(1 / 3)


__all__ = ["fit_ramps", "Parameter", "Variance", "Diff", "JUMP_DET", "DefaultThreshold"]
