from enum import Enum

import numpy as np

from ._fit import fit_ramps
from ._jump import JUMP_DET, Parameter, Variance


class DefaultThreshold(Enum):
    INTERCEPT = np.float32(5.5)
    CONSTANT = np.float32(1 / 3)


__all__ = ["fit_ramps", "Parameter", "Variance", "Diff", "JUMP_DET", "DefaultThreshold"]
