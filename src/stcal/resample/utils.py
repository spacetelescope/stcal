import numpy as np
from astropy.nddata.bitmask import interpret_bit_flags

__all__ = [
    "build_mask", "get_tmeasure",
]


def build_mask(dqarr, bitvalue, flag_name_map=None):
    """Build a bit mask from an input DQ array and a bitvalue flag

    In the returned bit mask, 1 is good, 0 is bad
    """
    bitvalue = interpret_bit_flags(bitvalue, flag_name_map=flag_name_map)

    if bitvalue is None:
        return np.ones(dqarr.shape, dtype=np.uint8)
    return np.logical_not(np.bitwise_and(dqarr, ~bitvalue)).astype(np.uint8)


def get_tmeasure(model):
    """
    Check if the measurement_time keyword is present in the datamodel
    for use in exptime weighting. If not, revert to using exposure_time.

    Returns a tuple of (exptime, is_measurement_time)
    """
    try:
        tmeasure = model.meta.exposure.measurement_time
    except AttributeError:
        return model.meta.exposure.exposure_time, False
    if tmeasure is None:
        return model.meta.exposure.exposure_time, False
    else:
        return tmeasure, True
