import os
from pathlib import Path, PurePath

import numpy as np
from astropy.nddata.bitmask import interpret_bit_flags

__all__ = [
    "build_mask", "build_output_model_name", "get_tmeasure", "bytes2human"
]


def build_output_model_name(input_filename_list):
    fnames = {f for f in input_filename_list if f is not None}

    if not fnames:
        return "resampled_data_{resample_suffix}{resample_file_ext}"

    # TODO: maybe remove ending suffix for single file names?
    prefix = os.path.commonprefix(
        [PurePath(f).stem.strip('_- ') for f in fnames]
    )

    return prefix + "{resample_suffix}{resample_file_ext}"


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
        tmeasure = model["measurement_time"]
    except KeyError:
        return model["exposure_time"], False
    if tmeasure is None:
        return model["exposure_time"], False
    else:
        return tmeasure, True


# FIXME: temporarily copied here to avoid this import:
# from stdatamodels.jwst.library.basic_utils import bytes2human
def bytes2human(n):
    """Convert bytes to human-readable format

    Taken from the `psutil` library which references
    http://code.activestate.com/recipes/578019

    Parameters
    ----------
    n : int
        Number to convert

    Returns
    -------
    readable : str
        A string with units attached.

    Examples
    --------
    >>> bytes2human(10000)
        '9.8K'

    >>> bytes2human(100001221)
        '95.4M'
    """
    symbols = ('K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y')
    prefix = {}
    for i, s in enumerate(symbols):
        prefix[s] = 1 << (i + 1) * 10
    for s in reversed(symbols):
        if n >= prefix[s]:
            value = float(n) / prefix[s]
            return '%.1f%s' % (value, s)
    return "%sB" % n