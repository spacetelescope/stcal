from copy import deepcopy
import asdf

import numpy as np
from astropy.nddata.bitmask import interpret_bit_flags

__all__ = [
    "build_mask", "get_tmeasure", "bytes2human", "load_custom_wcs"
]


def load_custom_wcs(asdf_wcs_file, output_shape=None):
    """ Load a custom output WCS from an ASDF file.

    Parameters
    ----------
    asdf_wcs_file : str
        Path to an ASDF file containing a GWCS structure.

    output_shape : tuple of int, optional
        Array shape (in ``[x, y]`` order) for the output data. If not provided,
        the custom WCS must specify one of: pixel_shape,
        array_shape, or bounding_box.

    Returns
    -------
    wcs : WCS
        The output WCS to resample into.

    """
    if not asdf_wcs_file:
        return None

    with asdf.open(asdf_wcs_file) as af:
        wcs = deepcopy(af.tree["wcs"])
        wcs.pixel_area = af.tree.get("pixel_area", None)
        wcs.pixel_shape = af.tree.get("pixel_shape", None)
        wcs.array_shape = af.tree.get("array_shape", None)

    if output_shape is not None:
        wcs.array_shape = output_shape[::-1]
        wcs.pixel_shape = output_shape
    elif wcs.pixel_shape is not None:
        wcs.array_shape = wcs.pixel_shape[::-1]
    elif wcs.array_shape is not None:
        wcs.pixel_shape = wcs.array_shape[::-1]
    elif wcs.bounding_box is not None:
        wcs.array_shape = tuple(
            int(axs[1] + 0.5)
            for axs in wcs.bounding_box.bounding_box(order="C")
        )
    else:
        raise ValueError(
            "Step argument 'output_shape' is required when custom WCS "
            "does not have neither of 'array_shape', 'pixel_shape', or "
            "'bounding_box' attributes set."
        )

    return wcs


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