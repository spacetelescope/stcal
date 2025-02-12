import logging
import warnings

import numpy as np
from scipy.ndimage import median_filter
from astropy.nddata.bitmask import (
    bitfield_to_boolean_mask,
    interpret_bit_flags,
)
from astropy import units as u
from spherical_geometry.polygon import SphericalPolygon  # type: ignore[import-untyped]


__all__ = [
    "build_driz_weight",
    "build_mask",
    "compute_mean_pixel_area",
    "get_tmeasure",
    "is_flux_density",
    "is_imaging_wcs",
    "resample_range",
]

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def resample_range(data_shape, bbox=None):
    # Find range of input pixels to resample:
    if len(data_shape) != 2:
        raise ValueError("Expected 'data_shape' to be of length 2.")

    if bbox is None:
        xmin = ymin = 0
        xmax = max(data_shape[1] - 1, 0)
        ymax = max(data_shape[0] - 1, 0)

    else:
        if len(bbox) != 2:
            raise ValueError("Expected bounding box to be of length 2.")
        ((x1, x2), (y1, y2)) = bbox
        xmin = max(0, int(x1 + 0.5))
        ymin = max(0, int(y1 + 0.5))
        xmax = max(xmin, min(data_shape[1] - 1, int(x2 + 0.5)))
        ymax = max(ymin, min(data_shape[0] - 1, int(y2 + 0.5)))

    return xmin, xmax, ymin, ymax


def build_mask(dqarr, good_bits, flag_name_map=None):
    """Build a bit mask from an input DQ array and a bitvalue flag

    In the returned bit mask, 1 is good, 0 is bad
    """
    good_bits = interpret_bit_flags(good_bits, flag_name_map=flag_name_map)

    dqmask = bitfield_to_boolean_mask(
        dqarr,
        good_bits,
        good_mask_value=1,
        dtype=np.uint8,
        flag_name_map=flag_name_map,
    )
    return dqmask

    # if bitvalue is None:
    #     return np.ones(dqarr.shape, dtype=np.uint8)
    # return np.logical_not(np.bitwise_and(dqarr, ~bitvalue)).astype(np.uint8)


    # bitvalue = interpret_bit_flags(bitvalue, mnemonic_map=pixel)

    # if bitvalue is None:
    #     return np.ones(dqarr.shape, dtype=np.uint8)

    # bitvalue = np.array(bitvalue).astype(dqarr.dtype)
    # return np.logical_not(np.bitwise_and(dqarr, ~bitvalue)).astype(np.uint8)


def build_driz_weight(model, weight_type=None, good_bits=None,
                      flag_name_map=None):
    """ Create a weight map that is used for weighting input images when
    they are co-added to the ouput model.

    Parameters
    ----------
    model : dict
        Input model: a dictionary of relevant keywords and values.

    weight_type : {"exptime", "ivm"}, optional
        The weighting type for adding models' data. For
        ``weight_type="ivm"`` (the default), the weighting will be
        determined per-pixel using the inverse of the read noise
        (VAR_RNOISE) array stored in each input image. If the
        ``VAR_RNOISE`` array does not exist,
        the variance is set to 1 for all pixels (i.e., equal weighting).
        If ``weight_type="exptime"``, the weight will be set equal
        to the measurement time when available and to
        the exposure time otherwise.

    good_bits : int, str, None, optional
        An integer bit mask, `None`, a Python list of bit flags, a comma-,
        or ``'|'``-separated, ``'+'``-separated string list of integer
        bit flags or mnemonic flag names that indicate what bits in models'
        DQ bitfield array should be *ignored* (i.e., zeroed).

        See `Resample` for more information.

    flag_name_map : astropy.nddata.BitFlagNameMap, dict, None, optional
        A `~astropy.nddata.BitFlagNameMap` object or a dictionary that provides
        mapping from mnemonic bit flag names to integer bit values in order to
        translate mnemonic flags to numeric values when ``bit_flags``
        that are comma- or '+'-separated list of menmonic bit flag names.

    """
    data = model["data"]
    dq = model["dq"]

    dqmask = bitfield_to_boolean_mask(
        dq,
        good_bits,
        good_mask_value=1,
        dtype=np.uint8,
        flag_name_map=flag_name_map,
    )

    if weight_type == 'ivm':
        var_rnoise = model["var_rnoise"]
        if (var_rnoise is not None and
                var_rnoise.shape == data.shape):
            with np.errstate(divide="ignore", invalid="ignore"):
                inv_variance = var_rnoise**-1
            inv_variance[~np.isfinite(inv_variance)] = 1
        else:
            warnings.warn(
                "'var_rnoise' array not available. "
                "Setting drizzle weight map to 1",
                RuntimeWarning
            )
            inv_variance = 1.0
        result = inv_variance * dqmask

    elif weight_type == 'exptime':
        exptime, s = get_tmeasure(model)
        result = exptime * dqmask

    else:
        result = np.ones(data.shape, dtype=data.dtype) * dqmask

    return result.astype(np.float32)


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


def is_imaging_wcs(wcs):
    """ Returns `True` if ``wcs`` is an imaging WCS and `False` otherwise. """
    imaging = all(
        ax == 'SPATIAL' for ax in wcs.output_frame.axes_type
    )
    return imaging


def compute_mean_pixel_area(wcs, shape=None):
    """ Computes the average pixel area (in steradians) based on input WCS
    using pixels within either the bounding box (if available) or the entire
    data array as defined either by ``wcs.array_shape`` or the ``shape``
    argument.

    Parameters
    ----------
    shape : tuple, optional
        Shape of the region over which average pixel area will be computed.
        When not provided, pixel average will be estimated over a region
        defined by ``wcs.array_shape``.

    Returns
    -------
    pix_area : float
        Pixel area in steradians.

    Notes
    -----

    This function takes the outline of the region in which the average is
    computed (a rectangle defined by either the bounding box or
    ``wcs.array_shape`` or the ``shape``) and projects it to world coordinates.
    It then uses ``spherical_geometry`` to compute the area of the polygon
    defined by this outline on the sky. In order to minimize errors due to
    distortions in the ``wcs``, the code defines the outline using pixels
    spaced no more than 15 pixels apart along the border of the rectangle
    in which the average is computed.

    """
    if (shape := (shape or wcs.array_shape)) is None:
        raise ValueError(
            "Either WCS must have 'array_shape' attribute set or 'shape' "
            "argument must be supplied."
        )

    valid_polygon = False
    spatial_idx = np.where(
        np.array(wcs.output_frame.axes_type) == 'SPATIAL'
    )[0]

    ny, nx = shape

    if wcs.bounding_box is None:
        ((xmin, xmax), (ymin, ymax)) = ((-0.5, nx - 0.5), (-0.5, ny - 0.5))
    else:
        ((xmin, xmax), (ymin, ymax)) = wcs.bounding_box

    if xmin > xmax:
        (xmin, xmax) = (xmax, xmin)
    if ymin > ymax:
        (ymin, ymax) = (ymax, ymin)

    xmin = max(0, int(xmin + 0.5))
    xmax = min(nx - 1, int(xmax - 0.5))
    ymin = max(0, int(ymin + 0.5))
    ymax = min(ny - 1, int(ymax - 0.5))

    k = 0
    dxy = [1, -1, -1, 1]

    while xmin < xmax and ymin < ymax:
        try:
            (x, y, image_area, center, b, r, t, l) = _get_boundary_points(
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
                dx=min((xmax - xmin) // 4, 15),
                dy=min((ymax - ymin) // 4, 15)
            )
        except ValueError:
            return None

        world = wcs(x, y)
        ra = world[spatial_idx[0]]
        dec = world[spatial_idx[1]]

        limits = [ymin, xmax, ymax, xmin]

        for j in range(4):
            sl = [b, r, t, l][k]
            if not (np.all(np.isfinite(ra[sl])) and
                    np.all(np.isfinite(dec[sl]))):
                limits[k] += dxy[k]
                ymin, xmax, ymax, xmin = limits
                k = (k + 1) % 4
                break
            k = (k + 1) % 4
        else:
            valid_polygon = True
            break

        ymin, xmax, ymax, xmin = limits

    if not valid_polygon:
        return None

    world = wcs(*center)
    wcenter = (world[spatial_idx[0]], world[spatial_idx[1]])

    sky_area = SphericalPolygon.from_radec(ra, dec, center=wcenter).area()
    if sky_area > 2 * np.pi:
        log.warning(
            "Unexpectedly large computed sky area for an image. "
            "Setting area to: 4*Pi - area"
        )
        sky_area = 4 * np.pi - sky_area
    pix_area = sky_area / image_area

    return pix_area


def _get_boundary_points(xmin, xmax, ymin, ymax, dx=None, dy=None,
                         shrink=0):  # noqa: E741
    """
    Creates a list of ``x`` and ``y`` coordinates of points along the perimiter
    of the rectangle defined by ``xmin``, ``xmax``, ``ymin``, ``ymax``, and
    ``shrink`` in counter-clockwise order.

    Parameters
    ----------

    xmin : int
        X-coordinate of the left edge of a rectangle.

    xmax : int
        X-coordinate of the right edge of a rectangle.

    ymin : int
        Y-coordinate of the bottom edge of a rectangle.

    ymax : int
        Y-coordinate of the top edge of a rectangle.

    dx : int, float, None, optional
        Desired spacing between ajacent points alog horizontal edges of
        the rectangle.

    dy : int, float, None, optional
        Desired spacing between ajacent points alog vertical edges of
        the rectangle.

    shrink : int, optional
        Amount to be applied to input ``xmin``, ``xmax``, ``ymin``, ``ymax``
        to reduce the rectangle size.

    Returns
    -------

    x : numpy.ndarray
        An array of X-coordinates of points along the perimiter
        of the rectangle defined by ``xmin``, ``xmax``, ``ymin``, ``ymax``, and
        ``shrink`` in counter-clockwise order.

    y : numpy.ndarray
        An array of Y-coordinates of points along the perimiter
        of the rectangle defined by ``xmin``, ``xmax``, ``ymin``, ``ymax``, and
        ``shrink`` in counter-clockwise order.

    area : float
        Area in units of pixels of the region defined by ``xmin``, ``xmax``,
        ``ymin``, ``ymax``, and ``shrink``.

    center : tuple
        A tuple of pixel coordinates at the center of the rectangle defined
        by ``xmin``, ``xmax``, ``ymin``, ``ymax``.

    bottom : slice
        A `slice` object that allows selection of pixels from ``x`` and ``y``
        arrays along the bottom edge of the rectangle.

    right : slice
        A `slice` object that allows selection of pixels from ``x`` and ``y``
        arrays along the right edge of the rectangle.

    top : slice
        A `slice` object that allows selection of pixels from ``x`` and ``y``
        arrays along the top edge of the rectangle.

    left : slice
        A `slice` object that allows selection of pixels from ``x`` and ``y``
        arrays along the left edge of the rectangle.

    """
    nx = xmax - xmin + 1
    ny = ymax - ymin + 1

    if dx is None:
        dx = nx
    if dy is None:
        dy = ny

    if nx - 2 * shrink < 1 or ny - 2 * shrink < 1:
        raise ValueError("Image size is too small.")

    sx = max(1, int(np.ceil(nx / dx)))
    sy = max(1, int(np.ceil(ny / dy)))

    xmin += shrink
    xmax -= shrink
    ymin += shrink
    ymax -= shrink

    size = 2 * sx + 2 * sy
    x = np.empty(size)
    y = np.empty(size)

    bottom = np.s_[0:sx]  # bottom edge
    right = np.s_[sx:sx + sy]  # right edge
    top = np.s_[sx + sy:2 * sx + sy]  # top edge
    left = np.s_[2 * sx + sy:2 * sx + 2 * sy]  # noqa: E741  left edge

    x[bottom] = np.linspace(xmin, xmax, sx, False)
    y[bottom] = ymin
    x[right] = xmax
    y[right] = np.linspace(ymin, ymax, sy, False)
    x[top] = np.linspace(xmax, xmin, sx, False)
    y[top] = ymax
    x[left] = xmin
    y[left] = np.linspace(ymax, ymin, sy, False)

    area = (xmax - xmin) * (ymax - ymin)
    center = (0.5 * (xmin + xmax), 0.5 * (ymin + ymax))

    return x, y, area, center, bottom, right, top, left


def is_flux_density(bunit):
    """
    Differentiate between surface brightness and flux density data units.

    Parameters
    ----------
    bunit : str or `~astropy.units.Unit`
       Data units, e.g. 'MJy' (is flux density) or 'MJy/sr' (is not).

    Returns
    -------
    bool
        True if the units are equivalent to flux density units.
    """
    try:
        flux_density = u.Unit(bunit).is_equivalent(u.Jy)
    except (ValueError, TypeError):
        flux_density = False
    return flux_density
