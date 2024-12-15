from copy import deepcopy
import logging

import asdf
import numpy as np
from astropy.nddata.bitmask import interpret_bit_flags
from spherical_geometry.polygon import SphericalPolygon


__all__ = [
    "build_mask",
    "bytes2human",
    "compute_mean_pixel_area",
    "get_tmeasure",
    "is_imaging_wcs",
    "load_custom_wcs",
    "resample_range",
]

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def resample_range(data_shape, bbox=None):
    # Find range of input pixels to resample:
    if bbox is None:
        xmin = ymin = 0
        xmax = data_shape[1] - 1
        ymax = data_shape[0] - 1
    else:
        ((x1, x2), (y1, y2)) = bbox
        xmin = max(0, int(x1 + 0.5))
        ymin = max(0, int(y1 + 0.5))
        xmax = min(data_shape[1] - 1, int(x2 + 0.5))
        ymax = min(data_shape[0] - 1, int(y2 + 0.5))

    return xmin, xmax, ymin, ymax


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
    wcs : ~gwcs.wcs.WCS
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

    Taken from the ``psutil`` library which references
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
