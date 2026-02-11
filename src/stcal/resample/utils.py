import logging
import math

import numpy as np
from astropy import units as u
from astropy.nddata.bitmask import (
    bitfield_to_boolean_mask,
    interpret_bit_flags,
)
from scipy import interpolate
from spherical_geometry.polygon import SphericalPolygon  # type: ignore[import-untyped]

__all__ = [
    "calc_pixmap",
    "build_driz_weight",
    "build_mask",
    "compute_mean_pixel_area",
    "get_tmeasure",
    "is_flux_density",
    "is_imaging_wcs",
    "resample_range",
]

log = logging.getLogger(__name__)


def calc_pixmap(wcs_from, wcs_to, shape=None, disable_bbox="to", stepsize=1, order=1):
    """
    Calculate pixel coordinates of one WCS corresponding to the native pixel grid of another WCS.

    .. note::
       This function assumes that output frames of ``wcs_from`` and ``wcs_to``
       WCS have the same units.

    Parameters
    ----------
    wcs_from : object
        A WCS object representing the coordinate system you are
        converting from. This object's ``array_shape`` (or ``pixel_shape``)
        property will be used to define the shape of the pixel map array.
        If ``shape`` parameter is provided, it will take precedence
        over this object's ``array_shape`` value.

    wcs_to : object
        A WCS object representing the coordinate system you are
        converting to.

    shape : tuple, None, optional
        A tuple of integers indicating the shape of the output array in the
        ``numpy.ndarray`` order. When provided, it takes precedence over the
        ``wcs_from.array_shape`` property.

    disable_bbox : str, optional
        Indicates whether to use or not to use the bounding box of either
        (both) ``wcs_from`` or (and) ``wcs_to`` when computing pixel map.
        Allowable values: "to", "from", "both", "none". When
        ``disable_bbox`` is "none", pixel coordinates outside of the bounding
        box are set to `NaN` only if ``wcs_from`` or (and) ``wcs_to`` sets
        world coordinates to NaN when input pixel coordinates are outside of
        the bounding box.

    stepsize : int, optional
        If ``stepsize>1``, perform the full WCS calculation on a sparser
        grid and use interpolation to fill in the rest of the pixels.  This
        option speeds up pixel map computation by reducing the number of WCS
        calls, though at the cost of reduced pixel map accuracy.  The loss
        of accuracy is typically negligible if the underlying distortion
        correction is smooth, but if the distortion is non-smooth,
        ``stepsize>1`` is not recommended.  Large ``stepsize`` values are
        automatically reduced to no more than 1/10 of image size.
        Default 1.

    order : int, optional
        Order of the 2D spline to interpolate the sparse pixel mapping
        if stepsize>1.  Supported values are: 1 (bilinear) or 3 (bicubic).
        This Parameter is ignored when ``stepsize <= 1``.  Default 1.

    Returns
    -------
    pixmap : numpy.ndarray
        A three dimensional array representing the transformation between
        the two. The last dimension is of length two and contains the x and
        y coordinates of a pixel center, respectively. The other two
        coordinates correspond to the two coordinates of the image the first
        WCS is from.

    Raises
    ------
    ValueError
        A `ValueError` is raised when output pixel map shape cannot be
        determined from provided inputs.

    Notes
    -----
    When ``shape`` is not provided and ``wcs_from.array_shape`` is not set
    (i.e., it is `None`), `calc_pixmap` will attempt to determine pixel map
    shape from the ``bounding_box`` property of the input ``wcs_from`` object.
    If ``bounding_box`` is not available, a `ValueError` will be raised.

    """
    if (bbox_from := getattr(wcs_from, "bounding_box", None)) is not None:
        try:
            # to avoid dependency on astropy just to check whether
            # the bounding box is an instance of
            # modeling.bounding_box.ModelBoundingBox, we try to
            # directly use and bounding_box(order='F') and if it fails,
            # fall back to converting the bounding box to a tuple
            # (of intervals):
            bbox_from = bbox_from.bounding_box(order="F")
        except AttributeError:
            bbox_from = tuple(bbox_from)

    if (bbox_to := getattr(wcs_to, "bounding_box", None)) is not None:
        try:
            # to avoid dependency on astropy just to check whether
            # the bounding box is an instance of
            # modeling.bounding_box.ModelBoundingBox, we try to
            # directly use and bounding_box(order='F') and if it fails,
            # fall back to converting the bounding box to a tuple
            # (of intervals):
            bbox_to = bbox_to.bounding_box(order="F")
        except AttributeError:
            bbox_to = tuple(bbox_to)

    if shape is None:
        shape = wcs_from.array_shape
        if shape is None and bbox_from is not None:
            if (ndim := np.ndim(bbox_from)) == 1:
                bbox_from = (bbox_from,)
            if ndim > 1:
                shape = tuple(math.ceil(lim[1] + 0.5) for lim in bbox_from[::-1])

    if shape is None:
        raise ValueError('The "from" WCS must have pixel_shape property set.')

    # temporarily disable the bounding box for the "from" WCS:
    if disable_bbox in ["from", "both"] and bbox_from is not None:
        wcs_from.bounding_box = None
    if disable_bbox in ["to", "both"] and bbox_to is not None:
        wcs_to.bounding_box = None

    # find integer boundaries of of the pixel grid to be computed:
    if bbox_from is None:
        xmin, xmax = 0, shape[1] - 1
        ymin, ymax = 0, shape[0] - 1
    else:
        ((xmin, xmax), (ymin, ymax)) = bbox_from
        xmin = max(0, int(math.ceil(xmin)))
        xmax = min(shape[1] - 1, int(math.floor(xmax)))
        ymin = max(0, int(math.ceil(ymin)))
        ymax = min(shape[0] - 1, int(math.floor(ymax)))

    eff_width = xmax - xmin + 1
    eff_height = ymax - ymin + 1

    try:
        if stepsize == 1 or max(eff_width, eff_height) <= 10:
            y, x = np.indices(shape, dtype=np.float64)
            x, y = wcs_to.world_to_pixel_values(*wcs_from.pixel_to_world_values(x, y))
        else:
            if order not in [1, 3]:
                raise ValueError("Interpolation order should be either 1 or 3.")

            # Because of the way RectBivariateSpline works, we need to pick
            # only those points that have finite coordinates. While using
            # bounding boxes does not guarantee that all points within them
            # will have finite world coordinates, it is the best we can do
            # without calling the WCS on all pixels.

            y_in, x_in = np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1)

            # Number of points so that step size is no larger than requested
            # but at least 10 points are used (or the number of input pixels
            # in each dimension, if smaller than that).
            npts_x = max(int(math.ceil(eff_width / stepsize)), min(10, eff_width))
            npts_y = max(int(math.ceil(eff_height / stepsize)), min(10, eff_height))

            x_coarse = np.linspace(xmin, xmax, npts_x)
            y_coarse = np.linspace(ymin, ymax, npts_y)

            sparsegrid = np.meshgrid(x_coarse, y_coarse)

            pixmap_coarse = wcs_to.world_to_pixel_values(
                *wcs_from.pixel_to_world_values(sparsegrid[0], sparsegrid[1])
            )

            if np.all(np.isfinite(pixmap_coarse[0])) and np.all(np.isfinite(pixmap_coarse[1])):
                fx = interpolate.RectBivariateSpline(y_coarse, x_coarse, pixmap_coarse[0], kx=order, ky=order)
                fy = interpolate.RectBivariateSpline(y_coarse, x_coarse, pixmap_coarse[1], kx=order, ky=order)

                # Evaluate the spline on the full grid
                x = fx(y_in, x_in)
                y = fy(y_in, x_in)

                if not (xmin == 0 and ymin == 0 and xmax == shape[1] - 1 and ymax == shape[0] - 1):
                    # we need to create a full grid and inject the interpolated
                    # values into it:
                    x_full = np.full(shape, np.nan)
                    y_full = np.full(shape, np.nan)
                    x_full[ymin : ymax + 1, xmin : xmax + 1] = x
                    y_full[ymin : ymax + 1, xmin : xmax + 1] = y
                    x, y = x_full, y_full

            else:
                # revert to the full WCS calculation if there are any
                # non-finite values in the coarse grid:
                y, x = np.indices(shape, dtype=np.float64)
                x, y = wcs_to.world_to_pixel_values(*wcs_from.pixel_to_world_values(x, y))

    finally:
        if bbox_from is not None:
            wcs_from.bounding_box = bbox_from
        if bbox_to is not None:
            wcs_to.bounding_box = bbox_to

    pixmap = np.dstack([x, y])
    return pixmap


def resample_range(data_shape, bbox=None):  # noqa: D103
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
    """Build a bit mask from an input DQ array and a bitvalue flag.

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


def build_driz_weight(model, weight_type=None, good_bits=None, flag_name_map=None):
    """
    Build drizzle weight map.

    Create a weight map that is used for weighting input images when
    they are co-added to the output model.

    Parameters
    ----------
    model : dict
        Input model: a dictionary of relevant keywords and values.

    weight_type : str or None, optional
        The weighting type ("exptime", "ivm", or "ivm-sky") for adding models' data.
        For ``weight_type="ivm"`` and ``weight_type="ivm-sky"``,
        the weighting will be determined
        per-pixel using the inverse of either the read noise (VAR_RNOISE) or
        sky variance (VAR_SKY) arrays, respectively. If the array does not
        exist, the weight is set to 1 for all pixels (i.e., equal weighting).
        If ``weight_type="exptime"``, the weight will be set equal to the
        measurement time when available and to the exposure time otherwise for
        pixels not flagged in the DQ array of the model. The default value of
        `None` will set weights to 1 for pixels not flagged in the DQ array of
        the model. Pixels flagged as "bad" in the DQ array will have their
        weights set to 0.

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
        that are comma- or '+'-separated list of mnemonic bit flag names.

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

    if weight_type == "ivm":
        inv_variance = _get_inverse_variance(
            model["var_rnoise"] if "var_rnoise" in model else None,
            data.shape,
            "var_rnoise",
        )
        result = inv_variance * dqmask

    elif weight_type == "ivm-sky":
        inv_sky_variance = _get_inverse_variance(
            model["var_sky"] if "var_sky" in model else None,
            data.shape,
            "var_sky",
        )
        result = inv_sky_variance * dqmask

    elif weight_type == "exptime":
        exptime, _ = get_tmeasure(model)
        result = np.float32(exptime) * dqmask

    elif weight_type is None:
        result = dqmask

    else:
        raise ValueError(
            f"Invalid weight type: {repr(weight_type)}."
            "Allowed weight types are 'ivm', 'ivm-sky', 'exptime', or None."
        )

    return result.astype(np.float32)


def get_tmeasure(model):
    """
    Get tmeasure from datamodel.

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
    """Return `True` if ``wcs`` is an imaging WCS and `False` otherwise."""
    imaging = all(ax == "SPATIAL" for ax in wcs.output_frame.axes_type)
    return imaging


def compute_mean_pixel_area(wcs, shape=None):
    """
    Compute mean pixel area.

    Computes the average pixel area (in steradians) based on input WCS
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
            "Either WCS must have 'array_shape' attribute set or 'shape' argument must be supplied."
        )

    valid_polygon = False
    spatial_idx = np.where(np.array(wcs.output_frame.axes_type) == "SPATIAL")[0]

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
                dy=min((ymax - ymin) // 4, 15),
            )
        except ValueError:
            return None

        world = wcs(x, y)
        ra = world[spatial_idx[0]]
        dec = world[spatial_idx[1]]

        limits = [ymin, xmax, ymax, xmin]

        for _ in range(4):
            sl = [b, r, t, l][k]
            if not (np.all(np.isfinite(ra[sl])) and np.all(np.isfinite(dec[sl]))):
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
        log.warning("Unexpectedly large computed sky area for an image. Setting area to: 4*Pi - area")
        sky_area = 4 * np.pi - sky_area
    if image_area == 0:
        log.error("Image area is zero; cannot compute pixel area.")
        return None
    pix_area = sky_area / image_area

    return pix_area


def _get_boundary_points(xmin, xmax, ymin, ymax, dx=None, dy=None, shrink=0):  # noqa: E741
    """
    Get boundary points.

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
        Desired spacing between adjacent points along horizontal edges of
        the rectangle.

    dy : int, float, None, optional
        Desired spacing between adjacent points along vertical edges of
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

    if dx is None or dx <= 0:
        dx = nx
    if dy is None or dy <= 0:
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
    right = np.s_[sx : sx + sy]  # right edge
    top = np.s_[sx + sy : 2 * sx + sy]  # top edge
    left = np.s_[2 * sx + sy : 2 * sx + 2 * sy]  # noqa: E741  left edge

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


def _get_inverse_variance(array, data_shape, array_name):
    """
    Compute the inverse variance array for weighting.

    Parameters
    ----------
    array : numpy.ndarray or None
        Input variance array.
    data_shape : tuple
        Expected shape of the output array.

    Returns
    -------
    inv : numpy.ndarray
        Inverse variance array. If input array is missing or has wrong shape,
        returns an array filled with ones of shape `data_shape`.  Otherwise the inverse of the variance
        array is returned, with invalid values replaced by zeros.
    """
    if array is not None and array.shape == data_shape:
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / array
        inv[~np.isfinite(inv)] = 0  # zeros for bad pixels
    else:
        log.warning(
            f"'{array_name}' array not available. Setting drizzle weight map to 1",
        )
        inv = np.full(data_shape, 1, dtype=np.float32)  # ones for missing/misshaped array

    return inv
