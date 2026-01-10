import logging
import math

import numpy as np
import shapely.geometry  # type: ignore[import-untyped]
from scipy import interpolate

from stcal.alignment import util

log = logging.getLogger(__name__)


def calc_pixmap(wcs_from, wcs_to, shape=None, disable_bbox="to", stepsize=1, order=1):
    """
    Calculate pixel coordinates of one WCS corresponding to the native pixel grid of another WCS.

    .. note::
       This function assumes that output frames of ``wcs_from`` and ``wcs_to``
       WCS have the same units.

    Parameters
    ----------
    wcs_from : wcs
        A WCS object representing the coordinate system you are
        converting from. This object's ``array_shape`` (or ``pixel_shape``)
        property will be used to define the shape of the pixel map array.
        If ``shape`` parameter is provided, it will take precedence
        over this object's ``array_shape`` value.

    wcs_to : wcs
        A WCS object representing the coordinate system you are
        converting to.

    shape : tuple, None, optional
        A tuple of integers indicating the shape of the output array in the
        ``numpy.ndarray`` order. When provided, it takes precedence over the
        ``wcs_from.array_shape`` property.

    disable_bbox : {"to", "from", "both", "none"}, optional
        Indicates whether to use or not to use the bounding box of either
        (both) ``wcs_from`` or (and) ``wcs_to`` when computing pixel map. When
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
        y coordinates of a pixel center, respectively. The other two coordinates
        correspond to the two coordinates of the image the first WCS is from.

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
    try:
        if stepsize == 1:
            y, x = np.indices(shape, dtype=np.float64)
            x, y = wcs_to.world_to_pixel_values(*wcs_from.pixel_to_world_values(x, y))
        else:
            if order not in [1, 3]:
                raise ValueError("Interpolation order should be either 1 or 3.")

            y_in, x_in = np.arange(shape[0]), np.arange(shape[1])

            # Number of points so that step size is no larger than requested
            # but at least 10 points are used (or the number of input pixels
            # in each dimension, if smaller than that).

            npts_x = max(int(math.ceil(shape[1] / stepsize)), min(10, shape[1]))
            npts_y = max(int(math.ceil(shape[0] / stepsize)), min(10, shape[0]))

            x_coarse = np.linspace(0, x_in[-1], npts_x)
            y_coarse = np.linspace(0, y_in[-1], npts_y)

            sparsegrid = np.meshgrid(x_coarse, y_coarse)

            pixmap_coarse = wcs_to.world_to_pixel_values(
                *wcs_from.pixel_to_world_values(sparsegrid[0], sparsegrid[1])
            )

            fx = interpolate.RectBivariateSpline(x_coarse, y_coarse, pixmap_coarse[0], kx=order, ky=order)
            fy = interpolate.RectBivariateSpline(x_coarse, y_coarse, pixmap_coarse[1], kx=order, ky=order)

            # Evaluate the spline on the full grid

            x = fx(x_in, y_in)
            y = fy(x_in, y_in)

    finally:
        if bbox_from is not None:
            wcs_from.bounding_box = bbox_from
        if bbox_to is not None:
            wcs_to.bounding_box = bbox_to

    pixmap = np.dstack([x, y])
    return pixmap


def combine_sregions(sregion_list, det2world, intersect_footprint=None):
    """
    Combine s_regions from input models to compute the s_region for the resampled data.

    Parameters
    ----------
    sregion_list : list[str] or list[np.ndarray]
        List of s_regions from input models. If an element is a string,
        it will be converted to a footprint using `util.sregion_to_footprint`.
        If an element is already a footprint (2-D array of shape (N, 2)),
        it will be used directly.
    det2world : `~astropy.modeling.Model`
        WCS detector-to-world transform for the resampled data.
        Must take in exactly two inputs (x, y) and return exactly two outputs (RA, Dec).
        Must have a valid inverse transform.
    intersect_footprint : np.ndarray, optional
        Footprint of the output WCS in world coordinates, shape (N, 2).
        If provided, the combined footprint from the input s_region list
        will be intersected with this footprint.

    Returns
    -------
    str
        The combined s_region.

    Raises
    ------
    ValueError
        If there is no overlap between the input s_regions and the intersection footprint.
    """
    footprints = np.array(
        [
            util.sregion_to_footprint(sregion) if isinstance(sregion, str) else sregion
            for sregion in sregion_list
        ]
    )

    # convert from world to pixel coordinates
    footprints_flat = footprints.reshape(-1, 2)
    world2det = det2world.inverse
    x, y = world2det(footprints_flat[:, 0], footprints_flat[:, 1])
    footprints_pixels = np.vstack([x, y]).T.reshape(footprints.shape)

    # combine footprints with Shapely
    combined_polygons = combine_footprints(footprints_pixels)

    # intersect with output WCS footprint
    if intersect_footprint is not None:
        x, y = world2det(intersect_footprint[:, 0], intersect_footprint[:, 1])
        intersect_footprint_pixels = np.vstack([x, y]).T
        final_polygons = _intersect_with_bbox(combined_polygons, intersect_footprint_pixels)
        if not final_polygons:
            raise ValueError("No overlap between input s_regions and intersection footprint")
    else:
        final_polygons = combined_polygons

    # convert back from pixel to world coordinates
    combined_polygons_world = []
    for polygon in final_polygons:
        ra, dec = det2world(polygon[:, 0], polygon[:, 1])
        combined_polygons_world.append(np.vstack([ra, dec]).T)

    # turn lists of indices into a single S_REGION string
    sregion = _polygons_to_sregion(combined_polygons_world)

    return sregion


def _polygons_to_sregion(polygons):
    """
    Create an S_REGION from a list of polygons.

    Parameters
    ----------
    polygons : list[np.ndarray]
        List of polygons. Each polygon should have shape (V, 2), where V is the number of vertices.
        V can be different for each polygon.

    Returns
    -------
    str
        S_REGION string.
    """
    s_region = ""
    for poly in polygons:
        poly = " ".join(f"{x:.9f} {y:.9f}" for x, y in poly)
        s_region += f"POLYGON ICRS  {poly}  "
    return s_region.strip()


def _convert_to_array(polygon):
    """
    Convert a Shapely polygon to a numpy array and simplify it to minimize number of vertices.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Input polygon.

    Returns
    -------
    np.ndarray
        2D array of shape (N, 2) representing the polygon vertices.
    """
    x, y = polygon.exterior.coords.xy
    poly_out = np.vstack([x, y]).T
    poly_out = poly_out[:-1]  # remove duplicate last point
    return _simplify_by_angle(poly_out)


def combine_footprints(footprints):
    """
    Combine a list of footprints into one or more combined footprints using a Shapely union.

    Parameters
    ----------
    footprints : list of np.ndarray
        List of footprints, where each footprint is a 2-D array of shape (N, 2).

    Returns
    -------
    list of np.ndarray
        List of combined footprints, where each footprint is a 2-D array of shape (M, 2).
    """
    footprints_shapely = [shapely.geometry.Polygon(footprint) for footprint in footprints]
    combined_footprints = shapely.unary_union(footprints_shapely)
    if isinstance(combined_footprints, shapely.geometry.Polygon):
        combined_footprints = [combined_footprints]
    elif isinstance(combined_footprints, shapely.geometry.MultiPolygon):
        combined_footprints = combined_footprints.geoms
    combined_polys = []
    for poly in combined_footprints:
        combined_polys.append(_convert_to_array(poly))
    return combined_polys


def _intersect_with_bbox(polygons, bbox):
    """
    Intersect a list of polygons with a bounding box.

    Parameters
    ----------
    polygons : list[np.ndarray]
        List of polygons. Each polygon should have shape (V, 2), where V is the number of vertices.
    bbox : np.ndarray
        2D array of shape (N, 2) representing the bounding box vertices.

    Returns
    -------
    np.ndarray
        2D array of shape (M, 2) representing the intersected polygon vertices.
    """
    intersect_polygon = shapely.geometry.Polygon(bbox)
    final_polygons = []
    for polygon in polygons:
        polygon = shapely.geometry.Polygon(polygon)
        intersection = shapely.intersection(polygon, intersect_polygon)
        if not intersection.is_empty:
            final_polygons.append(_convert_to_array(intersection))
    return final_polygons


def _simplify_by_angle(coords, point_thresh=5e-2, angle_thresh=1e-6):
    """
    Simplify a polygon by removing points that are collinear with their neighbors.

    MAST has a check for duplicated points in their code which is set to 1e-7 deg.
    This code is meant to be used in pixel space; for NIRCam, 1 pixel is about 0.031 arcsec,
    and 1e-7 deg is about 0.36 mas or a factor of 100 smaller.
    So we need to make the tolerance bigger than that: 1/10 pixel makes sense.
    This doesn't matter for the angle threshold

    Parameters
    ----------
    coords : np.ndarray
        2D array of shape (N, 2) representing the polygon vertices.
    point_thresh : float, optional
        Threshold for considering two points to be the same (in pixels).
    angle_thresh : float, optional
        Threshold for considering three points to be collinear (in radians).

    Returns
    -------
    np.ndarray
        2D array of shape (M, 2) representing the simplified polygon vertices.
    """
    # Indices for previous, current, next points (wrap around)
    n = len(coords)
    idx_prev = np.arange(-1, n - 1)
    idx_curr = np.arange(n)
    idx_next = np.arange(1, n + 1) % n

    p0 = coords[idx_prev]
    p1 = coords[idx_curr]
    p2 = coords[idx_next]

    # Vectors
    v1 = p1 - p0
    v2 = p2 - p1

    # Check closeness
    # We only need to check closeness in one direction, to avoid removing both points
    # if there's a duplicated pair. Wrapped indices ensure no bugs with first/last points.
    close = (np.abs(v1[:, 0]) < point_thresh) & (np.abs(v1[:, 1]) < point_thresh)

    # Slopes
    m1 = v1[:, 1] / (v1[:, 0] + 1e-12)
    m2 = v2[:, 1] / (v2[:, 0] + 1e-12)
    delta_theta = np.arctan((m2 - m1) / (1 + m1 * m2))

    collinear = close | (np.abs(delta_theta) < angle_thresh)

    # Keep only non-collinear points
    return coords[~collinear]
